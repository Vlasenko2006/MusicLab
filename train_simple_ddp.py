#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP Training for Simple Transformer on WAV Pairs

Multi-GPU training using DistributedDataParallel
EnCodec encoder/decoder are frozen
"""

import os
import argparse
import torch.multiprocessing as mp

from train_simple_worker import train_worker


def main():
    parser = argparse.ArgumentParser(description='Train Simple Transformer on WAV Pairs')
    
    # Model parameters
    parser.add_argument('--encoding_dim', type=int, default=128,
                       help='EnCodec encoding dimension (128 for 24kHz)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of internal transformer encoder layers per stage')
    parser.add_argument('--num_transformer_layers', type=int, default=1,
                       help='Number of cascade stages (1=no cascade, 2+=cascade refinement)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # EnCodec parameters
    parser.add_argument('--encodec_bandwidth', type=float, default=6.0,
                       help='EnCodec bandwidth (6.0 for high quality)')
    parser.add_argument('--encodec_sr', type=int, default=24000,
                       help='EnCodec sample rate')
    
    # Dataset parameters
    parser.add_argument('--dataset_folder', type=str, required=True,
                       help='Dataset folder (contains train/ and val/ subdirs)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers (0 for GPU encoding)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_simple',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # DDP parameters
    parser.add_argument('--world_size', type=int, default=4,
                       help='Number of GPUs')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0,
                       help='Local rank for distributed training (automatically set by torch.distributed.launch)')
    
    # Unity test parameter
    parser.add_argument('--unity_test', type=lambda x: x.lower() == 'true', default=False,
                       help='Unity test: set target=input for sanity check (true/false)')
    
    # Target shuffling parameter
    parser.add_argument('--shuffle_targets', type=lambda x: x.lower() == 'true', default=False,
                       help='Shuffle targets: random target instead of continuation (true/false)')
    
    # Anti-cheating parameter (for cascade stages 2+)
    parser.add_argument('--anti_cheating', type=float, default=0.0,
                       help='Anti-cheating noise level for cascade (0.0=no noise, 1.0=heavy noise on target)')
    
    # Loss weight parameters
    parser.add_argument('--loss_weight_input', type=float, default=0.0,
                       help='Weight for RMS(predicted, input) loss component')
    parser.add_argument('--loss_weight_target', type=float, default=1.0,
                       help='Weight for RMS(predicted, target) loss component')
    parser.add_argument('--loss_weight_spectral', type=float, default=0.0,
                       help='Weight for multi-resolution STFT loss component')
    parser.add_argument('--loss_weight_mel', type=float, default=0.0,
                       help='Weight for mel-spectrogram loss component')
    
    # Complementary masking parameters
    parser.add_argument('--mask_type', type=str, default='none',
                       choices=['none', 'temporal', 'frequency', 'spectral', 'energy', 'hybrid'],
                       help='Complementary masking strategy for style transfer')
    parser.add_argument('--mask_temporal_segment', type=int, default=150,
                       help='Temporal segment length in frames (~150 = 1 second)')
    parser.add_argument('--mask_freq_split', type=float, default=0.3,
                       help='Frequency split ratio (0.3 = low 30%% vs high 70%%)')
    parser.add_argument('--mask_channel_keep', type=float, default=0.5,
                       help='Channel keep ratio for spectral dropout')
    parser.add_argument('--mask_energy_threshold', type=float, default=0.7,
                       help='Energy threshold percentile (0.7 = 70th percentile)')
    
    # Creative agent parameters (learnable masking)
    parser.add_argument('--use_creative_agent', type=lambda x: x.lower() == 'true', default=False,
                       help='Use learnable creative agent instead of fixed masking (true/false)')
    parser.add_argument('--use_compositional_agent', type=lambda x: x.lower() == 'true', default=False,
                       help='Use compositional agent (rhythm/harmony/timbre decomposition) - mutually exclusive with use_creative_agent')
    parser.add_argument('--mask_reg_weight', type=float, default=0.1,
                       help='Weight for mask regularization loss (complementarity + coverage) or novelty loss')
    parser.add_argument('--balance_loss_weight', type=float, default=5.0,
                       help='Weight for balance loss (enforces 50/50 input/target mixing). Higher = stronger enforcement. Typical: 5-10')
    parser.add_argument('--corr_weight', type=float, default=0.0,
                       help='Weight for anti-modulation correlation cost (prevents copying amplitude envelopes)')
    
    # GAN parameters
    parser.add_argument('--gan_weight', type=float, default=0.0,
                       help='Weight for GAN adversarial loss (0.0=disabled, 0.01-0.1 typical)')
    parser.add_argument('--disc_lr', type=float, default=None,
                       help='Discriminator learning rate (default: 0.5 * lr for stability)')
    parser.add_argument('--disc_update_freq', type=int, default=1,
                       help='Update discriminator every N batches (1=every batch, higher for stability)')
    
    # Pure GAN mode (curriculum learning from music-to-music → noise-to-music)
    parser.add_argument('--pure_gan_mode', type=float, default=0.0,
                       help='Curriculum learning rate: transition from music to noise. 0.0=disabled, 0.01=full noise after 100 epochs')
    parser.add_argument('--gan_curriculum_start_epoch', type=int, default=0,
                       help='Epoch to start GAN curriculum (allows pre-training on music before transitioning to noise)')
    parser.add_argument('--gan_noise_ceiling', type=float, default=1.0,
                       help='Maximum noise fraction (alpha ceiling). Use to freeze noise level, e.g., 0.3 = freeze at 30%% noise')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("="*80)
    print("SIMPLE TRANSFORMER TRAINING ON WAV PAIRS")
    print("="*80)
    print("Configuration:")
    print(f"  Dataset: {args.dataset_folder}")
    print(f"  Model:")
    print(f"    - Encoding dim: {args.encoding_dim}")
    print(f"    - Attention heads: {args.nhead}")
    print(f"    - Internal transformer layers: {args.num_layers}")
    print(f"    - Cascade stages: {args.num_transformer_layers}")
    print(f"    - Dropout: {args.dropout}")
    print(f"  EnCodec:")
    print(f"    - Sample rate: {args.encodec_sr} Hz")
    print(f"    - Bandwidth: {args.encodec_bandwidth}")
    print(f"    - Status: FROZEN (encoder + decoder)")
    print(f"  Training:")
    print(f"    - Epochs: {args.epochs}")
    print(f"    - Batch size: {args.batch_size} per GPU × {args.world_size} GPUs = {args.batch_size * args.world_size}")
    print(f"    - Learning rate: {args.lr}")
    print(f"    - Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"    - Loss: Combined perceptual loss")
    print(f"      * Input weight: {args.loss_weight_input} (RMS reconstruction)")
    print(f"      * Target weight: {args.loss_weight_target} (RMS continuation)")
    print(f"      * Spectral weight: {args.loss_weight_spectral} (multi-resolution STFT)")
    print(f"      * Mel weight: {args.loss_weight_mel} (mel-spectrogram)")
    if args.gan_weight > 0:
        print(f"      * GAN weight: {args.gan_weight} (adversarial loss)")
    print(f"    - Unity test: {'ENABLED (target=input)' if args.unity_test else 'DISABLED'}")
    print(f"    - Shuffle targets: {'ENABLED (random pairs)' if args.shuffle_targets else 'DISABLED (matched pairs)'}")
    if args.mask_type != 'none':
        print(f"  Complementary Masking:")
        print(f"    - Type: {args.mask_type}")
        if args.mask_type == 'temporal':
            print(f"    - Segment length: {args.mask_temporal_segment} frames (~{args.mask_temporal_segment/75:.1f}s)")
        elif args.mask_type in ['frequency', 'hybrid']:
            print(f"    - Frequency split: {args.mask_freq_split:.0%} low / {1-args.mask_freq_split:.0%} high")
        elif args.mask_type == 'spectral':
            print(f"    - Channel keep: {args.mask_channel_keep:.0%} per source")
        elif args.mask_type == 'energy':
            print(f"    - Energy threshold: {args.mask_energy_threshold:.0%} percentile")
    if args.use_compositional_agent:
        print(f"  🎼 Compositional Creative Agent:")
        print(f"    - Enabled: True (rhythm/harmony/timbre decomposition)")
        print(f"    - Novelty regularization weight: {args.mask_reg_weight}")
        print(f"    - Extracts: rhythm (3-5 kernel), harmony (7-9 kernel), timbre (15-21 kernel)")
        print(f"    - Composer: 4-layer transformer with 8 heads")
    elif args.use_creative_agent:
        print(f"  🎨 Attention-Based Creative Agent:")
        print(f"    - Enabled: True (learnable masking)")
        print(f"    - Mask regularization weight: {args.mask_reg_weight}")
    if args.gan_weight > 0:
        disc_lr = args.disc_lr if args.disc_lr is not None else args.lr * 0.5
        print(f"  GAN Training:")
        print(f"    - Enabled: True (adversarial training)")
        print(f"    - Discriminator LR: {disc_lr:.2e}")
        print(f"    - Discriminator update frequency: every {args.disc_update_freq} batch(es)")
    print(f"  Checkpoints: {args.checkpoint_dir}/")
    print("="*80)
    print("="*80)
    
    # Launch DDP training
    mp.spawn(
        train_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()
