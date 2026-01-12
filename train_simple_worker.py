#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP Training Worker for Simple Transformer

Each worker runs on one GPU
"""

import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*git.*')
warnings.filterwarnings('ignore', message='.*No device id is provided.*')  # Suppress DDP device warnings

# Suppress git warnings from MLflow
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
import sys
import signal
from tqdm import tqdm
import encodec

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("Warning: mlflow not installed. Logging disabled.")

import model_simple_transformer
from model_simple_transformer import SimpleTransformer
from dataset_wav_pairs import WavPairsDataset
from complementary_masking import apply_complementary_mask, get_mask_description
from audio_discriminator import AudioDiscriminator, discriminator_loss, generator_loss, roll_targets
from training.losses import rms_loss, stft_loss, mel_loss, combined_loss, spectral_outlier_penalty


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    # MASTER_PORT should be set by launch script, don't override it here
    # If not set, use default (but this should never happen)
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
        if rank == 0:
            print(f"WARNING: MASTER_PORT not set, using default 29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()


def load_encodec_model(bandwidth=6.0, sample_rate=24000, device='cuda'):
    """Load and freeze EnCodec model"""
    model = encodec.EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model = model.to(device)
    
    # Keep in train mode to allow gradients to flow through
    model.train()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def train_epoch(model, encodec_model, dataloader, optimizer, device, rank, epoch, 
                unity_test=False, loss_weight_input=0.0, loss_weight_target=1.0, 
                loss_weight_spectral=0.0, loss_weight_mel=0.0,
                mask_type='none', mask_temporal_segment=150, mask_freq_split=0.3,
                mask_channel_keep=0.5, mask_energy_threshold=0.7, mask_reg_weight=0.1,
                balance_loss_weight=5.0,
                discriminator=None, disc_optimizer=None, gan_weight=0.0, disc_update_freq=1,
                corr_weight=0.0, pure_gan_mode=0.0, gan_curriculum_counter=0, gan_noise_ceiling=1.0):
    """Train for one epoch with optional GAN training."""
    model.train()
    if discriminator is not None:
        discriminator.train()
    
    total_loss = 0.0
    total_rms_input = 0.0
    total_rms_target = 0.0
    total_spectral = 0.0
    total_mel = 0.0
    total_corr_penalty = 0.0  # Track correlation penalty
    total_mask_reg_loss = 0.0
    total_balance_loss_raw = 0.0  # Track separate balance loss
    total_input_mask_mean = 0.0
    total_target_mask_mean = 0.0
    total_mask_overlap = 0.0
    # Creative agent detailed loss components
    total_balance_loss = 0.0
    total_temporal_diversity = 0.0
    total_complementarity = 0.0
    total_coverage = 0.0
    # Compositional agent statistics
    total_input_rhythm_w = 0.0
    total_input_harmony_w = 0.0
    total_target_rhythm_w = 0.0
    total_target_harmony_w = 0.0
    total_gan_loss = 0.0
    total_disc_loss = 0.0
    total_disc_real_acc = 0.0
    total_disc_fake_acc = 0.0
    num_batches = 0
    
    if rank == 0:
        # Force tqdm to work properly even when output is piped to tee
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", 
                   ncols=120, leave=True, position=0,
                   file=sys.stdout, mininterval=2.0, ascii=True,
                   dynamic_ncols=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = dataloader
    
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)  # Now raw audio [B, 1, samples]
        
        # Debug: Print input statistics on first batch
        if num_batches == 0 and rank == 0:
            # Enable detailed numerical debugging for first batch of specific epochs
            if epoch in [1, 2, 3, 4, 5, 8]:
                print(f"\n🔍 ENABLING DETAILED NUMERICAL DEBUGGING FOR EPOCH {epoch}, BATCH 0")
                model_simple_transformer.DEBUG_NUMERICS = True
            
            print(f"\n📊 Encoded input statistics:")
            print(f"  Shape: {inputs.shape}")
            print(f"  Min: {inputs.min().item():.4f}")
            print(f"  Max: {inputs.max().item():.4f}")
            print(f"  Mean: {inputs.mean().item():.4f}")
            print(f"  Std: {inputs.std().item():.4f}")
            print(f"  RMS: {torch.sqrt((inputs**2).mean()).item():.4f}")
            print(f"\n📊 Target audio statistics:")
            print(f"  Shape: {targets.shape}")
            print(f"  Min: {targets.min().item():.4f}")
            print(f"  Max: {targets.max().item():.4f}\n")
        
        # Unity test: encode target audio, substitute with input encoding
        if unity_test:
            with torch.no_grad():
                # For unity test, we want output_audio = input_audio
                # Decode inputs to get input_audio, use as target
                input_audio = encodec_model.decoder(inputs)  # inputs is [B, D, T], decoder outputs [B, 1, samples]
                targets = input_audio
        
        # Store original clean inputs/targets for loss computation
        original_inputs = inputs.clone()
        original_targets = targets.clone()
        
        # Pure GAN mode: Curriculum learning from music-to-music → noise-to-music
        if pure_gan_mode > 0:
            # Compute interpolation coefficient: α = min(ceiling, pure_gan_mode × counter)
            # pure_gan_mode: rate of transition (e.g., 0.01 = full noise after 100 epochs)
            # counter: cumulative epochs since start of curriculum
            # gan_noise_ceiling: maximum alpha value (e.g., 0.3 = freeze at 30% noise)
            alpha = min(gan_noise_ceiling, pure_gan_mode * gan_curriculum_counter)
            
            if alpha > 0:
                with torch.no_grad():
                    # Generate noise with same statistics as real data (PER-SAMPLE)
                    # inputs: [B, D, T], compute std per sample along (D, T) dimensions
                    input_std = inputs.std(dim=(1, 2), keepdim=True)  # [B, 1, 1]
                    target_std = targets.std(dim=(1, 2), keepdim=True)  # [B, 1, 1]
                    
                    input_noise = torch.randn_like(inputs) * input_std
                    target_noise = torch.randn_like(targets) * target_std
                    
                    # Interpolate: (1-α) × real + α × noise
                    inputs = (1.0 - alpha) * inputs + alpha * input_noise
                    targets = (1.0 - alpha) * targets + alpha * target_noise
                    
                    # Debug: Print curriculum status on first batch
                    if num_batches == 0 and rank == 0:
                        print(f"\n🎲 Pure GAN Curriculum Learning:")
                        print(f"  Alpha: {alpha:.4f} (0=music, 1=noise)")
                        print(f"  Input:  {(1-alpha)*100:.1f}% music + {alpha*100:.1f}% noise")
                        print(f"  Target: {(1-alpha)*100:.1f}% music + {alpha*100:.1f}% noise")
                        if alpha >= gan_noise_ceiling:
                            if gan_noise_ceiling < 1.0:
                                print(f"  🔒 Noise ceiling reached ({gan_noise_ceiling*100:.0f}% max noise)")
                            else:
                                print(f"  ✓ Full noise mode achieved!")
                        print()
        
        # Get input audio for combined loss (using ORIGINAL clean inputs)
        with torch.no_grad():
            input_audio = encodec_model.decoder(original_inputs)  # [B, 1, samples]
            
            # For cascade mode, encode target audio (potentially noisy for model)
            if model.module.num_transformer_layers > 1:
                encoded_target = encodec_model.encoder(targets)  # [B, D, T_enc] - may be noisy
                
                # Apply complementary masking if enabled
                if mask_type != 'none':
                    inputs, encoded_target = apply_complementary_mask(
                        inputs, encoded_target, 
                        mask_type=mask_type,
                        temporal_segment_frames=mask_temporal_segment,
                        freq_split_ratio=mask_freq_split,
                        channel_keep_ratio=mask_channel_keep,
                        energy_threshold=mask_energy_threshold
                    )
                    
                    # Debug: Print masking info on first batch
                    if num_batches == 0 and rank == 0:
                        print(f"🎭 Complementary Masking Applied:")
                        print(f"  Type: {mask_type}")
                        print(f"  Description: {get_mask_description(mask_type)}")
                        if mask_type == 'temporal':
                            print(f"  Segment length: {mask_temporal_segment} frames (~{mask_temporal_segment/75:.1f}s)")
                        elif mask_type in ['frequency', 'hybrid']:
                            print(f"  Frequency split: {mask_freq_split:.1%} low / {1-mask_freq_split:.1%} high")
                        elif mask_type == 'spectral':
                            print(f"  Channel keep ratio: {mask_channel_keep:.1%}")
                        elif mask_type == 'energy':
                            print(f"  Energy threshold: {mask_energy_threshold:.1%} percentile")
                        print()
            else:
                encoded_target = None
        
        # Forward pass through transformer (returns denormalized encoded output and optional mask/balance loss)
        if encoded_target is not None:
            encoded_output, mask_reg_loss, balance_loss = model(inputs, encoded_target)  # Cascade mode: [B, D, T_enc], optional losses
        else:
            encoded_output, mask_reg_loss, balance_loss = model(inputs), None, None  # Single stage mode: [B, D, T_enc], None
        
        # Decode to audio space for loss computation
        output_audio = encodec_model.decoder(encoded_output)  # [B, 1, samples]
        
        # Combined loss with all components (using ORIGINAL clean inputs/targets)
        # Important: Loss always compares to real audio, not noise
        loss, rms_input, rms_target, spectral, mel_value, corr_penalty = combined_loss(
            output_audio, input_audio, original_targets,  # Use original_targets, not noisy 
            loss_weight_input, loss_weight_target, 
            loss_weight_spectral, loss_weight_mel,
            weight_correlation=corr_weight
        )
        
        # Add mask regularization loss if creative agent is being used
        if mask_reg_loss is not None:
            loss = loss + mask_reg_weight * mask_reg_loss
            # Note: Statistics will be printed at epoch summary, not per batch
            # to avoid cluttering the progress bar
        
        # Add SEPARATE balance loss with MUCH HIGHER weight
        # Balance loss enforces 50/50 mixing and must be strong enough to overcome
        # the reconstruction loss bias toward copying the target
        # Weight configurable via --balance_loss_weight (default: 5.0)
        # Typical range: 5-10 (higher = stronger 50/50 enforcement)
        if balance_loss is not None:
            loss = loss + balance_loss_weight * balance_loss
        
        # Add spectral outlier penalty (suppress artifacts at 2kHz, 4kHz, 6kHz)
        spectral_penalty = spectral_outlier_penalty(
            output_audio, 
            sample_rate=24000,
            freq_bands=[(1800, 2200), (3800, 4200), (5800, 6200)]
        )
        spectral_penalty_weight = 0.01  # Small weight to avoid dominating other losses
        loss = loss + spectral_penalty_weight * spectral_penalty
        
        # DIAGNOSTIC: Compute how much output correlates with input vs target
        # This reveals if the model is actually mixing or just copying one source
        with torch.no_grad():
            # Flatten to [B*T] for correlation
            output_flat = output_audio.reshape(-1)
            input_flat = input_audio.reshape(-1)
            target_flat = targets.reshape(-1)
            
            # Correlations (range: -1 to 1)
            output_input_corr = torch.corrcoef(torch.stack([output_flat, input_flat]))[0, 1]
            output_target_corr = torch.corrcoef(torch.stack([output_flat, target_flat]))[0, 1]
            
            # Energy ratios (how much energy comes from each source)
            # If output = α*input + β*target, these approximate α and β
            input_energy = (input_flat ** 2).mean()
            target_energy = (target_flat ** 2).mean()
            output_energy = (output_flat ** 2).mean()
            
            # Store for logging
            output_input_corr_val = output_input_corr.item()
            output_target_corr_val = output_target_corr.item()
            energy_ratio = (output_energy / (input_energy + target_energy + 1e-8)).item()
        
        # Anti-modulation correlation cost (prevent copying amplitude envelopes)
        # GAN training: Discriminator + Generator adversarial loss
        gan_g_loss = torch.tensor(0.0).to(device)
        disc_loss_value = torch.tensor(0.0).to(device)
        disc_real_acc = 0.0
        disc_fake_acc = 0.0
        
        if discriminator is not None and gan_weight > 0:
            # Step 1: Train Discriminator (every disc_update_freq batches)
            if num_batches % disc_update_freq == 0:
                disc_optimizer.zero_grad()
                
                # Real samples: roll targets by 1 to create musical examples
                real_samples = roll_targets(encoded_target.detach())  # [B, D, T]
                real_logits = discriminator(real_samples)  # [B, 1]
                
                # Fake samples: generator output (detach to not backprop to generator)
                fake_samples = encoded_output.detach()  # [B, D, T]
                fake_logits = discriminator(fake_samples)  # [B, 1]
                
                # Discriminator loss
                disc_loss, disc_real_acc, disc_fake_acc = discriminator_loss(real_logits, fake_logits)
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()
                
                disc_loss_value = disc_loss.detach()
            
            # Step 2: Generator adversarial loss (fool discriminator)
            # Generator wants discriminator to classify its output as real
            fake_logits_for_gen = discriminator(encoded_output)  # [B, 1]
            gan_g_loss = generator_loss(fake_logits_for_gen)
            
            # Add to total loss
            loss = loss + gan_weight * gan_g_loss
            
            # Note: GAN statistics will be printed at epoch summary, not per batch
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Debug: Check gradients on first batch of each epoch
        if num_batches == 0 and rank == 0:
            print(f"\n{'='*80}")
            print(f"GRADIENT DEBUGGING (First batch, Epoch {epoch})")
            print(f"{'='*80}")
            total_grad_norm = 0.0
            grad_info = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    grad_info.append((name, grad_norm, param.grad.abs().mean().item(), param.grad.abs().max().item()))
            
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Total gradient norm (before clipping): {total_grad_norm:.6f}")
            
            # Show top 10 gradients by norm
            print("\nTop 10 parameters by gradient norm:")
            for name, norm, mean, max_val in sorted(grad_info, key=lambda x: -x[1])[:10]:
                print(f"  {name:50s}: norm={norm:.6f}, mean={mean:.6f}, max={max_val:.6f}")
            
            # Check for zero gradients
            zero_grad = [name for name, norm, _, _ in grad_info if norm < 1e-8]
            if zero_grad:
                print(f"\n⚠️  WARNING: {len(zero_grad)} parameters have near-zero gradients:")
                for name in zero_grad[:5]:
                    print(f"    {name}")
                if len(zero_grad) > 5:
                    print(f"    ... and {len(zero_grad)-5} more")
            
            print(f"{'='*80}\n")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # Increased from 5.0 to allow healthy gradients, catch true explosions
        optimizer.step()
        
        # Disable numerical debugging after first batch
        if num_batches == 0 and rank == 0:
            model_simple_transformer.DEBUG_NUMERICS = False
            print(f"🔍 Numerical debugging disabled after batch 0\n")
        
        total_loss += loss.item()
        total_rms_input += rms_input.item() if isinstance(rms_input, torch.Tensor) else rms_input
        total_rms_target += rms_target.item() if isinstance(rms_target, torch.Tensor) else rms_target
        total_spectral += spectral.item() if isinstance(spectral, torch.Tensor) else spectral
        total_mel += mel_value.item() if isinstance(mel_value, torch.Tensor) else mel_value
        total_corr_penalty += corr_penalty.item() if isinstance(corr_penalty, torch.Tensor) else corr_penalty
        
        # Track creative agent statistics
        if mask_reg_loss is not None:
            total_mask_reg_loss += mask_reg_loss.item()
        if balance_loss is not None:
            total_balance_loss_raw += balance_loss.item()
        
        # Accumulate correlation statistics
        if 'output_input_corr_val' in locals():
            if not hasattr(train_epoch, 'total_output_input_corr'):
                train_epoch.total_output_input_corr = 0.0
                train_epoch.total_output_target_corr = 0.0
                train_epoch.num_corr_samples = 0
            train_epoch.total_output_input_corr += output_input_corr_val
            train_epoch.total_output_target_corr += output_target_corr_val
            train_epoch.num_corr_samples += 1
        
        # Track component weights and metrics
        model_unwrapped = model.module if hasattr(model, 'module') else model
        use_compositional = getattr(model_unwrapped, 'use_compositional', False)
        
        if use_compositional:
            # Compositional agent
            total_input_rhythm_w += getattr(model_unwrapped, '_last_input_rhythm_weight', 0.0)
            total_input_harmony_w += getattr(model_unwrapped, '_last_input_harmony_weight', 0.0)
            total_target_rhythm_w += getattr(model_unwrapped, '_last_target_rhythm_weight', 0.0)
            total_target_harmony_w += getattr(model_unwrapped, '_last_target_harmony_weight', 0.0)
        else:
            # Masking agent
            total_input_mask_mean += getattr(model_unwrapped, '_last_input_mask_mean', 0.0)
            total_target_mask_mean += getattr(model_unwrapped, '_last_target_mask_mean', 0.0)
            total_mask_overlap += getattr(model_unwrapped, '_last_mask_overlap', 0.0)
            # Detailed loss components from creative agent
            if hasattr(model_unwrapped, 'creative_agent') and model_unwrapped.creative_agent is not None:
                mask_gen = model_unwrapped.creative_agent.mask_generator
                total_balance_loss += getattr(mask_gen, '_last_balance_loss', 0.0)
                total_temporal_diversity += getattr(mask_gen, '_last_temporal_diversity', 0.0)
                total_complementarity += getattr(mask_gen, '_last_complementarity', 0.0)
                total_coverage += getattr(mask_gen, '_last_coverage', 0.0)
        
        # Track GAN metrics
        if discriminator is not None and gan_weight > 0:
            total_gan_loss += gan_g_loss.item()
            total_disc_loss += disc_loss_value.item()
            total_disc_real_acc += disc_real_acc
            total_disc_fake_acc += disc_fake_acc
        
        num_batches += 1
        
        if rank == 0:
            postfix = {
                'loss': f'{loss.item():.4f}',
                'rms_in': f'{rms_input.item() if isinstance(rms_input, torch.Tensor) else rms_input:.4f}',
                'rms_tgt': f'{rms_target.item() if isinstance(rms_target, torch.Tensor) else rms_target:.4f}',
                'spec': f'{spectral.item() if isinstance(spectral, torch.Tensor) else spectral:.4f}',
                'mel': f'{mel_value.item() if isinstance(mel_value, torch.Tensor) else mel_value:.4f}'
            }
            
            # Add mask statistics if creative agent is active
            if mask_reg_loss is not None:
                postfix['mask_reg'] = f'{mask_reg_loss.item():.4f}'
            if balance_loss is not None:
                postfix['balance'] = f'{balance_loss.item():.4f}'
            
            pbar.set_postfix(postfix)
    
    avg_loss = total_loss / num_batches
    avg_rms_input = total_rms_input / num_batches
    avg_rms_target = total_rms_target / num_batches
    avg_spectral = total_spectral / num_batches
    avg_mel = total_mel / num_batches
    avg_corr_penalty = total_corr_penalty / num_batches if num_batches > 0 else 0.0
    
    # Compute correlation averages (diagnostic)
    avg_output_input_corr = 0.0
    avg_output_target_corr = 0.0
    if hasattr(train_epoch, 'num_corr_samples') and train_epoch.num_corr_samples > 0:
        avg_output_input_corr = train_epoch.total_output_input_corr / train_epoch.num_corr_samples
        avg_output_target_corr = train_epoch.total_output_target_corr / train_epoch.num_corr_samples
        # Reset for next epoch
        train_epoch.total_output_input_corr = 0.0
        train_epoch.total_output_target_corr = 0.0
        train_epoch.num_corr_samples = 0
    
    # Calculate creative agent averages if used
    creative_agent_metrics = None
    if total_mask_reg_loss > 0:
        model_unwrapped = model.module if hasattr(model, 'module') else model
        use_compositional = getattr(model_unwrapped, 'use_compositional', False)
        
        if use_compositional:
            # Compositional agent metrics
            creative_agent_metrics = {
                'type': 'compositional',
                'mask_reg_loss': total_mask_reg_loss / num_batches,  # Actually novelty loss
                'input_rhythm_weight': total_input_rhythm_w / num_batches,
                'input_harmony_weight': total_input_harmony_w / num_batches,
                'target_rhythm_weight': total_target_rhythm_w / num_batches,
                'target_harmony_weight': total_target_harmony_w / num_batches,
            }
        else:
            # Masking agent metrics
            creative_agent_metrics = {
                'type': 'masking',
                'mask_reg_loss': total_mask_reg_loss / num_batches,
                'balance_loss_raw': total_balance_loss_raw / num_batches if total_balance_loss_raw > 0 else 0.0,
                'input_mask_mean': total_input_mask_mean / num_batches,
                'target_mask_mean': total_target_mask_mean / num_batches,
                'mask_overlap': total_mask_overlap / num_batches,
                'complementarity': 1.0 - (total_mask_overlap / num_batches),
                'balance_loss': total_balance_loss / num_batches if total_balance_loss > 0 else 0.0,
                'temporal_diversity': total_temporal_diversity / num_batches if total_temporal_diversity > 0 else 0.0,
                'complementarity_raw': total_complementarity / num_batches if total_complementarity > 0 else 0.0,
                'coverage_loss': total_coverage / num_batches if total_coverage > 0 else 0.0,
                'output_input_correlation': avg_output_input_corr,
                'output_target_correlation': avg_output_target_corr,
                'correlation_penalty': avg_corr_penalty,
            }
    
    # Calculate GAN averages if used
    gan_metrics = None
    if discriminator is not None and gan_weight > 0 and num_batches > 0:
        gan_metrics = {
            'gan_loss': total_gan_loss / num_batches,
            'disc_loss': total_disc_loss / num_batches,
            'disc_real_acc': total_disc_real_acc / num_batches,
            'disc_fake_acc': total_disc_fake_acc / num_batches
        }
    
    # Synchronize all processes before returning
    if dist.is_initialized():
        dist.barrier()
    
    return avg_loss, avg_rms_input, avg_rms_target, avg_spectral, avg_mel, avg_corr_penalty, creative_agent_metrics, gan_metrics


def validate_epoch(model, encodec_model, dataloader, device, rank, epoch, 
                   unity_test=False, loss_weight_input=0.0, loss_weight_target=0.0,
                   loss_weight_spectral=0.0, loss_weight_mel=0.0,
                   mask_reg_weight=0.0, corr_weight=0.0):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_rms_input = 0.0
    total_rms_target = 0.0
    total_spectral = 0.0
    total_mel = 0.0
    total_novelty = 0.0
    num_batches = 0
    
    # Store first batch for waveform visualization
    first_input_audio = None
    first_target_audio = None
    first_output_audio = None
    
    if rank == 0:
        # Force tqdm to work properly even when output is piped to tee
        pbar = tqdm(dataloader, desc="Validation", 
                   ncols=120, leave=True, position=0,
                   file=sys.stdout, mininterval=1.0, ascii=True)
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)  # Now raw audio [B, 1, samples]
            
            # Unity test: decode inputs to use as target
            if unity_test:
                input_audio = encodec_model.decoder(inputs)  # inputs is [B, D, T]
                targets = input_audio
            
            # Get input audio for combined loss
            input_audio = encodec_model.decoder(inputs)  # [B, 1, samples]
            
            # For cascade mode, encode target audio
            if model.module.num_transformer_layers > 1:
                encoded_target = encodec_model.encoder(targets)  # [B, D, T_enc]
            else:
                encoded_target = None
            
            # Forward through transformer (returns output and optional novelty/balance loss)
            if encoded_target is not None:
                result = model(inputs, encoded_target)  # Cascade mode
                if isinstance(result, tuple):
                    encoded_output = result[0]
                    mask_reg_loss = result[1] if len(result) > 1 else None
                    balance_loss = result[2] if len(result) > 2 else None
                    novelty_loss = mask_reg_loss  # For compatibility
                else:
                    encoded_output = result
                    novelty_loss = None
            else:
                result = model(inputs)  # Single stage mode
                if isinstance(result, tuple):
                    encoded_output = result[0]
                    novelty_loss = result[1] if len(result) > 1 else None
                else:
                    encoded_output = result
                    novelty_loss = None
            
            # Decode to audio space
            output_audio = encodec_model.decoder(encoded_output)  # [B, 1, samples]
            
            # Store first sample for visualization
            if first_input_audio is None and rank == 0:
                first_input_audio = input_audio[0, 0].cpu()  # [samples]
                first_target_audio = targets[0, 0].cpu()  # [samples]
                first_output_audio = output_audio[0, 0].cpu()  # [samples]
            
            # Combined loss with all components (validation always uses clean targets)
            loss, rms_input, rms_target, spectral, mel_value, corr_penalty = combined_loss(
                output_audio, input_audio, targets,
                loss_weight_input, loss_weight_target,
                loss_weight_spectral, loss_weight_mel,
                weight_correlation=corr_weight
            )
            
            # Add novelty loss if creative agent is being used
            if novelty_loss is not None:
                loss = loss + mask_reg_weight * novelty_loss
            
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            total_rms_input += rms_input.item() if isinstance(rms_input, torch.Tensor) else rms_input
            total_rms_target += rms_target.item() if isinstance(rms_target, torch.Tensor) else rms_target
            total_spectral += spectral.item() if isinstance(spectral, torch.Tensor) else spectral
            total_mel += mel_value.item() if isinstance(mel_value, torch.Tensor) else mel_value
            if novelty_loss is not None:
                total_novelty += novelty_loss.item()
            num_batches += 1
            
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() if isinstance(loss, torch.Tensor) else loss:.4f}',
                    'rms_in': f'{rms_input.item() if isinstance(rms_input, torch.Tensor) else rms_input:.4f}',
                    'rms_tgt': f'{rms_target.item() if isinstance(rms_target, torch.Tensor) else rms_target:.4f}',
                    'spec': f'{spectral.item() if isinstance(spectral, torch.Tensor) else spectral:.4f}',
                    'mel': f'{mel_value.item() if isinstance(mel_value, torch.Tensor) else mel_value:.4f}'
                })
    
    avg_loss = total_loss / num_batches
    avg_rms_input = total_rms_input / num_batches
    avg_rms_target = total_rms_target / num_batches
    avg_spectral = total_spectral / num_batches
    avg_mel = total_mel / num_batches
    
    # Display waveform visualization on rank 0
    if rank == 0 and first_input_audio is not None:
        print()  # New line after validation progress bar
        
        # Compute RMS levels
        input_rms = torch.sqrt(torch.mean(first_input_audio ** 2))
        target_rms = torch.sqrt(torch.mean(first_target_audio ** 2))
        output_rms = torch.sqrt(torch.mean(first_output_audio ** 2))
        
        # Compute correlations
        import numpy as np
        out_np = first_output_audio.numpy()
        tgt_np = first_target_audio.numpy()
        in_np = first_input_audio.numpy()
        corr_out_tgt = np.corrcoef(out_np, tgt_np)[0, 1]
        corr_out_in = np.corrcoef(out_np, in_np)[0, 1]
        
        # Display waveforms
        print(_create_waveform_visualization(first_target_audio, "Target "))
        print(_create_waveform_visualization(first_output_audio, "Output "))
        
        # Display statistics
        rms_ratio = output_rms.item() / input_rms.item() if input_rms.item() > 1e-8 else 0.0
        print(f"\n  RMS: Input={input_rms.item():.4f}  Target={target_rms.item():.4f}  Output={output_rms.item():.4f}  (Out/In ratio: {rms_ratio:.2f})")
        print(f"  Correlation: Out→Target={corr_out_tgt:.3f}  Out→Input={corr_out_in:.3f}")
    
    # Synchronize all processes before returning
    if dist.is_initialized():
        dist.barrier()
    
    # Note: Validation doesn't return correlation penalty (not computed during validation)
    return avg_loss, avg_rms_input, avg_rms_target, avg_spectral, avg_mel


def _create_waveform_visualization(audio, label, num_segments=50):
    """
    Create a text-based waveform visualization showing loudness over time.
    
    Args:
        audio: 1D audio tensor [samples]
        label: Label for the waveform (e.g., "Input", "Target", "Output")
        num_segments: Number of time segments to display (default: 50)
    
    Returns:
        String with visualization (e.g., "Input  : ░░██░░██░░████...")
    """
    # Split audio into segments
    segment_length = len(audio) // num_segments
    
    # Compute RMS for each segment
    rms_values = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]
        rms = torch.sqrt(torch.mean(segment ** 2))
        rms_values.append(rms.item())
    
    # Normalize to 0-1 range
    rms_values = np.array(rms_values)
    max_rms = rms_values.max()
    if max_rms > 0:
        rms_values = rms_values / max_rms
    
    # Create visualization string
    vis = ""
    for rms in rms_values:
        if rms > 0.5:
            vis += "██"
        elif rms > 0.25:
            vis += "░░"
        else:
            vis += "  "
    
    return f"{label}: {vis}"


def train_worker(rank, world_size, args):
    """Main training worker for each GPU"""
    
    # Setup graceful shutdown handler
    shutdown_flag = {'stop': False}
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully"""
        if rank == 0:
            print("\n🛑 Received interrupt signal. Shutting down gracefully...")
        shutdown_flag['stop'] = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup DDP
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Worker {rank}: Starting training on {world_size} GPUs")
        if args.unity_test:
            print(f"⚠️  UNITY TEST MODE: Target will be replaced with Input")
        print(f"{'='*80}\n")
        
        # Initialize MLflow
        if HAS_MLFLOW:
            mlflow.set_tracking_uri("file:./mlruns")  # Local tracking
            mlflow.set_experiment("simple_transformer_training")
            mlflow.start_run(run_name=f"{'unity_test' if args.unity_test else 'continuation'}_lr{args.lr}_layers{args.num_layers}")
            
            # Log parameters
            mlflow.log_params({
                "encoding_dim": args.encoding_dim,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "num_transformer_layers": args.num_transformer_layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "world_size": args.world_size,
                "unity_test": args.unity_test,
                "encodec_sr": args.encodec_sr,
                "encodec_bandwidth": args.encodec_bandwidth,
                "use_compositional_agent": args.use_compositional_agent,
                "use_creative_agent": args.use_creative_agent,
                "mask_reg_weight": args.mask_reg_weight,
                "corr_weight": args.corr_weight,
                "gan_weight": args.gan_weight,
                "loss_weight_spectral": args.loss_weight_spectral,
                "anti_cheating": args.anti_cheating
            })
            print("MLflow tracking enabled: ./mlruns")
        else:
            print("MLflow not available - logging disabled")
    
    # Set random seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Load frozen EnCodec model
    if rank == 0:
        print("Loading EnCodec model (frozen)...")
    
    encodec_model = load_encodec_model(
        bandwidth=args.encodec_bandwidth,
        sample_rate=args.encodec_sr,
        device=device
    )
    
    if rank == 0:
        print(f"  EnCodec loaded: {args.encodec_sr} Hz, bandwidth={args.encodec_bandwidth}")
    
    # Load datasets
    if rank == 0:
        print("Loading WAV pairs datasets...")
    
    train_folder = os.path.join(args.dataset_folder, 'train')
    val_folder = os.path.join(args.dataset_folder, 'val')
    
    train_dataset = WavPairsDataset(train_folder, encodec_model, device, 
                                    shuffle_targets=args.shuffle_targets)
    val_dataset = WavPairsDataset(val_folder, encodec_model, device, 
                                  shuffle_targets=False)  # Always matched pairs for validation
    
    if rank == 0:
        print(f"  Train: {len(train_dataset)} pairs")
        print(f"  Val: {len(val_dataset)} pairs")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False  # Already on GPU from encoding
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    if rank == 0:
        print(f"  Train batches per GPU: {len(train_loader)}")
        print(f"  Val batches per GPU: {len(val_loader)}")
    
    # Create model
    if rank == 0:
        print("\nCreating SimpleTransformer model...")
    
    model = SimpleTransformer(
        encoding_dim=args.encoding_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_transformer_layers=args.num_transformer_layers,
        anti_cheating=args.anti_cheating,
        use_creative_agent=args.use_creative_agent,
        use_compositional_agent=args.use_compositional_agent
    ).to(device)
    
    # Wrap with DDP (find_unused_parameters needed for creative agent discriminator)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Create discriminator if GAN training is enabled
    discriminator = None
    disc_optimizer = None
    if args.gan_weight > 0:
        if rank == 0:
            print("\nCreating Audio Discriminator...")
        
        discriminator = AudioDiscriminator(
            encoding_dim=args.encoding_dim,
            hidden_dims=[256, 512, 512, 256]
        ).to(device)
        
        # Wrap with DDP
        discriminator = DDP(discriminator, device_ids=[rank])
        
        # Separate optimizer for discriminator
        disc_lr = args.disc_lr if args.disc_lr is not None else args.lr * 0.5
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=disc_lr,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.999)  # Recommended for GANs
        )
        
        if rank == 0:
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            print(f"  Discriminator LR: {disc_lr:.2e}")
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {num_params:,}")
        print(f"  Loss: RMS in decoded audio space")
    
    # Optimizer (no criterion - using RMS loss function)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    gan_curriculum_counter = 0  # Track epochs for pure GAN curriculum learning
    
    if args.resume:
        if rank == 0:
            print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore discriminator state if available
        if discriminator is not None and 'discriminator_state_dict' in checkpoint:
            discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            if rank == 0:
                print(f"  Restored discriminator state")
        
        if rank == 0:
            print(f"  Resumed from epoch {start_epoch}")
    
    # Training loop
    if rank == 0:
        print(f"\n{'='*80}")
        print("Starting training loop")
        print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Check for shutdown signal
        if shutdown_flag['stop']:
            if rank == 0:
                print("\n✓ Training interrupted. Saving checkpoint before exit...")
                # Save checkpoint
                if args.checkpoint_dir:
                    final_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'patience_counter': patience_counter
                    }
                    final_path = os.path.join(args.checkpoint_dir, 'interrupted_checkpoint.pt')
                    torch.save(final_checkpoint, final_path)
                    print(f"✓ Saved interrupted checkpoint: {final_path}")
            break
        
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Update GAN curriculum counter
        if epoch >= args.gan_curriculum_start_epoch:
            gan_curriculum_counter = epoch - args.gan_curriculum_start_epoch + 1
        else:
            gan_curriculum_counter = 0
        
        # Train
        train_loss, train_rms_input, train_rms_target, train_spectral, train_mel, train_corr_penalty, train_creative, gan_metrics = train_epoch(
            model, encodec_model, train_loader, optimizer,
            device, rank, epoch + 1, unity_test=args.unity_test,
            loss_weight_input=args.loss_weight_input,
            loss_weight_target=args.loss_weight_target,
            loss_weight_spectral=args.loss_weight_spectral,
            loss_weight_mel=args.loss_weight_mel,
            mask_type=args.mask_type,
            mask_temporal_segment=args.mask_temporal_segment,
            mask_freq_split=args.mask_freq_split,
            mask_channel_keep=args.mask_channel_keep,
            mask_energy_threshold=args.mask_energy_threshold,
            mask_reg_weight=args.mask_reg_weight,
            balance_loss_weight=args.balance_loss_weight,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            gan_weight=args.gan_weight,
            disc_update_freq=args.disc_update_freq,
            corr_weight=args.corr_weight,
            pure_gan_mode=args.pure_gan_mode,
            gan_curriculum_counter=gan_curriculum_counter,
            gan_noise_ceiling=args.gan_noise_ceiling
        )
        
        # Validate
        val_loss, val_rms_input, val_rms_target, val_spectral, val_mel = validate_epoch(
            model, encodec_model, val_loader, device, rank, epoch + 1,
            unity_test=args.unity_test,
            loss_weight_input=args.loss_weight_input,
            loss_weight_target=args.loss_weight_target,
            loss_weight_spectral=args.loss_weight_spectral,
            loss_weight_mel=args.loss_weight_mel,
            mask_reg_weight=args.mask_reg_weight,
            corr_weight=args.corr_weight
        )
        
        # Update scheduler - DISABLED for debugging constant validation loss
        # scheduler.step(val_loss)
        
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f} (rms_in: {train_rms_input:.4f}, rms_tgt: {train_rms_target:.4f}, spec: {train_spectral:.4f}, mel: {train_mel:.4f}, corr_penalty: {train_corr_penalty:.4f})")
            if train_creative is not None:
                if train_creative.get('type') == 'compositional':
                    print(f"  🎼 Compositional Agent: novelty={train_creative['mask_reg_loss']:.4f}, in_rhythm={train_creative['input_rhythm_weight']:.3f}, in_harmony={train_creative['input_harmony_weight']:.3f}, tgt_rhythm={train_creative['target_rhythm_weight']:.3f}, tgt_harmony={train_creative['target_harmony_weight']:.3f}")
                else:
                    print(f"  🎨 Creative Agent: mask_reg={train_creative['mask_reg_loss']:.4f}, complementarity={train_creative['complementarity']:.1%}, overlap={train_creative['mask_overlap']:.3f}")
                    print(f"     Input mask: {train_creative['input_mask_mean']:.3f}, Target mask: {train_creative['target_mask_mean']:.3f}")
                    print(f"     Balance loss (raw): {train_creative['balance_loss_raw']:.4f} [×{args.balance_loss_weight} weight = {train_creative['balance_loss_raw']*args.balance_loss_weight:.4f}]")
                    print(f"     Temporal diversity: {train_creative['temporal_diversity']:.4f}")
                    print(f"  📊 Output Correlation Analysis:")
                    print(f"     Output→Input corr: {train_creative['output_input_correlation']:.3f} (closer to 1.0 = using input)")
                    print(f"     Output→Target corr: {train_creative['output_target_correlation']:.3f} (closer to 1.0 = copying target)")
                    if abs(train_creative['output_target_correlation']) > abs(train_creative['output_input_correlation']) * 2:
                        print(f"     ⚠️  OUTPUT IS COPYING TARGET! (target corr >> input corr)")
                    elif abs(train_creative['output_input_correlation']) > abs(train_creative['output_target_correlation']) * 2:
                        print(f"     ⚠️  OUTPUT IS COPYING INPUT! (input corr >> target corr)")
                    else:
                        print(f"     ✓ Output appears to mix both sources")
            if gan_metrics is not None:
                print(f"  GAN: gen_loss={gan_metrics['gan_loss']:.4f}, disc_loss={gan_metrics['disc_loss']:.4f}, disc_acc(real/fake)={gan_metrics['disc_real_acc']:.1%}/{gan_metrics['disc_fake_acc']:.1%}")
            print(f"  Val Loss: {val_loss:.4f} (rms_in: {val_rms_input:.4f}, rms_tgt: {val_rms_target:.4f}, spec: {val_spectral:.4f}, mel: {val_mel:.4f})")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Log to MLflow
            if HAS_MLFLOW:
                metrics = {
                    "train_loss": train_loss,
                    "train_rms_input": train_rms_input,
                    "train_rms_target": train_rms_target,
                    "train_spectral": train_spectral,
                    "train_mel": train_mel,
                    "train_corr_penalty": train_corr_penalty,
                    "val_loss": val_loss,
                    "val_rms_input": val_rms_input,
                    "val_rms_target": val_rms_target,
                    "val_spectral": val_spectral,
                    "val_mel": val_mel,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                
                # Add creative agent metrics if available
                if train_creative is not None:
                    if train_creative['type'] == 'compositional':
                        metrics.update({
                            "train_novelty_loss": train_creative['mask_reg_loss'],  # mask_reg_loss contains novelty for compositional
                            "train_input_rhythm_weight": train_creative['input_rhythm_weight'],
                            "train_input_harmony_weight": train_creative['input_harmony_weight'],
                            "train_target_rhythm_weight": train_creative['target_rhythm_weight'],
                            "train_target_harmony_weight": train_creative['target_harmony_weight']
                        })
                    else:  # masking agent
                        metrics.update({
                            "train_mask_reg_loss": train_creative['mask_reg_loss'],
                            "train_input_mask_mean": train_creative['input_mask_mean'],
                            "train_target_mask_mean": train_creative['target_mask_mean'],
                            "train_mask_overlap": train_creative['mask_overlap'],
                            "train_complementarity": train_creative['complementarity'],
                            "train_output_input_corr": train_creative['output_input_correlation'],
                            "train_output_target_corr": train_creative['output_target_correlation']
                        })
                
                # Add GAN metrics if available
                if gan_metrics is not None:
                    metrics.update({
                        "train_gan_loss": gan_metrics['gan_loss'],
                        "train_disc_loss": gan_metrics['disc_loss'],
                        "train_disc_real_acc": gan_metrics['disc_real_acc'],
                        "train_disc_fake_acc": gan_metrics['disc_fake_acc']
                    })
                
                mlflow.log_metrics(metrics, step=epoch + 1)
        
        # Save checkpoint
        if rank == 0:
            # Save regular checkpoint
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f'checkpoint_epoch_{epoch + 1}.pt'
                )
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'args': vars(args)
                }
                if discriminator is not None:
                    checkpoint_dict['discriminator_state_dict'] = discriminator.module.state_dict()
                    checkpoint_dict['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
                torch.save(checkpoint_dict, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                best_checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'args': vars(args)
                }
                if discriminator is not None:
                    best_checkpoint_dict['discriminator_state_dict'] = discriminator.module.state_dict()
                    best_checkpoint_dict['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
                torch.save(best_checkpoint_dict, best_path)
                print(f"  ✓ New best model! Val loss: {val_loss:.6f}")
                
                # Log best model to MLflow
                if HAS_MLFLOW:
                    mlflow.log_metric("best_val_loss", val_loss, step=epoch + 1)
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.patience})")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Sync early stopping across all ranks
        if rank == 0:
            should_stop = torch.tensor([patience_counter >= args.patience], dtype=torch.int32, device=device)
        else:
            should_stop = torch.tensor([0], dtype=torch.int32, device=device)
        dist.broadcast(should_stop, src=0)
        
        if should_stop.item():
            break
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"{'='*80}\n")
        
        # End MLflow run
        if HAS_MLFLOW:
            mlflow.log_metric("final_best_val_loss", best_val_loss)
            mlflow.end_run()
    
    # Proper cleanup to avoid TCPStore warnings
    try:
        dist.barrier()  # Ensure all ranks reach this point
    except Exception:
        pass  # Ignore barrier errors during shutdown
    
    try:
        dist.destroy_process_group()  # Clean shutdown of process group
    except Exception:
        pass  # Ignore cleanup errors
    
    torch.cuda.empty_cache()  # Clear GPU memory
    
    # Cleanup
    cleanup_ddp()
