#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on a validation sample and save as audio file

Takes:
- Trained checkpoint from checkpoints_simple/
- Sample from validation set
- Generates audio continuation
- Saves input, target, and predicted audio
"""

import os
import torch
import soundfile as sf
import numpy as np
import encodec
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from model_simple_transformer import SimpleTransformer


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    args = checkpoint.get('args', {})
    
    # Create model with FULL configuration (including creative agent)
    model = SimpleTransformer(
        encoding_dim=args.get('encoding_dim', 128),
        nhead=args.get('nhead', 8),
        num_layers=args.get('num_layers', 4),
        num_transformer_layers=args.get('num_transformer_layers', 1),
        dropout=args.get('dropout', 0.1),
        anti_cheating=args.get('anti_cheating', 0.0),
        use_creative_agent=args.get('use_creative_agent', False),
        use_compositional_agent=args.get('use_compositional_agent', False)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Cascade stages: {args.get('num_transformer_layers', 1)}")
    print(f"  Creative agent: {args.get('use_creative_agent', False)}")
    print(f"  Compositional agent: {args.get('use_compositional_agent', False)}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_encodec(bandwidth=6.0, sample_rate=24000, device='cuda'):
    """Load EnCodec model"""
    print("\nLoading EnCodec model...")
    model = encodec.EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Bandwidth: {bandwidth}")
    
    return model


def generate_audio(model, encodec_model, input_wav_path, target_wav_path, device='cuda'):
    """
    Generate audio using creative agent mixing input and target
    
    Args:
        model: Trained transformer with creative agent
        encodec_model: EnCodec model
        input_wav_path: Path to input WAV file
        target_wav_path: Path to target WAV file
        device: Device to run on
        
    Returns:
        input_audio: Original input audio [samples]
        predicted_audio: Generated mixed output [samples]
    """
    print(f"\nGenerating audio from:")
    print(f"  Input: {input_wav_path}")
    print(f"  Target: {target_wav_path}")
    
    # Load input audio
    input_audio, sr = sf.read(input_wav_path)
    print(f"  Input shape: {input_audio.shape}, SR: {sr}")
    
    # Load target audio
    target_audio, sr2 = sf.read(target_wav_path)
    print(f"  Target shape: {target_audio.shape}, SR: {sr2}")
    assert sr == sr2, f"Sample rates don't match: {sr} vs {sr2}"
    
    # Convert stereo to mono if needed
    if input_audio.ndim == 2:
        input_audio = input_audio.mean(axis=1)
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)
    
    # Convert to tensors
    input_audio_tensor = torch.from_numpy(input_audio).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
    target_audio_tensor = torch.from_numpy(target_audio).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
    
    # Encode input and target
    with torch.no_grad():
        encoded_input = encodec_model.encoder(input_audio_tensor)  # [1, 128, T]
        encoded_target = encodec_model.encoder(target_audio_tensor)  # [1, 128, T]
        print(f"  Encoded input shape: {encoded_input.shape}")
        print(f"  Encoded target shape: {encoded_target.shape}")
        
        # Transform with trained model (creative agent mixes input + target)
        model_output = model(encoded_input, encoded_target)
        
        # Handle output format (could be tuple or tensor)
        if isinstance(model_output, tuple):
            encoded_output = model_output[0]  # [1, 128, T]
            print(f"  Transformed shape: {encoded_output.shape}")
        else:
            encoded_output = model_output
            print(f"  Transformed shape: {encoded_output.shape}")
        
        # Decode to audio
        predicted_audio = encodec_model.decoder(encoded_output)  # [1, 1, samples]
        predicted_audio = predicted_audio.squeeze().cpu().numpy().astype(np.float32)  # [samples]
    
    # Check output magnitude (skip first 5000 samples for RMS calculation to avoid silent starts)
    start_idx = 5000
    output_rms = np.sqrt(np.mean(predicted_audio[start_idx:] ** 2))
    input_rms = np.sqrt(np.mean(input_audio[start_idx:] ** 2))
    
    print(f"  Output shape: {predicted_audio.shape}")
    print(f"  Output range: [{predicted_audio.min():.3f}, {predicted_audio.max():.3f}]")
    print(f"  Output RMS (excl. silent start): {output_rms:.6f}")
    print(f"  Input RMS (excl. silent start): {input_rms:.6f}")
    print(f"  RMS ratio (output/input): {output_rms/input_rms:.3f}")
    
    if output_rms < 0.01:
        print(f"  ⚠️  WARNING: Output is very quiet! Likely a training issue.")
    
    return input_audio, predicted_audio


def save_audio(audio, path, sample_rate=24000):
    """Save audio to file"""
    # Ensure float32 dtype
    audio = np.array(audio, dtype=np.float32)
    # Normalize to [-1, 1] if needed
    audio = np.clip(audio, -1.0, 1.0)
    
    sf.write(path, audio, sample_rate, subtype='PCM_16')
    print(f"Saved: {path}")


def visualize_waveforms(input_audio, target_audio, predicted_audio, output_path, sample_rate=24000):
    """
    Visualize input, target, and predicted waveforms
    
    Args:
        input_audio: Input audio [samples]
        target_audio: Target audio [samples]
        predicted_audio: Predicted audio [samples]
        output_path: Path to save figure
        sample_rate: Sample rate
    """
    # Create time axis
    duration = len(input_audio) / sample_rate
    time_input = np.linspace(0, duration, len(input_audio))
    time_output = np.linspace(duration, 2*duration, len(target_audio))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot input
    axes[0].plot(time_input, input_audio, color='blue', linewidth=0.5, alpha=0.7)
    axes[0].set_title('Input Audio (First 16s)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_xlim(0, duration)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot target
    axes[1].plot(time_output, target_audio, color='green', linewidth=0.5, alpha=0.7)
    axes[1].set_title('Target Audio (Ground Truth Continuation)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_xlim(duration, 2*duration)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot predicted
    axes[2].plot(time_output, predicted_audio, color='red', linewidth=0.5, alpha=0.7)
    axes[2].set_title('Predicted Audio (Model Output)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].set_xlim(duration, 2*duration)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def visualize_spectrograms(input_audio, target_audio, predicted_audio, output_path, sample_rate=24000):
    """
    Visualize spectrograms of input, target, and predicted audio
    
    Args:
        input_audio: Input audio [samples]
        target_audio: Target audio [samples]
        predicted_audio: Predicted audio [samples]
        output_path: Path to save figure
        sample_rate: Sample rate
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Parameters for spectrogram
    n_fft = 2048
    hop_length = 512
    
    # Plot input spectrogram
    D_input = np.abs(plt.specgram(input_audio, Fs=sample_rate, NFFT=n_fft, 
                                    noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[0].clear()
    im0 = axes[0].imshow(10 * np.log10(D_input + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(input_audio)/sample_rate, 0, sample_rate/2])
    axes[0].set_title('Input Audio Spectrogram (First 16s)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylim(0, 8000)
    plt.colorbar(im0, ax=axes[0], label='Power (dB)')
    
    # Plot target spectrogram
    D_target = np.abs(plt.specgram(target_audio, Fs=sample_rate, NFFT=n_fft, 
                                     noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[1].clear()
    im1 = axes[1].imshow(10 * np.log10(D_target + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(target_audio)/sample_rate, 0, sample_rate/2])
    axes[1].set_title('Target Audio Spectrogram (Ground Truth)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylim(0, 8000)
    plt.colorbar(im1, ax=axes[1], label='Power (dB)')
    
    # Plot predicted spectrogram
    D_pred = np.abs(plt.specgram(predicted_audio, Fs=sample_rate, NFFT=n_fft, 
                                   noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[2].clear()
    im2 = axes[2].imshow(10 * np.log10(D_pred + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(predicted_audio)/sample_rate, 0, sample_rate/2])
    axes[2].set_title('Predicted Audio Spectrogram (Model Output)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[2].set_ylim(0, 8000)
    plt.colorbar(im2, ax=axes[2], label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spectrogram: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate audio continuation from validation sample')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to checkpoint')
    parser.add_argument('--val_folder', type=str, default='dataset/val',
                       help='Validation dataset folder')
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of samples to generate (default: 2)')
    parser.add_argument('--output_folder', type=str, default='validation_samples',
                       help='Output folder for audio files')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--boost', type=float, default=1.0,
                       help='Additional volume boost (1.0=normal, 2.0=double, etc.)')
    
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    print("="*80)
    print("AUDIO GENERATION FROM VALIDATION SAMPLES")
    print("="*80)
    
    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_checkpoint(args.checkpoint, device)
    encodec_model = load_encodec(device=device)
    
    # Check if model was trained with unity test
    checkpoint = torch.load(args.checkpoint, map_location=device)
    args_dict = checkpoint.get('args', {})
    unity_test_trained = args_dict.get('unity_test', False)
    
    if unity_test_trained:
        print("\n" + "!"*80)
        print("WARNING: This model was trained with UNITY TEST (target=input)")
        print("It will try to reconstruct the input, NOT predict continuation!")
        print("For real continuation, train with --unity_test false")
        print("!"*80 + "\n")
    
    # Find validation samples
    val_files = sorted([f for f in os.listdir(args.val_folder) if f.endswith('_input.wav')])
    
    if len(val_files) == 0:
        print(f"\nError: No validation samples found in {args.val_folder}")
        return
    
    # Filter to only include pairs where both input and output exist
    valid_pairs = []
    for input_file in val_files:
        output_file = input_file.replace('_input.wav', '_output.wav')
        input_path = os.path.join(args.val_folder, input_file)
        output_path = os.path.join(args.val_folder, output_file)
        
        if os.path.exists(input_path) and os.path.exists(output_path):
            valid_pairs.append(input_file)
    
    if len(valid_pairs) == 0:
        print(f"\nError: No complete pairs found in {args.val_folder}")
        print(f"Make sure both *_input.wav and *_output.wav files exist")
        return
    
    # Limit to requested number of samples
    num_samples = min(args.num_samples, len(valid_pairs))
    print(f"\nProcessing {num_samples} complete pairs from {len(valid_pairs)} available")
    
    # Process each sample
    total_rms_error = 0.0
    
    for sample_idx in range(num_samples):
        print("\n" + "-"*80)
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print("-"*80)
        
        input_file = valid_pairs[sample_idx]
        output_file = input_file.replace('_input.wav', '_output.wav')
        
        input_path = os.path.join(args.val_folder, input_file)
        target_path = os.path.join(args.val_folder, output_file)
        
        print(f"  Input: {input_file}")
        print(f"  Target: {output_file}")
        
        # Generate audio (creative agent mixes input + target)
        input_audio, predicted_audio = generate_audio(model, encodec_model, input_path, target_path, device)
        
        # Load target audio for comparison (already loaded in generate_audio, but load again for consistency)
        target_audio, _ = sf.read(target_path)
        if target_audio.ndim == 2:
            target_audio = target_audio.mean(axis=1)
        
        # DON'T normalize or boost - save raw model output
        # (Normalization seems to corrupt the audio for some players)
        
        # Save results
        sample_name = input_file.replace('_input.wav', '')
        
        save_audio(input_audio, os.path.join(args.output_folder, f'{sample_name}_1_input.wav'))
        save_audio(target_audio, os.path.join(args.output_folder, f'{sample_name}_2_target.wav'))
        save_audio(predicted_audio, os.path.join(args.output_folder, f'{sample_name}_3_predicted.wav'))
        
        # Visualize waveforms
        print("\nGenerating visualizations...")
        visualize_waveforms(input_audio, target_audio, predicted_audio, 
                          os.path.join(args.output_folder, f'{sample_name}_waveforms.png'))
        
        # Visualize spectrograms
        visualize_spectrograms(input_audio, target_audio, predicted_audio,
                             os.path.join(args.output_folder, f'{sample_name}_spectrograms.png'))
        
        # Compute RMS error
        rms_error = np.sqrt(np.mean((predicted_audio - target_audio) ** 2))
        total_rms_error += rms_error
        print(f"  RMS error: {rms_error:.6f}")
    
    # Average RMS error
    avg_rms_error = total_rms_error / num_samples
    
    print("\n" + "="*80)
    print("AUDIO GENERATION COMPLETE")
    print("="*80)
    print(f"\nProcessed {num_samples} samples")
    print(f"Average RMS error: {avg_rms_error:.6f}")
    print(f"\nGenerated files in: {args.output_folder}/")
    print(f"  *_1_input.wav         - Original input (16s)")
    print(f"  *_2_target.wav        - Ground truth continuation (16s)")
    print(f"  *_3_predicted.wav     - Model prediction (16s)")
    print(f"  *_waveforms.png       - Waveform visualization")
    print(f"  *_spectrograms.png    - Spectrogram visualization")
    print("\nPlay WAV files in order to hear: input → prediction → target")
    print("View PNG files to see waveforms and spectrograms")
    print("="*80)


if __name__ == "__main__":
    main()
