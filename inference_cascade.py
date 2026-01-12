#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on validation samples with CASCADE model

Takes:
- Trained cascade checkpoint from checkpoints_spectral/
- Samples from dataset_pairs_wav/val/
- Generates audio continuation using cascade architecture
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
from scipy import signal

from model_simple_transformer import SimpleTransformer


def remove_frequency_outliers(audio, sample_rate=24000, percentile=99.5, filter_order=5):
    """
    Remove dominant frequency outliers that cause noise/artifacts.
    
    Detects abnormally strong frequency components using FFT and applies
    notch filters to remove them while preserving musical content.
    
    Args:
        audio: Input audio array [samples]
        sample_rate: Audio sample rate (Hz)
        percentile: Percentile threshold for outlier detection (99.5 = top 0.5%)
        filter_order: Order of notch filter (higher = sharper notch)
    
    Returns:
        Filtered audio array
    """
    # Compute FFT
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitudes = np.abs(fft)
    
    # Find outlier frequencies (abnormally strong components)
    threshold = np.percentile(magnitudes, percentile)
    outlier_indices = np.where(magnitudes > threshold)[0]
    outlier_freqs = freqs[outlier_indices]
    
    if len(outlier_freqs) == 0:
        print(f"  🔇 No frequency outliers detected (threshold={threshold:.1f})")
        return audio
    
    print(f"  🔇 Detected {len(outlier_freqs)} frequency outliers:")
    print(f"     Threshold: {threshold:.1f}, Percentile: {percentile}%")
    
    # Apply notch filter to each outlier frequency
    filtered_audio = audio.copy()
    for freq in outlier_freqs[:10]:  # Limit to top 10 outliers
        if freq < 50:  # Skip DC and very low frequencies
            continue
        
        # Design notch filter with quality factor Q
        Q = 30.0  # Bandwidth = freq/Q
        b, a = signal.iirnotch(freq, Q, sample_rate)
        filtered_audio = signal.filtfilt(b, a, filtered_audio)
        
        outlier_mag = magnitudes[np.argmin(np.abs(freqs - freq))]
        print(f"     - {freq:>7.1f} Hz (magnitude: {outlier_mag:>10.1f})")
    
    # Normalize to match original RMS
    original_rms = np.sqrt(np.mean(audio**2))
    filtered_rms = np.sqrt(np.mean(filtered_audio**2))
    if filtered_rms > 0:
        filtered_audio = filtered_audio * (original_rms / filtered_rms)
    
    return filtered_audio


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load trained cascade model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    args = checkpoint.get('args', {})
    
    # Create cascade model
    model = SimpleTransformer(
        encoding_dim=args.get('encoding_dim', 128),
        nhead=args.get('nhead', 8),
        num_layers=args.get('num_layers', 4),
        num_transformer_layers=args.get('num_transformer_layers', 1),  # CASCADE parameter!
        dropout=args.get('dropout', 0.1),
        anti_cheating=args.get('anti_cheating', 0.0),  # Anti-cheating parameter
        use_creative_agent=args.get('use_creative_agent', False),  # Attention-based agent
        use_compositional_agent=args.get('use_compositional_agent', False)  # Compositional agent
    ).to(device)
    
    # Load weights (strict=False to allow loading old checkpoints without RMS parameters)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Detect which creative agent is used
    creative_agent_type = "None"
    if args.get('use_compositional_agent', False):
        creative_agent_type = "Compositional (rhythm/harmony/timbre)"
    elif args.get('use_creative_agent', False):
        creative_agent_type = "Attention-based (masking)"
    
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Cascade stages: {args.get('num_transformer_layers', 1)}")
    print(f"  Creative agent: {creative_agent_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, args


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


def generate_audio_cascade(model, encodec_model, input_wav_path, target_wav_path, 
                           num_cascade_stages, device='cuda', apply_filter=False,
                           filter_percentile=99.5):
    """
    Generate audio continuation using cascade model
    
    Args:
        model: Trained cascade transformer
        encodec_model: EnCodec model
        input_wav_path: Path to input WAV file
        target_wav_path: Path to target WAV file (needed for cascade stage 1)
        num_cascade_stages: Number of cascade stages (1 = single stage, 2+ = cascade)
        device: Device to run on
        apply_filter: If True, remove frequency outliers from output
        filter_percentile: Percentile threshold for outlier detection (99.5 = top 0.5%)
        
    Returns:
        input_audio: Original input audio [samples]
        target_audio: Target continuation audio [samples]
        predicted_audio: Generated continuation [samples]
    """
    print(f"\nGenerating audio from: {input_wav_path}")
    
    # Load input audio
    input_audio, sr = sf.read(input_wav_path)
    print(f"  Input shape: {input_audio.shape}, SR: {sr}")
    
    # Load target audio (needed for cascade stage 1)
    target_audio, _ = sf.read(target_wav_path)
    print(f"  Target shape: {target_audio.shape}")
    
    # Convert stereo to mono if needed
    if input_audio.ndim == 2:
        input_audio = input_audio.mean(axis=1)
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)
    
    # Convert to tensors
    input_audio_tensor = torch.from_numpy(input_audio).float()
    input_audio_tensor = input_audio_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
    
    target_audio_tensor = torch.from_numpy(target_audio).float()
    target_audio_tensor = target_audio_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
    
    # Encode input and target
    with torch.no_grad():
        encoded_input = encodec_model.encoder(input_audio_tensor)  # [1, 128, T]
        encoded_target = encodec_model.encoder(target_audio_tensor)  # [1, 128, T]
        print(f"  Encoded input shape: {encoded_input.shape}")
        print(f"  Encoded target shape: {encoded_target.shape}")
        
        # Transform with cascade model
        if num_cascade_stages > 1:
            # CASCADE MODE: Pass both encoded_input and encoded_target
            print(f"  Running CASCADE mode ({num_cascade_stages} stages)...")
            result = model(encoded_input, encoded_target)
            encoded_output = result[0] if isinstance(result, tuple) else result  # Handle both (output, loss, balance) and output
            
            # Show creative agent statistics if available
            if hasattr(model, '_last_input_mask_mean'):
                print(f"  Creative Agent Masks:")
                print(f"    Input mask mean: {model._last_input_mask_mean:.3f}")
                print(f"    Target mask mean: {model._last_target_mask_mean:.3f}")
                print(f"    Mask overlap: {model._last_mask_overlap:.3f}")
                print(f"    Mask sum: {model._last_input_mask_mean + model._last_target_mask_mean:.3f}")
        else:
            # SINGLE STAGE MODE: Only pass encoded_input
            print(f"  Running SINGLE STAGE mode...")
            result = model(encoded_input)
            encoded_output = result[0] if isinstance(result, tuple) else result  # Handle both formats
        
        print(f"  Transformed shape: {encoded_output.shape}")
        
        # Debug: Check encoded RMS values
        encoded_input_rms = torch.sqrt(torch.mean(encoded_input ** 2)).item()
        encoded_target_rms = torch.sqrt(torch.mean(encoded_target ** 2)).item()
        encoded_output_rms = torch.sqrt(torch.mean(encoded_output ** 2)).item()
        print(f"  Encoded RMS - Input: {encoded_input_rms:.4f}, Target: {encoded_target_rms:.4f}, Output: {encoded_output_rms:.4f}")
        
        # Evaluate rhythm transfer if creative agent is active
        if num_cascade_stages > 1 and hasattr(model, '_last_input_mask_mean'):
            try:
                from creative_agent import evaluate_rhythm_transfer
                rhythm_metrics = evaluate_rhythm_transfer(encoded_input, encoded_target, encoded_output)
                print(f"  Rhythm Transfer Analysis:")
                print(f"    Input→Output rhythm correlation: {rhythm_metrics['input_rhythm_corr']:.3f}")
                print(f"    Target→Output rhythm correlation: {rhythm_metrics['target_rhythm_corr']:.3f}")
                print(f"    Rhythm balance (Input/Target): {rhythm_metrics['rhythm_balance']:.3f}")
                print(f"    Rhythm preserved: {'✓' if rhythm_metrics['rhythm_preserved'] else '✗'}")
                print(f"    Output rhythm energy: {rhythm_metrics['output_rhythm_energy']:.3f}")
            except ImportError:
                pass
        
        # Decode to audio
        predicted_audio = encodec_model.decoder(encoded_output)  # [1, 1, samples]
        predicted_audio = predicted_audio.squeeze().cpu().numpy().astype(np.float32)  # [samples]
    
    # Statistics and amplitude normalization
    start_idx = 5000
    output_rms = np.sqrt(np.mean(predicted_audio[start_idx:] ** 2))
    input_rms = np.sqrt(np.mean(input_audio[start_idx:] ** 2))
    target_rms = np.sqrt(np.mean(target_audio[start_idx:] ** 2))
    
    print(f"  Output shape: {predicted_audio.shape}")
    print(f"  Output range BEFORE norm: [{predicted_audio.min():.3f}, {predicted_audio.max():.3f}]")
    print(f"  Input RMS: {input_rms:.6f}")
    print(f"  Target RMS: {target_rms:.6f}")
    print(f"  Output RMS: {output_rms:.6f}")
    print(f"  RMS ratio (Out/In): {output_rms/input_rms if input_rms > 1e-6 else 0:.3f}")
    
    # TEMPORARY FIX: Normalize decoded audio RMS to match target
    # This compensates for EnCodec decoder not preserving RMS from encoded space
    # After retraining with balanced masks, this may not be needed
    if output_rms > 1e-6:
        target_rms_combined = np.sqrt((input_rms**2 + target_rms**2) / 2)
        amplitude_scale = target_rms_combined / output_rms
        # Limit scaling to prevent over-amplification
        amplitude_scale = min(amplitude_scale, 3.0)
        predicted_audio = predicted_audio * amplitude_scale
        output_rms_normalized = np.sqrt(np.mean(predicted_audio[start_idx:] ** 2))
        print(f"  Applied amplitude normalization: {amplitude_scale:.3f}x")
        print(f"  Output RMS after norm: {output_rms_normalized:.6f}")
        print(f"  Output range after norm: [{predicted_audio.min():.3f}, {predicted_audio.max():.3f}]")
    
    # Check if clipping will occur and apply gentle limiting if needed
    max_abs = max(abs(predicted_audio.min()), abs(predicted_audio.max()))
    if max_abs > 1.0:
        print(f"  ⚠️  WARNING: Output exceeds [-1,1] range! Max absolute value: {max_abs:.3f}")
        # Apply soft limiting to prevent hard clipping while preserving dynamics
        # Scale down by max_abs + 10% headroom
        soft_scale = 0.9 / max_abs
        predicted_audio = predicted_audio * soft_scale
        print(f"     Applied soft limiting: scale factor {soft_scale:.3f}")
        print(f"     New range: [{predicted_audio.min():.3f}, {predicted_audio.max():.3f}]")
    
    if output_rms < 0.01:
        print(f"  ⚠️  WARNING: Output is very quiet! Likely a training issue.")
    
    # Apply frequency outlier filter if requested
    if apply_filter:
        print(f"\n  🔇 Applying frequency outlier filter (percentile={filter_percentile}%)...")
        predicted_audio = remove_frequency_outliers(
            predicted_audio, 
            sample_rate=24000, 
            percentile=filter_percentile
        )
    
    return input_audio, target_audio, predicted_audio


def save_audio(audio, path, sample_rate=24000):
    """Save audio to file"""
    audio = np.array(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    
    sf.write(path, audio, sample_rate, subtype='PCM_16')
    print(f"Saved: {path}")


def visualize_waveforms(input_audio, target_audio, predicted_audio, output_path, sample_rate=24000):
    """Visualize input, target, and predicted waveforms"""
    duration = len(input_audio) / sample_rate
    time_input = np.linspace(0, duration, len(input_audio))
    time_output = np.linspace(0, len(target_audio) / sample_rate, len(target_audio))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot input
    axes[0].plot(time_input, input_audio, color='blue', linewidth=0.5, alpha=0.7)
    axes[0].set_title('Input Audio (16s)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_xlim(0, duration)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot target
    axes[1].plot(time_output, target_audio, color='green', linewidth=0.5, alpha=0.7)
    axes[1].set_title('Target Audio (Ground Truth Continuation, 16s)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_xlim(0, len(target_audio) / sample_rate)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot predicted
    axes[2].plot(time_output, predicted_audio, color='red', linewidth=0.5, alpha=0.7)
    axes[2].set_title('Predicted Audio (Cascade Model Output, 16s)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].set_xlim(0, len(predicted_audio) / sample_rate)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def visualize_spectrograms(input_audio, target_audio, predicted_audio, output_path, sample_rate=24000):
    """Visualize spectrograms of input, target, and predicted audio"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    n_fft = 2048
    hop_length = 512
    
    # Input spectrogram
    D_input = np.abs(plt.specgram(input_audio, Fs=sample_rate, NFFT=n_fft, 
                                    noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[0].clear()
    im0 = axes[0].imshow(10 * np.log10(D_input + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(input_audio)/sample_rate, 0, sample_rate/2])
    axes[0].set_title('Input Audio Spectrogram (16s)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylim(0, 8000)
    plt.colorbar(im0, ax=axes[0], label='Power (dB)')
    
    # Target spectrogram
    D_target = np.abs(plt.specgram(target_audio, Fs=sample_rate, NFFT=n_fft, 
                                     noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[1].clear()
    im1 = axes[1].imshow(10 * np.log10(D_target + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(target_audio)/sample_rate, 0, sample_rate/2])
    axes[1].set_title('Target Audio Spectrogram (Ground Truth)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylim(0, 8000)
    plt.colorbar(im1, ax=axes[1], label='Power (dB)')
    
    # Predicted spectrogram
    D_pred = np.abs(plt.specgram(predicted_audio, Fs=sample_rate, NFFT=n_fft, 
                                   noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[2].clear()
    im2 = axes[2].imshow(10 * np.log10(D_pred + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(predicted_audio)/sample_rate, 0, sample_rate/2])
    axes[2].set_title('Predicted Audio Spectrogram (Cascade Model)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[2].set_ylim(0, 8000)
    plt.colorbar(im2, ax=axes[2], label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spectrogram: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate audio continuation with CASCADE model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to cascade checkpoint')
    parser.add_argument('--val_folder', type=str, default='dataset_pairs_wav/val',
                       help='Validation dataset folder')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to generate (default: 3)')
    parser.add_argument('--output_folder', type=str, default='inference_outputs',
                       help='Output folder for audio files')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--shuffle_targets', action='store_true',
                       help='Use random targets instead of matched continuations')
    parser.add_argument('--filter', action='store_true',
                       help='Apply frequency outlier filter to remove noise/artifacts')
    parser.add_argument('--filter_percentile', type=float, default=99.5,
                       help='Percentile threshold for frequency outlier detection (default: 99.5 = top 0.5%%)')
    
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    print("="*80)
    print("CASCADE AUDIO GENERATION FROM VALIDATION SAMPLES")
    print("="*80)
    
    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, model_args = load_checkpoint(args.checkpoint, device)
    encodec_model = load_encodec(device=device)
    
    # Get cascade configuration
    num_cascade_stages = model_args.get('num_transformer_layers', 1)
    unity_test_trained = model_args.get('unity_test', False)
    
    if unity_test_trained:
        print("\n" + "!"*80)
        print("⚠️  WARNING: This model was trained with UNITY TEST (target=input)")
        print("It will try to reconstruct the input, NOT predict continuation!")
        print("Output will likely sound like the input, not the target.")
        print("For real continuation, train with --unity_test false")
        print("!"*80 + "\n")
    
    # Find validation samples
    val_files = sorted([f for f in os.listdir(args.val_folder) if f.endswith('_input.wav')])
    all_output_files = sorted([f for f in os.listdir(args.val_folder) if f.endswith('_output.wav')])
    
    if len(val_files) == 0:
        print(f"\nError: No validation samples found in {args.val_folder}")
        return
    
    # Filter to complete pairs
    valid_pairs = []
    for input_file in val_files:
        output_file = input_file.replace('_input.wav', '_output.wav')
        input_path = os.path.join(args.val_folder, input_file)
        output_path = os.path.join(args.val_folder, output_file)
        
        if os.path.exists(input_path) and os.path.exists(output_path):
            valid_pairs.append(input_file)
    
    if len(valid_pairs) == 0:
        print(f"\nError: No complete pairs found in {args.val_folder}")
        return
    
    num_samples = min(args.num_samples, len(valid_pairs))
    print(f"\nProcessing {num_samples} complete pairs from {len(valid_pairs)} available")
    print(f"Cascade stages: {num_cascade_stages}")
    print(f"Shuffle targets: {'ENABLED (random pairs)' if args.shuffle_targets else 'DISABLED (matched pairs)'}")
    print(f"Frequency filter: {'ENABLED' if args.filter else 'DISABLED'}")
    if args.filter:
        print(f"  Filter percentile: {args.filter_percentile}% (removes top {100-args.filter_percentile}% frequency outliers)")
    
    # Process samples
    total_rms_error = 0.0
    
    for sample_idx in range(num_samples):
        print("\n" + "-"*80)
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print("-"*80)
        
        input_file = valid_pairs[sample_idx]
        
        # Select target: either matched continuation or random
        if args.shuffle_targets:
            # Random target from all available outputs
            output_file = np.random.choice(all_output_files)
            print(f"  🎲 RANDOM TARGET SELECTED")
        else:
            # Matched continuation
            output_file = input_file.replace('_input.wav', '_output.wav')
        
        input_path = os.path.join(args.val_folder, input_file)
        target_path = os.path.join(args.val_folder, output_file)
        
        print(f"  Input: {input_file}")
        print(f"  Target: {output_file}")
        
        # Generate audio with cascade
        input_audio, target_audio, predicted_audio = generate_audio_cascade(
            model, encodec_model, input_path, target_path, num_cascade_stages, device,
            apply_filter=args.filter, filter_percentile=args.filter_percentile
        )
        
        # Save results
        sample_name = input_file.replace('_input.wav', '')
        
        save_audio(input_audio, os.path.join(args.output_folder, f'{sample_name}_1_input.wav'))
        save_audio(target_audio, os.path.join(args.output_folder, f'{sample_name}_2_target.wav'))
        save_audio(predicted_audio, os.path.join(args.output_folder, f'{sample_name}_3_predicted.wav'))
        
        # Visualize
        print("\nGenerating visualizations...")
        visualize_waveforms(input_audio, target_audio, predicted_audio, 
                          os.path.join(args.output_folder, f'{sample_name}_waveforms.png'))
        
        visualize_spectrograms(input_audio, target_audio, predicted_audio,
                             os.path.join(args.output_folder, f'{sample_name}_spectrograms.png'))
        
        # Compute errors
        rms_error = np.sqrt(np.mean((predicted_audio - target_audio) ** 2))
        total_rms_error += rms_error
        print(f"  RMS error (vs target): {rms_error:.6f}")
    
    avg_rms_error = total_rms_error / num_samples
    
    print("\n" + "="*80)
    print("CASCADE AUDIO GENERATION COMPLETE")
    print("="*80)
    print(f"\nProcessed {num_samples} samples with {num_cascade_stages}-stage cascade")
    print(f"Average RMS error: {avg_rms_error:.6f}")
    print(f"\nGenerated files in: {args.output_folder}/")
    print(f"  *_1_input.wav         - Original input (16s)")
    print(f"  *_2_target.wav        - Ground truth continuation (16s)")
    print(f"  *_3_predicted.wav     - Cascade model prediction (16s)")
    print(f"  *_waveforms.png       - Waveform visualization")
    print(f"  *_spectrograms.png    - Spectrogram visualization")
    print("\n🎵 Play WAV files in order: input → target → predicted")
    print("📊 View PNG files to compare waveforms and spectrograms")
    print("="*80)


if __name__ == "__main__":
    main()
