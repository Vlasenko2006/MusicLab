"""
Loss functions for audio continuation training.

Includes:
- RMS loss: Simple time-domain reconstruction
- STFT loss: Multi-resolution spectral convergence
- Mel loss: Perceptual mel-spectrogram similarity
- Combined loss: Weighted combination of all losses
"""

import torch
import torchaudio


def rms_loss(output, target):
    """
    RMS (Root Mean Square) loss in audio space.
    
    Args:
        output: Predicted audio [B, T]
        target: Ground truth audio [B, T]
        
    Returns:
        Scalar RMS loss
    """
    return torch.sqrt(torch.mean((output - target) ** 2))


def stft_loss(y_pred, y_target, fft_sizes=[512, 1024, 2048], 
              hop_sizes=[128, 256, 512], win_sizes=[512, 1024, 2048]):
    """
    Multi-resolution STFT loss (spectral convergence + log magnitude).
    
    Based on Parallel WaveGAN and HiFi-GAN papers. Measures spectral
    similarity at multiple time-frequency resolutions.
    
    Args:
        y_pred: Predicted audio [B, T]
        y_target: Ground truth audio [B, T]
        fft_sizes: List of FFT sizes for multi-resolution analysis
        hop_sizes: List of hop sizes
        win_sizes: List of window sizes
        
    Returns:
        Combined spectral loss (scalar)
    """
    spectral_convergence_loss = 0.0
    log_magnitude_loss = 0.0
    
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # Compute STFT with Hann window
        window = torch.hann_window(win_size).to(y_pred.device)
        
        stft_pred = torch.stft(
            y_pred, 
            n_fft=fft_size, 
            hop_length=hop_size, 
            win_length=win_size,
            window=window,
            return_complex=True
        )
        
        stft_target = torch.stft(
            y_target, 
            n_fft=fft_size, 
            hop_length=hop_size, 
            win_length=win_size,
            window=window,
            return_complex=True
        )
        
        # Compute magnitude spectrograms
        mag_pred = torch.abs(stft_pred)
        mag_target = torch.abs(stft_target)
        
        # Spectral convergence: ||mag_pred - mag_target||_F / ||mag_target||_F
        spectral_convergence = torch.norm(mag_pred - mag_target, p='fro') / (torch.norm(mag_target, p='fro') + 1e-6)
        spectral_convergence_loss += spectral_convergence
        
        # Log magnitude distance: ||log(mag_pred) - log(mag_target)||_1
        log_mag_distance = torch.mean(torch.abs(torch.log(mag_pred + 1e-5) - torch.log(mag_target + 1e-5)))
        log_magnitude_loss += log_mag_distance
    
    # Average over all resolutions
    num_scales = len(fft_sizes)
    return (spectral_convergence_loss + log_magnitude_loss) / num_scales


def mel_loss(y_pred, y_target, sample_rate=24000, n_fft=1024, n_mels=80, 
             hop_length=256, fmin=0.0, fmax=8000.0):
    """
    Mel-spectrogram L1 loss - captures perceptual similarity.
    
    Converts audio to mel-scale frequency representation and computes
    L1 distance in log-mel space, which correlates well with perceptual quality.
    
    Args:
        y_pred: Predicted audio [B, T]
        y_target: Ground truth audio [B, T]
        sample_rate: Audio sample rate (Hz)
        n_fft: FFT size
        n_mels: Number of mel frequency bins
        hop_length: Hop length for STFT
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz), None for Nyquist
        
    Returns:
        Mel loss (scalar)
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        f_min=fmin,
        f_max=fmax
    ).to(y_pred.device)
    
    # Compute mel spectrograms
    mel_pred = mel_transform(y_pred)
    mel_target = mel_transform(y_target)
    
    # Log mel spectrograms and compute L1 distance
    log_mel_pred = torch.log(mel_pred + 1e-5)
    log_mel_target = torch.log(mel_target + 1e-5)
    
    return torch.mean(torch.abs(log_mel_pred - log_mel_target))


def combined_loss(output_audio, input_audio, target_audio, 
                  weight_input, weight_target, weight_spectral, weight_mel,
                  weight_correlation=0.0):
    """
    Combined loss with RMS + spectral + mel components + correlation penalty.
    
    Provides flexible weighting of different loss components for
    balancing reconstruction quality, spectral accuracy, and perceptual similarity.
    
    Args:
        output_audio: Model output decoded to audio [B, 1, T]
        input_audio: Original input audio [B, 1, T]
        target_audio: Target continuation audio [B, 1, T]
        weight_input: Weight for RMS(output, input) loss
        weight_target: Weight for RMS(output, target) loss
        weight_spectral: Weight for multi-resolution STFT loss (now compares to BOTH)
        weight_mel: Weight for mel-spectrogram loss (now compares to BOTH)
        weight_correlation: Weight for correlation penalty (decorrelate from both sources)
        
    Returns:
        tuple: (total_loss, rms_input_loss, rms_target_loss, spectral_loss, mel_loss_value, corr_penalty)
    """
    # Squeeze to [B, T] for loss computation
    output_audio_squeezed = output_audio.squeeze(1)
    input_audio_squeezed = input_audio.squeeze(1)
    target_audio_squeezed = target_audio.squeeze(1)
    
    # RMS losses
    rms_input_loss = 0.0
    rms_target_loss = 0.0
    if weight_input > 0:
        rms_input_loss = rms_loss(output_audio_squeezed, input_audio_squeezed)
    if weight_target > 0:
        rms_target_loss = rms_loss(output_audio_squeezed, target_audio_squeezed)
    
    # Spectral loss (NOW computed against BOTH input and target)
    spectral_loss_value = 0.0
    if weight_spectral > 0:
        spectral_loss_input = stft_loss(output_audio_squeezed, input_audio_squeezed)
        spectral_loss_target = stft_loss(output_audio_squeezed, target_audio_squeezed)
        spectral_loss_value = 0.5 * (spectral_loss_input + spectral_loss_target)  # Average both
    
    # Mel loss (NOW computed against BOTH input and target)
    mel_loss_value = 0.0
    if weight_mel > 0:
        mel_loss_input = mel_loss(output_audio_squeezed, input_audio_squeezed)
        mel_loss_target = mel_loss(output_audio_squeezed, target_audio_squeezed)
        mel_loss_value = 0.5 * (mel_loss_input + mel_loss_target)  # Average both
    
    # Correlation penalty (decorrelate from BOTH sources)
    corr_penalty = 0.0
    if weight_correlation > 0:
        from correlation_penalty import compute_modulation_correlation_penalty
        corr_penalty = compute_modulation_correlation_penalty(
            input_audio, target_audio, output_audio, M_parts=250
        )
    
    # Combine all losses
    total_loss = (weight_input * rms_input_loss + 
                  weight_target * rms_target_loss + 
                  weight_spectral * spectral_loss_value + 
                  weight_mel * mel_loss_value +
                  weight_correlation * corr_penalty)
    
    return total_loss, rms_input_loss, rms_target_loss, spectral_loss_value, mel_loss_value, corr_penalty


def spectral_outlier_penalty(audio, sample_rate=24000, n_fft=2048, hop_length=512, 
                              freq_bands=[(1800, 2200), (3800, 4200), (5800, 6200)],
                              percentile=90):
    """
    Penalize spectral outliers in specific frequency bands across batch.
    
    Computes STFT, finds energy in problematic frequency bands (e.g., 2kHz, 4kHz, 6kHz),
    and penalizes values that exceed the batch's typical energy in those bands.
    
    Args:
        audio: Audio tensor [B, T] or [B, 1, T]
        sample_rate: Sample rate in Hz
        n_fft: FFT size
        hop_length: Hop length for STFT
        freq_bands: List of (low_hz, high_hz) tuples defining problematic bands
        percentile: Percentile threshold (penalize above this)
        
    Returns:
        Scalar penalty (higher = more outliers)
    """
    # Ensure [B, T] shape
    if audio.dim() == 3:
        audio = audio.squeeze(1)  # [B, 1, T] -> [B, T]
    
    B, T = audio.shape
    device = audio.device
    
    # Compute STFT: [B, n_fft//2+1, frames]
    window = torch.hann_window(n_fft).to(device)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                      win_length=n_fft, window=window, 
                      return_complex=True, center=True)
    
    # Magnitude spectrogram: [B, n_fft//2+1, frames]
    magnitude = torch.abs(stft)
    
    # Frequency resolution (Hz per bin)
    freq_resolution = sample_rate / n_fft
    
    total_penalty = 0.0
    
    for low_hz, high_hz in freq_bands:
        # Convert Hz to bin indices
        low_bin = int(low_hz / freq_resolution)
        high_bin = int(high_hz / freq_resolution)
        low_bin = max(0, low_bin)
        high_bin = min(magnitude.shape[1], high_bin)
        
        # Extract energy in this frequency band: [B, frames]
        band_energy = magnitude[:, low_bin:high_bin, :].mean(dim=1)  # Average across bins
        
        # Compute batch statistics
        batch_mean = band_energy.mean()
        batch_std = band_energy.std() + 1e-8
        
        # Find outliers (z-score > percentile)
        z_scores = (band_energy - batch_mean) / batch_std
        outlier_mask = z_scores > torch.quantile(z_scores, percentile / 100.0)
        
        # Penalty: squared excess energy
        if outlier_mask.any():
            outlier_energy = band_energy[outlier_mask]
            penalty = torch.mean((outlier_energy - batch_mean) ** 2)
            total_penalty += penalty
    
    return total_penalty / len(freq_bands)  # Average across bands
