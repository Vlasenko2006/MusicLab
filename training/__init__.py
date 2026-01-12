"""
Training utilities for CASCADE audio continuation model.

This package contains:
- losses.py: Loss functions (RMS, STFT, Mel-spectrogram)
- trainer.py: Main training loop
- validator.py: Validation logic
- metrics.py: Metric tracking and logging
"""

from .losses import rms_loss, stft_loss, mel_loss, combined_loss

__all__ = [
    'rms_loss',
    'stft_loss', 
    'mel_loss',
    'combined_loss',
]
