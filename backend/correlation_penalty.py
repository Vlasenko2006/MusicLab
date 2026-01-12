#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Penalty for Creative Audio Generation

Prevents models from copying amplitude envelopes from input/target sources
by computing correlation between amplitude modulation patterns and applying
exponential penalties.

Used by both Creative Agent (masking) and Compositional Agent (decomposition).
"""

import torch


def compute_amplitude_envelopes(audio_tensor, M_parts):
    """
    Extract amplitude envelope by segmenting audio and finding max per segment.
    
    Args:
        audio_tensor: [B, T] - audio waveform
        M_parts: Number of segments
        
    Returns:
        envelope: [B, M_parts] - max amplitude per segment
    """
    B, T = audio_tensor.shape
    
    # Take absolute values
    abs_audio = torch.abs(audio_tensor)
    
    # Calculate segment size and truncate
    segment_size = T // M_parts
    useful_length = segment_size * M_parts
    abs_audio = abs_audio[:, :useful_length]
    
    # Reshape and compute max per segment
    abs_audio = abs_audio.view(B, M_parts, segment_size)
    envelope = abs_audio.max(dim=2)[0]  # [B, M_parts]
    
    return envelope


def compute_pearson_correlation(x, y, M_parts):
    """
    Compute Pearson correlation coefficient for each batch element.
    
    Args:
        x, y: [B, M_parts] - two sequences to correlate
        M_parts: Number of segments (for normalization)
        
    Returns:
        corr: [B] - correlation coefficient per batch element
    """
    # Center the data
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    
    # Compute covariance
    cov = (x_centered * y_centered).sum(dim=1) / (M_parts - 1)
    
    # Compute standard deviations
    std_x = torch.sqrt((x_centered ** 2).sum(dim=1) / (M_parts - 1))
    std_y = torch.sqrt((y_centered ** 2).sum(dim=1) / (M_parts - 1))
    
    # Compute correlation (eps for numerical stability)
    corr = cov / (std_x * std_y + 1e-8)
    corr = torch.clamp(corr, -1.0, 1.0)
    
    return corr


def correlation_to_exponential_cost(correlation):
    """
    Convert correlation coefficient to quadratic penalty.
    
    Uses formula: cost = corr²
    - Low correlation (0) → low cost (0)
    - High correlation (1) → high cost (1)
    
    Quadratic penalty is numerically stable with bounded gradients.
    Gradient: d(corr²)/d(corr) = 2*corr (max = 2.0 at corr=1.0)
    
    This penalizes outputs that copy input/target patterns while
    maintaining gradient stability.
    
    Args:
        correlation: [B] - correlation coefficients in range [-1, 1]
        
    Returns:
        cost: [B] - quadratic penalty values in range [0, 1]
    """
    corr_abs = torch.abs(correlation)
    cost = corr_abs ** 2  # Quadratic: numerically stable, bounded gradients
    
    return cost


def compute_modulation_correlation_penalty(input_audio, target_audio, output_audio, M_parts=250):
    """
    Compute anti-modulation correlation penalty to prevent copying amplitude envelopes.
    
    Penalizes the model for copying the amplitude modulation pattern from
    either the input or target, forcing it to create truly novel patterns.
    
    Args:
        input_audio: [B, 1, T] - Raw input audio waveform
        target_audio: [B, 1, T] - Raw target audio waveform
        output_audio: [B, 1, T] - Raw output audio waveform (prediction)
        M_parts: Number of segments (default: 250, ~64ms each at 24kHz)
    
    Returns:
        corr_cost: Scalar - Anti-modulation correlation cost
                  Higher when output copies input/target envelope
                  Range: [0, +∞] where 0 = independent, higher = copying
    
    Algorithm:
        1. Take absolute value of waveforms (amplitude)
        2. Split into M_parts segments
        3. Compute max amplitude in each segment (envelope)
        4. Compute Pearson correlation between envelopes
        5. Cost = -ln(1 - |corr_input|) - ln(1 - |corr_target|)
    
    Example:
        >>> input_audio = torch.randn(4, 1, 384000)  # 4 samples, 16s each at 24kHz
        >>> target_audio = torch.randn(4, 1, 384000)
        >>> output_audio = input_audio * 0.7 + target_audio * 0.3  # Mix
        >>> penalty = compute_modulation_correlation_penalty(input_audio, target_audio, output_audio)
        >>> print(f"Correlation penalty: {penalty.item():.4f}")
    """
    B, C, T = input_audio.shape
    assert C == 1, f"Expected mono audio, got {C} channels"
    
    # Remove channel dimension: [B, 1, T] -> [B, T]
    input_audio = input_audio.squeeze(1)
    target_audio = target_audio.squeeze(1)
    output_audio = output_audio.squeeze(1)
    
    # STABILITY: Check for NaN in decoded audio
    if torch.isnan(output_audio).any():
        print("⚠️  WARNING: NaN in output audio, correlation penalty set to 0")
        return torch.tensor(0.0, device=output_audio.device)
    
    # Extract amplitude envelopes
    envelope_input = compute_amplitude_envelopes(input_audio, M_parts)
    envelope_target = compute_amplitude_envelopes(target_audio, M_parts)
    envelope_output = compute_amplitude_envelopes(output_audio, M_parts)
    
    # Compute correlations
    corr_input = compute_pearson_correlation(envelope_input, envelope_output, M_parts)
    corr_target = compute_pearson_correlation(envelope_target, envelope_output, M_parts)
    
    # Convert to exponential cost
    cost_input = correlation_to_exponential_cost(corr_input)
    cost_target = correlation_to_exponential_cost(corr_target)
    
    # Total cost: average over batch
    corr_cost = (cost_input + cost_target).mean()  # scalar
    
    return corr_cost


if __name__ == "__main__":
    """Test correlation penalty functions"""
    
    print("="*80)
    print("Testing Correlation Penalty Functions")
    print("="*80)
    
    # Test 1: Independent signals (low correlation)
    print("\nTest 1: Independent random signals")
    input_audio = torch.randn(4, 1, 384000)
    target_audio = torch.randn(4, 1, 384000)
    output_audio = torch.randn(4, 1, 384000)
    
    penalty = compute_modulation_correlation_penalty(input_audio, target_audio, output_audio)
    print(f"  Penalty (should be low): {penalty.item():.4f}")
    
    # Test 2: Output copies input (high correlation with input)
    print("\nTest 2: Output = Input (copying)")
    output_audio = input_audio.clone()
    penalty = compute_modulation_correlation_penalty(input_audio, target_audio, output_audio)
    print(f"  Penalty (should be high): {penalty.item():.4f}")
    
    # Test 3: Output copies target (high correlation with target)
    print("\nTest 3: Output = Target (copying)")
    output_audio = target_audio.clone()
    penalty = compute_modulation_correlation_penalty(input_audio, target_audio, output_audio)
    print(f"  Penalty (should be high): {penalty.item():.4f}")
    
    # Test 4: Output mixes both (medium correlation)
    print("\nTest 4: Output = 0.5*Input + 0.5*Target (mixing)")
    output_audio = 0.5 * input_audio + 0.5 * target_audio
    penalty = compute_modulation_correlation_penalty(input_audio, target_audio, output_audio)
    print(f"  Penalty (should be medium): {penalty.item():.4f}")
    
    # Test 5: Different mixing ratios
    print("\nTest 5: Different mixing ratios")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        output_audio = alpha * input_audio + (1 - alpha) * target_audio
        penalty = compute_modulation_correlation_penalty(input_audio, target_audio, output_audio)
        print(f"  α={alpha:.2f} (Input={alpha:.0%}, Target={1-alpha:.0%}): penalty={penalty.item():.4f}")
    
    print("\n" + "="*80)
    print("✓ Correlation penalty tests complete!")
    print("="*80)
