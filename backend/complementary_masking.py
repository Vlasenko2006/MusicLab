#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complementary Masking Strategies for Style Transfer

Makes input and target complementary instead of overlapping,
forcing the model to create musical arrangements rather than simple blending.
"""

import torch
import numpy as np


def apply_complementary_mask(encoded_input, encoded_target, mask_type='temporal', 
                             temporal_segment_frames=150, freq_split_ratio=0.3,
                             channel_keep_ratio=0.5, energy_threshold=0.7):
    """
    Apply complementary masking to make input and target non-overlapping.
    
    Args:
        encoded_input: Input encoding [B, D, T]
        encoded_target: Target encoding [B, D, T]
        mask_type: Type of masking strategy:
            - 'temporal': Alternating time segments (rhythm-based)
            - 'frequency': Input=low freq, Target=high freq (bass vs melody)
            - 'spectral': Random channel dropout (complementary features)
            - 'energy': Input=transients, Target=sustain (attack vs hold)
            - 'hybrid': Combination of temporal + frequency
        temporal_segment_frames: Frames per segment for temporal masking (~150 = 1sec)
        freq_split_ratio: Frequency split point (0.3 = low 30% vs high 70%)
        channel_keep_ratio: Ratio of channels to keep in spectral dropout
        energy_threshold: Percentile threshold for energy-based masking
        
    Returns:
        masked_input: Masked input encoding [B, D, T]
        masked_target: Masked target encoding [B, D, T]
    """
    B, D, T = encoded_input.shape
    device = encoded_input.device
    
    if mask_type == 'temporal':
        # Alternating time segments: Input active on beats 1,3,5... Target on 2,4,6...
        mask = torch.zeros(1, 1, T, device=device)
        
        # Create alternating pattern
        for i in range(0, T, temporal_segment_frames * 2):
            mask[:, :, i:min(i+temporal_segment_frames, T)] = 1.0
        
        # Smooth transitions at segment boundaries (short fade)
        fade_frames = min(10, temporal_segment_frames // 15)
        if fade_frames > 0:
            for i in range(0, T, temporal_segment_frames):
                if i > 0 and i < T:
                    # Fade out previous segment
                    start = max(0, i - fade_frames)
                    for j in range(fade_frames):
                        if start + j < T:
                            mask[:, :, start + j] *= (fade_frames - j) / fade_frames
                    # Fade in next segment
                    for j in range(fade_frames):
                        if i + j < T:
                            mask[:, :, i + j] *= j / fade_frames
        
        masked_input = encoded_input * mask
        masked_target = encoded_target * (1.0 - mask)
        
    elif mask_type == 'frequency':
        # Frequency band split: Input=low freq (bass/rhythm), Target=high freq (melody)
        split_dim = int(D * freq_split_ratio)
        
        # Create frequency masks
        input_mask = torch.ones(1, D, 1, device=device)
        input_mask[:, split_dim:, :] = 0.2  # Attenuate high frequencies in input
        
        target_mask = torch.ones(1, D, 1, device=device)
        target_mask[:, :split_dim, :] = 0.2  # Attenuate low frequencies in target
        
        masked_input = encoded_input * input_mask
        masked_target = encoded_target * target_mask
        
    elif mask_type == 'spectral':
        # Random channel dropout: Each source gets random 50% of channels
        channels = torch.randperm(D, device=device)
        split = int(D * channel_keep_ratio)
        
        input_channels = channels[:split]
        target_channels = channels[split:]
        
        # Create masks
        input_mask = torch.zeros(1, D, 1, device=device)
        input_mask[:, input_channels, :] = 1.0
        
        target_mask = torch.zeros(1, D, 1, device=device)
        target_mask[:, target_channels, :] = 1.0
        
        masked_input = encoded_input * input_mask
        masked_target = encoded_target * target_mask
        
    elif mask_type == 'energy':
        # Energy-based: Input=transients (attacks), Target=sustain (held notes)
        # Compute energy per time frame
        input_energy = torch.sum(encoded_input ** 2, dim=1, keepdim=True)  # [B, 1, T]
        target_energy = torch.sum(encoded_target ** 2, dim=1, keepdim=True)  # [B, 1, T]
        
        # Compute threshold (e.g., 70th percentile)
        input_threshold = torch.quantile(input_energy, energy_threshold, dim=-1, keepdim=True)
        target_threshold = torch.quantile(target_energy, energy_threshold, dim=-1, keepdim=True)
        
        # Input mask: Keep only high-energy frames (transients)
        input_mask = (input_energy > input_threshold).float()
        
        # Target mask: Keep only low-energy frames (sustain)
        target_mask = (target_energy <= target_threshold).float()
        
        masked_input = encoded_input * input_mask
        masked_target = encoded_target * target_mask
        
    elif mask_type == 'hybrid':
        # Combination: Temporal alternation + Frequency split
        # Temporal mask
        temporal_mask = torch.zeros(1, 1, T, device=device)
        for i in range(0, T, temporal_segment_frames * 2):
            temporal_mask[:, :, i:min(i+temporal_segment_frames, T)] = 1.0
        
        # Frequency mask
        split_dim = int(D * freq_split_ratio)
        freq_mask_input = torch.ones(1, D, 1, device=device)
        freq_mask_input[:, split_dim:, :] = 0.3
        
        freq_mask_target = torch.ones(1, D, 1, device=device)
        freq_mask_target[:, :split_dim, :] = 0.3
        
        # Combine masks
        masked_input = encoded_input * temporal_mask * freq_mask_input
        masked_target = encoded_target * (1.0 - temporal_mask) * freq_mask_target
        
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}. Must be one of: "
                        "temporal, frequency, spectral, energy, hybrid")
    
    return masked_input, masked_target


def visualize_mask_effect(encoded_input, encoded_target, masked_input, masked_target, 
                          save_path=None, mask_type='unknown'):
    """
    Visualize the effect of masking on encoded representations.
    Shows which parts of input/target are kept/suppressed.
    
    Args:
        encoded_input: Original input [B, D, T]
        encoded_target: Original target [B, D, T]
        masked_input: Masked input [B, D, T]
        masked_target: Masked target [B, D, T]
        save_path: Path to save visualization (optional)
        mask_type: Name of masking strategy for title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping mask visualization")
        return
    
    # Take first batch element
    inp = encoded_input[0].cpu().numpy()  # [D, T]
    tgt = encoded_target[0].cpu().numpy()
    m_inp = masked_input[0].cpu().numpy()
    m_tgt = masked_target[0].cpu().numpy()
    
    # Compute masks
    inp_mask = np.abs(m_inp) / (np.abs(inp) + 1e-8)
    tgt_mask = np.abs(m_tgt) / (np.abs(tgt) + 1e-8)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    # Input mask
    im0 = axes[0, 0].imshow(inp_mask, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Input Mask ({mask_type})', fontweight='bold')
    axes[0, 0].set_xlabel('Time Frames')
    axes[0, 0].set_ylabel('Encoding Channels')
    plt.colorbar(im0, ax=axes[0, 0], label='Keep Ratio')
    
    # Target mask
    im1 = axes[0, 1].imshow(tgt_mask, aspect='auto', cmap='plasma', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Target Mask ({mask_type})', fontweight='bold')
    axes[0, 1].set_xlabel('Time Frames')
    axes[0, 1].set_ylabel('Encoding Channels')
    plt.colorbar(im1, ax=axes[0, 1], label='Keep Ratio')
    
    # Combined mask (shows overlap)
    combined = np.minimum(inp_mask + tgt_mask, 1.0)
    im2 = axes[1, 0].imshow(combined, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title('Combined Coverage (Green=Complementary, Red=Overlap)', fontweight='bold')
    axes[1, 0].set_xlabel('Time Frames')
    axes[1, 0].set_ylabel('Encoding Channels')
    plt.colorbar(im2, ax=axes[1, 0], label='Total Coverage')
    
    # Statistics
    inp_coverage = np.mean(inp_mask > 0.1)
    tgt_coverage = np.mean(tgt_mask > 0.1)
    overlap = np.mean((inp_mask > 0.1) & (tgt_mask > 0.1))
    
    stats_text = f"Input Coverage: {inp_coverage:.1%}\n"
    stats_text += f"Target Coverage: {tgt_coverage:.1%}\n"
    stats_text += f"Overlap: {overlap:.1%}\n"
    stats_text += f"Complementary: {1.0 - overlap:.1%}"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=14, family='monospace',
                   verticalalignment='center')
    axes[1, 1].set_title('Masking Statistics', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mask visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_mask_description(mask_type):
    """Get human-readable description of mask type"""
    descriptions = {
        'temporal': 'Alternating time segments (Input: beats 1,3,5... | Target: beats 2,4,6...)',
        'frequency': 'Frequency band split (Input: bass/rhythm | Target: melody/vocals)',
        'spectral': 'Random channel dropout (Input: 50% random channels | Target: other 50%)',
        'energy': 'Energy-based split (Input: transients/attacks | Target: sustain/holds)',
        'hybrid': 'Temporal + Frequency (Input: low-freq on odd beats | Target: high-freq on even beats)',
        'none': 'No masking (full overlap, simple blending)'
    }
    return descriptions.get(mask_type, 'Unknown masking strategy')
