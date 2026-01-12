#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Discriminator for GAN-based training

Distinguishes between real music (targets) and generated/fake music (model outputs).
Uses multi-scale architecture to capture both fine-grained and coarse musical features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioDiscriminator(nn.Module):
    """
    Discriminator for encoded audio representations.
    
    Takes encoded audio [B, D, T] and predicts whether it's real music or fake.
    Uses 1D convolutions to analyze temporal patterns in the encoded space.
    
    Architecture:
    - Multi-scale analysis (capture different temporal scales)
    - Spectral normalization for training stability
    - Leaky ReLU activation
    - Global average pooling + final classification
    """
    
    def __init__(self, encoding_dim=128, hidden_dims=[256, 512, 512, 256]):
        """
        Args:
            encoding_dim: Dimension of encoded audio (128 for EnCodec)
            hidden_dims: List of hidden dimensions for conv layers
        """
        super().__init__()
        
        self.encoding_dim = encoding_dim
        
        # Build convolutional layers with increasing receptive field
        layers = []
        in_dim = encoding_dim
        
        for i, out_dim in enumerate(hidden_dims):
            # Kernel size increases with depth (capture larger temporal patterns)
            kernel_size = 3 + (i * 2)  # 3, 5, 7, 9, ...
            stride = 2 if i < len(hidden_dims) - 1 else 1
            padding = kernel_size // 2
            
            layers.append(
                nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, 
                         stride=stride, padding=padding)
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
            
            in_dim = out_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Real vs Fake logit
        )
        
        print("AudioDiscriminator initialized:")
        print(f"  Encoding dim: {encoding_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Encoded audio [B, D, T]
            
        Returns:
            logits: [B, 1] - logits for real (positive) vs fake (negative)
        """
        # Convolutional feature extraction
        features = self.conv_layers(x)  # [B, hidden_dims[-1], T']
        
        # Global pooling
        pooled = self.global_pool(features).squeeze(-1)  # [B, hidden_dims[-1]]
        
        # Classification
        logits = self.classifier(pooled)  # [B, 1]
        
        return logits


def discriminator_loss(real_logits, fake_logits):
    """
    Discriminator loss (binary cross-entropy)
    
    Args:
        real_logits: [B, 1] - logits for real samples
        fake_logits: [B, 1] - logits for fake samples
        
    Returns:
        loss: Scalar discriminator loss
        real_acc: Accuracy on real samples
        fake_acc: Accuracy on fake samples
    """
    # Real samples should have logit > 0 (sigmoid -> close to 1)
    real_loss = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    )
    
    # Fake samples should have logit < 0 (sigmoid -> close to 0)
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    
    total_loss = real_loss + fake_loss
    
    # Compute accuracies
    real_pred = (torch.sigmoid(real_logits) > 0.5).float()
    fake_pred = (torch.sigmoid(fake_logits) < 0.5).float()
    
    real_acc = real_pred.mean().item()
    fake_acc = fake_pred.mean().item()
    
    return total_loss, real_acc, fake_acc


def generator_loss(fake_logits):
    """
    Generator loss (fool the discriminator)
    
    Generator wants discriminator to think fake samples are real.
    
    Args:
        fake_logits: [B, 1] - discriminator logits for generated samples
        
    Returns:
        loss: Scalar generator adversarial loss
    """
    # Generator wants fake samples to be classified as real (logit > 0)
    loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )
    
    return loss


def roll_targets(targets):
    """
    Roll targets by one position in the batch to create positive examples.
    
    This creates a simple augmentation where each sample is paired with
    a different (but still musical) target. The discriminator learns to
    recognize "musical" patterns rather than specific input-target pairs.
    
    Args:
        targets: [B, D, T] - batch of target encodings
        
    Returns:
        rolled: [B, D, T] - targets rolled by 1 position along batch dim
    """
    return torch.roll(targets, shifts=1, dims=0)
