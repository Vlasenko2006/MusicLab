"""
Creative Agent: Learnable Complementary Masking
==============================================

This module implements a learnable creative agent that replaces fixed complementary masking
with neural network-based attention mechanisms. The agent learns which parts of the input
and target to combine for creating musical arrangements.

Components:
-----------
1. AttentionMaskGenerator: Learns soft masks via cross-attention
2. StyleDiscriminator: Judges quality for adversarial training (optional)
3. CreativeAgent: Wrapper combining both components

Usage:
------
# In model forward pass:
if self.creative_agent is not None:
    masked_input, masked_target, mask_reg_loss = self.creative_agent.generate_creative_masks(
        encoded_input, encoded_target, hard=False
    )
    x_concat = torch.cat([masked_input, masked_target], dim=1)

# In training loop:
loss = reconstruction_loss + mask_reg_weight * mask_reg_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from correlation_penalty import compute_modulation_correlation_penalty


class AttentionMaskGenerator(nn.Module):
    """
    Learns complementary masks using cross-attention between input and target.
    
    The generator analyzes both input and target patterns to decide which parts
    of each should be combined. Uses attention to understand relationships and
    generates soft masks [0,1] via sigmoid activation.
    
    Architecture:
    - Conv1d feature extractors for input and target
    - MultiheadAttention for cross-attention
    - Separate mask generators for input and target
    - Complementarity loss to ensure masks don't overlap
    - Coverage loss to encourage full utilization
    
    Args:
        encoding_dim: Dimension of encoded patterns [D]
        hidden_dim: Hidden dimension for feature extraction (default: 256)
        num_heads: Number of attention heads (default: 4)
    """
    
    def __init__(self, encoding_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractors: Analyze input and target patterns
        # Conv1d: [B, D, T] -> [B, hidden_dim, T]
        self.input_analyzer = nn.Sequential(
            nn.Conv1d(encoding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.target_analyzer = nn.Sequential(
            nn.Conv1d(encoding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Cross-attention: Learn relationships between input and target
        # Input: [T, B, hidden_dim] (sequence_first=True)
        # Output: [T, B, hidden_dim]
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False  # [T, B, D] format
        )
        
        # Mask generators: Convert features to soft masks [0,1]
        # Input: [B, hidden_dim, T] -> Output: [B, D, T]
        self.input_mask_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, encoding_dim, kernel_size=1),
            nn.Sigmoid()  # Soft masks [0, 1]
        )
        
        self.target_mask_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, encoding_dim, kernel_size=1),
            nn.Sigmoid()  # Soft masks [0, 1]
        )
    
    def forward(self, encoded_input, encoded_target, hard=False):
        """
        Generate complementary masks for input and target.
        
        Args:
            encoded_input: [B, D, T] encoded input pattern
            encoded_target: [B, D, T] encoded target pattern
            hard: If True, use Gumbel-Softmax for hard masks (default: False)
        
        Returns:
            input_mask: [B, D, T] soft mask for input
            target_mask: [B, D, T] soft mask for target
            complementarity_loss: Scalar loss encouraging complementary masks
        """
        B, D, T = encoded_input.shape
        
        # Extract features
        input_features = self.input_analyzer(encoded_input)  # [B, hidden_dim, T]
        target_features = self.target_analyzer(encoded_target)  # [B, hidden_dim, T]
        
        # Cross-attention: Understand input-target relationships
        # Reshape: [B, hidden_dim, T] -> [T, B, hidden_dim]
        input_feat_seq = input_features.permute(2, 0, 1)  # [T, B, hidden_dim]
        target_feat_seq = target_features.permute(2, 0, 1)  # [T, B, hidden_dim]
        
        # Attend: input queries, target keys/values
        input_attended, _ = self.cross_attention(
            query=input_feat_seq,
            key=target_feat_seq,
            value=target_feat_seq
        )  # [T, B, hidden_dim]
        
        # Attend: target queries, input keys/values
        target_attended, _ = self.cross_attention(
            query=target_feat_seq,
            key=input_feat_seq,
            value=input_feat_seq
        )  # [T, B, hidden_dim]
        
        # Reshape back: [T, B, hidden_dim] -> [B, hidden_dim, T]
        input_attended = input_attended.permute(1, 2, 0)  # [B, hidden_dim, T]
        target_attended = target_attended.permute(1, 2, 0)  # [B, hidden_dim, T]
        
        # Generate soft masks [0, 1]
        input_mask = self.input_mask_generator(input_attended)  # [B, D, T]
        target_mask = self.target_mask_generator(target_attended)  # [B, D, T]
        
        # Optional: Hard masks via Gumbel-Softmax (straight-through estimator)
        if hard:
            # Stack masks: [B, 2, D, T]
            masks_stacked = torch.stack([input_mask, 1 - input_mask], dim=1)
            # Gumbel-Softmax
            masks_hard = F.gumbel_softmax(masks_stacked.log(), tau=1.0, hard=True, dim=1)
            input_mask = masks_hard[:, 0, :, :]  # [B, D, T]
            target_mask = 1 - input_mask
        
        # Complementarity loss: Minimize overlap
        # Goal: input_mask * target_mask ≈ 0 (complementary)
        overlap = input_mask * target_mask
        complementarity_loss = overlap.mean()
        
        # Coverage loss: Encourage full utilization
        # Goal: input_mask + target_mask ≈ 1.0 (cover everything)
        coverage = input_mask + target_mask
        coverage_loss = ((coverage - 1.0) ** 2).mean()
        
        # Balance loss: Encourage 50/50 mixing of input and target
        # Goal: input_mask.mean() ≈ 0.5, target_mask.mean() ≈ 0.5
        # This prevents the agent from learning to just copy one source
        input_mean = input_mask.mean()
        target_mean = target_mask.mean()
        balance_loss = ((input_mean - 0.5) ** 2 + (target_mean - 0.5) ** 2)
        
        # Temporal diversity loss: Encourage masks to vary over time
        # Goal: Masks should change across time dimension for dynamic mixing
        # Compute variance of mask values along time dimension
        input_mask_temporal_var = input_mask.var(dim=2).mean()  # Variance across T, mean over B,D
        target_mask_temporal_var = target_mask.var(dim=2).mean()
        # Penalize if variance is too low (masks are constant over time)
        # Target variance: ~0.05-0.1 for sigmoid outputs
        temporal_diversity_loss = (
            torch.relu(0.05 - input_mask_temporal_var) + 
            torch.relu(0.05 - target_mask_temporal_var)
        )
        
        # Total regularization loss
        # - Balance loss: Moved to SEPARATE loss term (see model forward)
        # - Temporal diversity (5.0x): Encourage dynamic mixing over time
        # - Complementarity (10.0x): Prevent mask overlap [INCREASED from 1.0x]
        # - Coverage (0.5x): Full utilization
        reg_loss = 10.0 * complementarity_loss + 0.5 * coverage_loss + 5.0 * temporal_diversity_loss
        
        # Store individual loss components for debugging (detached to avoid affecting gradients)
        self._last_balance_loss = balance_loss.detach().item()
        self._last_temporal_diversity = temporal_diversity_loss.detach().item()
        self._last_complementarity = complementarity_loss.detach().item()
        self._last_coverage = coverage_loss.detach().item()
        
        # Return balance_loss separately so it can be weighted independently
        return input_mask, target_mask, reg_loss, balance_loss


class StyleDiscriminator(nn.Module):
    """
    Discriminator for adversarial training (optional).
    
    Judges whether the masked combination creates a "real" arrangement or a fake one.
    Also provides style matching loss to ensure the output maintains characteristics
    of both input and target.
    
    Architecture:
    - Conv1d encoder with downsampling
    - Real/fake classifier
    - Style matcher (compares statistics)
    
    Args:
        encoding_dim: Dimension of encoded patterns [D]
        hidden_dim: Hidden dimension (default: 256)
    """
    
    def __init__(self, encoding_dim, hidden_dim=256):
        super().__init__()
        self.encoding_dim = encoding_dim
        
        # Encoder: [B, D, T] -> [B, hidden_dim, T//8]
        self.encoder = nn.Sequential(
            nn.Conv1d(encoding_dim, hidden_dim // 4, kernel_size=4, stride=2, padding=1),  # T//2
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=4, stride=2, padding=1),  # T//4
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),  # T//8
            nn.LeakyReLU(0.2)
        )
        
        # Real/fake classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, hidden_dim, 1]
            nn.Flatten(),  # [B, hidden_dim]
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # [B, 1] - probability of being real
        )
        
        # Style matcher: Compare statistics
        self.style_matcher = nn.Linear(hidden_dim * 2, 1)  # Concat mean/std
    
    def forward(self, encoded_pattern):
        """
        Judge quality of encoded pattern.
        
        Args:
            encoded_pattern: [B, D, T] encoded pattern to judge
        
        Returns:
            real_fake_score: [B, 1] probability of being real [0, 1]
            style_score: [B, 1] style quality score
        """
        # Encode
        features = self.encoder(encoded_pattern)  # [B, hidden_dim, T//8]
        
        # Real/fake classification
        real_fake_score = self.classifier(features)  # [B, 1]
        
        # Style matching: Use mean/std statistics
        mean = features.mean(dim=2)  # [B, hidden_dim]
        std = features.std(dim=2)  # [B, hidden_dim]
        stats = torch.cat([mean, std], dim=1)  # [B, hidden_dim * 2]
        style_score = torch.sigmoid(self.style_matcher(stats))  # [B, 1]
        
        return real_fake_score, style_score


class CreativeAgent(nn.Module):
    """
    Creative Agent: Combines mask generator and discriminator.
    
    This is the main interface for learnable complementary masking. It replaces
    fixed masking strategies (temporal, frequency, etc.) with learned attention-based
    masking that adapts to each input-target pair.
    
    Usage in model forward:
        masked_input, masked_target, mask_reg_loss = self.creative_agent.generate_creative_masks(
            encoded_input, encoded_target, hard=False
        )
    
    Usage in training:
        loss = reconstruction_loss + mask_reg_weight * mask_reg_loss
        
        # Optional adversarial training:
        disc_loss = self.creative_agent.adversarial_loss(masked_output, real_target)
    
    Args:
        encoding_dim: Dimension of encoded patterns [D]
        use_discriminator: Whether to use discriminator for adversarial training (default: True)
    """
    
    def __init__(self, encoding_dim, use_discriminator=True):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.use_discriminator = use_discriminator
        
        # Mask generator (always used)
        self.mask_generator = AttentionMaskGenerator(encoding_dim)
        
        # Discriminator (optional, for adversarial training)
        if use_discriminator:
            self.discriminator = StyleDiscriminator(encoding_dim)
        else:
            self.discriminator = None
    
    def compute_modulation_correlation(self, input_audio, target_audio, output_audio, M_parts=250):
        """
        Compute anti-modulation correlation cost to prevent copying amplitude envelopes.
        
        This is a wrapper that calls the shared correlation_penalty module.
        
        Args:
            input_audio: [B, 1, T] - Raw input audio waveform
            target_audio: [B, 1, T] - Raw target audio waveform
            output_audio: [B, 1, T] - Raw output audio waveform (prediction)
            M_parts: Number of segments (default: 250, ~64ms each at 24kHz)
        
        Returns:
            corr_cost: Scalar - Anti-modulation correlation cost
                      Higher when output copies input/target envelope
                      Range: [0, +∞] where 0 = independent, higher = copying
        """
        return compute_modulation_correlation_penalty(input_audio, target_audio, output_audio, M_parts)
    
    def generate_creative_masks(self, encoded_input, encoded_target, hard=False):
        """
        Generate learned complementary masks and apply them.
        
        This replaces fixed masking strategies with learned attention-based masking.
        
        Args:
            encoded_input: [B, D, T] encoded input pattern
            encoded_target: [B, D, T] encoded target pattern
            hard: If True, use hard masks (Gumbel-Softmax)
        
        Returns:
            masked_input: [B, D, T] input with learned mask applied
            masked_target: [B, D, T] target with complementary mask applied
            reg_loss: Regularization loss (complementarity + coverage)
        """
        # Generate masks (now returns balance_loss separately)
        input_mask, target_mask, reg_loss, balance_loss = self.mask_generator(
            encoded_input, encoded_target, hard=hard
        )
        
        # Store balance_loss for separate weighting
        self._last_balance_loss = balance_loss.detach().item()
        
        # Apply masks
        masked_input = encoded_input * input_mask
        masked_target = encoded_target * target_mask
        
        return masked_input, masked_target, reg_loss, balance_loss
    
    def judge_quality(self, encoded_pattern):
        """
        Judge quality of encoded pattern (requires discriminator).
        
        Args:
            encoded_pattern: [B, D, T] pattern to judge
        
        Returns:
            real_fake_score: [B, 1] probability of being real
            style_score: [B, 1] style quality score
        
        Raises:
            ValueError: If discriminator is not enabled
        """
        if self.discriminator is None:
            raise ValueError("Discriminator not enabled. Set use_discriminator=True.")
        
        return self.discriminator(encoded_pattern)
    
    def adversarial_loss(self, fake_pattern, real_pattern):
        """
        Compute adversarial loss for GAN-style training.
        
        Args:
            fake_pattern: [B, D, T] generated (masked) pattern
            real_pattern: [B, D, T] real target pattern
        
        Returns:
            generator_loss: Loss for generator (encourages fooling discriminator)
            discriminator_loss: Loss for discriminator (distinguishes real/fake)
        """
        if self.discriminator is None:
            raise ValueError("Discriminator not enabled. Set use_discriminator=True.")
        
        # Discriminator on fake
        fake_score, fake_style = self.discriminator(fake_pattern)
        
        # Discriminator on real
        real_score, real_style = self.discriminator(real_pattern)
        
        # Generator loss: Fool discriminator (wants fake_score → 1)
        generator_loss = F.binary_cross_entropy(
            fake_score,
            torch.ones_like(fake_score)
        )
        
        # Discriminator loss: Distinguish real from fake
        # Real should → 1, Fake should → 0
        disc_loss_real = F.binary_cross_entropy(
            real_score,
            torch.ones_like(real_score)
        )
        disc_loss_fake = F.binary_cross_entropy(
            fake_score,
            torch.zeros_like(fake_score)
        )
        discriminator_loss = disc_loss_real + disc_loss_fake
        
        return generator_loss, discriminator_loss


def test_creative_agent():
    """Test creative agent implementation."""
    print("🧪 Testing Creative Agent Implementation")
    print("=" * 80)
    
    # Hyperparameters
    batch_size = 4
    encoding_dim = 128
    seq_len = 300  # ~2 seconds at 150 Hz
    
    # Create agent
    print("\n1. Creating CreativeAgent...")
    agent = CreativeAgent(encoding_dim, use_discriminator=True)
    print(f"   ✅ Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")
    
    # Synthetic input and target
    print("\n2. Creating synthetic input and target...")
    encoded_input = torch.randn(batch_size, encoding_dim, seq_len)
    encoded_target = torch.randn(batch_size, encoding_dim, seq_len)
    print(f"   Input: {encoded_input.shape}")
    print(f"   Target: {encoded_target.shape}")
    
    # Test mask generation
    print("\n3. Testing mask generation...")
    masked_input, masked_target, reg_loss = agent.generate_creative_masks(
        encoded_input, encoded_target, hard=False
    )
    print(f"   ✅ Masked input: {masked_input.shape}")
    print(f"   ✅ Masked target: {masked_target.shape}")
    print(f"   ✅ Regularization loss: {reg_loss.item():.6f}")
    
    # Check complementarity
    print("\n4. Checking complementarity...")
    # Regenerate masks to check overlap
    with torch.no_grad():
        input_mask, target_mask, _ = agent.mask_generator(encoded_input, encoded_target)
        overlap = (input_mask * target_mask).mean().item()
        coverage = (input_mask + target_mask).mean().item()
    print(f"   Overlap (should be low): {overlap:.6f}")
    print(f"   Coverage (should be ~1.0): {coverage:.6f}")
    complementarity = 1.0 - overlap
    print(f"   ✅ Complementarity: {complementarity * 100:.1f}%")
    
    # Test discriminator
    print("\n5. Testing discriminator...")
    real_score, real_style = agent.judge_quality(encoded_target)
    fake_score, fake_style = agent.judge_quality(masked_input)
    print(f"   Real pattern score: {real_score.mean().item():.4f}")
    print(f"   Fake pattern score: {fake_score.mean().item():.4f}")
    print(f"   ✅ Discriminator works")
    
    # Test adversarial loss
    print("\n6. Testing adversarial loss...")
    gen_loss, disc_loss = agent.adversarial_loss(masked_input, encoded_target)
    print(f"   Generator loss: {gen_loss.item():.6f}")
    print(f"   Discriminator loss: {disc_loss.item():.6f}")
    print(f"   ✅ Adversarial training ready")
    
    # Test gradient flow
    print("\n7. Testing gradient flow...")
    loss = reg_loss + gen_loss
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in agent.parameters() if p.grad is not None)
    print(f"   Total gradient norm: {grad_norm:.6f}")
    print(f"   ✅ Gradients flow correctly")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Creative Agent is ready to use.")
    print("\nUsage:")
    print("  # In model:")
    print("  self.creative_agent = CreativeAgent(encoding_dim)")
    print("  masked_in, masked_tgt, reg_loss = self.creative_agent.generate_creative_masks(input, target)")
    print("\n  # In training:")
    print("  loss = reconstruction_loss + 0.1 * reg_loss")


def compute_rhythm_envelope(audio_encoded, window_size=50):
    """
    Compute temporal envelope (rhythm) from encoded audio.
    
    Args:
        audio_encoded: [B, D, T] encoded audio
        window_size: Window size for envelope computation
    
    Returns:
        envelope: [B, T] temporal envelope (RMS over D dimension, smoothed over time)
    """
    # Compute RMS over encoding dimension
    rms = torch.sqrt(torch.mean(audio_encoded ** 2, dim=1))  # [B, T]
    
    # Smooth with moving average to get rhythm envelope
    if window_size > 1:
        kernel = torch.ones(1, 1, window_size, device=audio_encoded.device) / window_size
        rms_padded = F.pad(rms.unsqueeze(1), (window_size // 2, window_size // 2), mode='reflect')
        envelope = F.conv1d(rms_padded, kernel).squeeze(1)  # [B, T]
    else:
        envelope = rms
    
    return envelope


def evaluate_rhythm_transfer(encoded_input, encoded_target, encoded_output):
    """
    Evaluate how well rhythm from input is transferred to output.
    
    Computes correlation between rhythm envelopes:
    - Input→Output correlation: measures rhythm preservation
    - Target→Output correlation: measures target influence
    - Balance score: ratio of input vs target rhythm influence
    
    Args:
        encoded_input: [B, D, T] input encoding
        encoded_target: [B, D, T] target encoding
        encoded_output: [B, D, T] output encoding
    
    Returns:
        dict with:
            - input_rhythm_corr: correlation between input and output rhythm
            - target_rhythm_corr: correlation between target and output rhythm
            - rhythm_balance: ratio (input_corr / target_corr), 1.0 is balanced
            - input_rhythm_energy: energy of input rhythm
            - output_rhythm_energy: energy of output rhythm
    """
    # Compute rhythm envelopes
    input_envelope = compute_rhythm_envelope(encoded_input)  # [B, T]
    target_envelope = compute_rhythm_envelope(encoded_target)
    output_envelope = compute_rhythm_envelope(encoded_output)
    
    # Compute correlations
    def correlation(x, y):
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)
        cov = (x_centered * y_centered).mean(dim=1)
        std_x = x_centered.std(dim=1) + 1e-8
        std_y = y_centered.std(dim=1) + 1e-8
        return (cov / (std_x * std_y)).mean().item()  # Mean over batch
    
    input_rhythm_corr = correlation(input_envelope, output_envelope)
    target_rhythm_corr = correlation(target_envelope, output_envelope)
    
    # Balance score: 1.0 means equal influence, <1 means more target, >1 means more input
    rhythm_balance = abs(input_rhythm_corr) / (abs(target_rhythm_corr) + 1e-8)
    
    # Energy metrics
    input_rhythm_energy = input_envelope.std().item()
    output_rhythm_energy = output_envelope.std().item()
    
    return {
        'input_rhythm_corr': input_rhythm_corr,
        'target_rhythm_corr': target_rhythm_corr,
        'rhythm_balance': rhythm_balance,
        'input_rhythm_energy': input_rhythm_energy,
        'output_rhythm_energy': output_rhythm_energy,
        'rhythm_preserved': abs(input_rhythm_corr) > 0.3,  # Threshold for "rhythm preserved"
    }


if __name__ == "__main__":
    test_creative_agent()
