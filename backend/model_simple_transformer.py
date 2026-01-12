#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Transformer for Audio Continuation (Pairs)

Input: Single encoded pattern [B, D, T_enc]
Output: Next encoded pattern [B, D, T_enc]

2-Stage Cascade Architecture (SIMPLIFIED - removed problematic stage 2):
- Stage 0: concat(input, target) [256-dim] -> output0
- Stage 1: concat(stage0, target, input) [384-dim] -> final_output

Both stages use FULL 1.0× residual (no weak residual to prevent RMS collapse)
"""

import torch
import torch.nn as nn
import math
from torch.nn.utils import spectral_norm

# Global flag for detailed numerical debugging
DEBUG_NUMERICS = False

def check_tensor_health(tensor, name, stage_info=""):
    """Check for NaN/Inf and print statistics if debugging enabled"""
    if not DEBUG_NUMERICS:
        return True
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"\n🔥 NUMERICAL ISSUE DETECTED: {name} {stage_info}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Has NaN: {has_nan}")
        print(f"   Has Inf: {has_inf}")
        print(f"   Min: {tensor.min().item():.6f}")
        print(f"   Max: {tensor.max().item():.6f}")
        print(f"   Mean: {tensor.mean().item():.6f}")
        print(f"   Std: {tensor.std().item():.6f}")
        return False
    else:
        print(f"✓ {name} {stage_info}: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
        return True

try:
    from creative_agent import CreativeAgent
    CREATIVE_AGENT_AVAILABLE = True
except ImportError:
    CREATIVE_AGENT_AVAILABLE = False
    print("Warning: creative_agent.py not found. Creative agent disabled.")

try:
    from compositional_creative_agent import CompositionalCreativeAgent
    COMPOSITIONAL_AGENT_AVAILABLE = True
except ImportError:
    COMPOSITIONAL_AGENT_AVAILABLE = False
    print("Warning: compositional_creative_agent.py not found. Compositional agent disabled.")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    Transformer for encoded audio continuation with cascade support
    
    Single stage (num_transformer_layers=1):
        Input: [B, D, T_enc]
        Output: [B, D, T_enc]
    
    Cascade (num_transformer_layers>1):
        Stage 1: concat(input, target) [B, 2*D, T_enc] -> output_1 [B, D, T_enc]
        Stage 2+: concat(input, prev_output, noisy_target) [B, 3*D, T_enc] -> output_i [B, D, T_enc]
    """
    
    def __init__(self, encoding_dim=128, nhead=8, num_layers=4, dropout=0.1, num_transformer_layers=1, anti_cheating=0.0, use_creative_agent=False, use_compositional_agent=False):
        """
        Args:
            encoding_dim: EnCodec encoding dimension (128 for 24kHz)
            nhead: Number of attention heads (8 for testing)
            num_layers: Number of internal transformer encoder layers per stage
            dropout: Dropout rate
            num_transformer_layers: Number of cascade stages (1=no cascade, 2+=cascade)
            anti_cheating: Noise level for cascade stages 2+ (0.0=no noise, 1.0=heavy noise)
            use_creative_agent: Use attention-based masking agent (default: False)
            use_compositional_agent: Use compositional creative agent (default: False)
        """
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_transformer_layers = num_transformer_layers
        self.anti_cheating = anti_cheating
        
        # Creative agents (mutually exclusive)
        if use_compositional_agent and num_transformer_layers > 1 and COMPOSITIONAL_AGENT_AVAILABLE:
            # Compositional agent: Extract rhythm/harmony/timbre, compose NEW patterns
            self.creative_agent = CompositionalCreativeAgent(
                encoding_dim=encoding_dim,
                component_dim=64,
                composer_heads=8,
                composer_layers=4,
                novelty_weight=0.1
            )
            self.use_compositional = True
            print("🎼 Compositional Creative Agent ENABLED (rhythm/harmony/timbre decomposition)")
        elif use_creative_agent and num_transformer_layers > 1 and CREATIVE_AGENT_AVAILABLE:
            # Attention-based masking agent (old approach)
            self.creative_agent = CreativeAgent(encoding_dim, use_discriminator=True)
            self.use_compositional = False
            print("🎨 Attention-Based Creative Agent ENABLED (masking)")
        else:
            self.creative_agent = None
            self.use_compositional = False
        
        if num_transformer_layers == 1:
            # Single stage: process input only
            self.d_model = encoding_dim
            
            # Input normalization
            self.input_norm = nn.LayerNorm(encoding_dim)
            
            # Single transformer
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=encoding_dim,
                    nhead=nhead,
                    dim_feedforward=4 * encoding_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation='gelu'
                ),
                num_layers=num_layers
            )
            
            # Post-transformer normalization
            self.post_norm = nn.LayerNorm(encoding_dim)
            
            # Output projection with spectral normalization
            self.output_proj = spectral_norm(nn.Linear(encoding_dim, encoding_dim))
        else:
            # 2-Stage Cascade (simplified from 3-stage to avoid gradient explosion):
            # Stage 0: concat(input, target) -> 256-dim
            # Stage 1: concat(stage0, target, input) -> 384-dim
            self.d_model_stage0 = 2 * encoding_dim  # 256
            self.d_model_stage1 = 3 * encoding_dim  # 384
            
            # Build 2 cascade stages only
            self.cascade_stages = nn.ModuleList()
            
            for stage_idx in range(2):  # Only 2 stages
                # Stage 0: concat(input, target) -> 2*D = 256
                # Stage 1: concat(stage0, target, input) -> 3*D = 384
                d_model = self.d_model_stage0 if stage_idx == 0 else self.d_model_stage1
                
                pos_enc = PositionalEncoding(d_model, dropout)
                input_norm = nn.LayerNorm(d_model)
                
                transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=4 * d_model,
                        dropout=dropout,
                        batch_first=True,
                        activation='gelu'
                    ),
                    num_layers=num_layers
                )
                
                post_norm = nn.LayerNorm(d_model)
                # Apply spectral normalization to constrain gradients
                output_proj = spectral_norm(nn.Linear(d_model, encoding_dim))
                
                self.cascade_stages.append(nn.ModuleDict({
                    'pos_encoding': pos_enc,
                    'input_norm': input_norm,
                    'transformer': transformer,
                    'post_norm': post_norm,
                    'output_proj': output_proj
                }))
        
        print(f"SimpleTransformer initialized:")
        print(f"  Encoding dim: {encoding_dim}")
        print(f"  Attention heads: {nhead}")
        print(f"  Internal transformer layers: {num_layers}")
        print(f"  Cascade stages: 2 (FIXED - removed problematic stage 2)")
        if num_transformer_layers == 1:
            print(f"  d_model: {self.d_model}")
            print(f"  Feedforward dim: {4 * self.d_model}")
        else:
            print(f"  d_model (stage 0): {self.d_model_stage0}")
            print(f"  d_model (stage 1): {self.d_model_stage1}")
            print(f"  Feedforward dim (stage 0): {4 * self.d_model_stage0}")
            print(f"  Feedforward dim (stage 1): {4 * self.d_model_stage1}")
            print(f"  Anti-cheating noise: {self.anti_cheating:.2f}")
        print(f"  Dropout: {dropout}")
        print(f"  Weight initialization: PyTorch default (Xavier/Kaiming)")
    
    def forward(self, encoded_input, encoded_target=None):
        """
        Forward pass with optional cascade processing
        
        Single stage (num_transformer_layers=1):
            Args:
                encoded_input: [B, D, T_enc] encoded input pattern
            Returns:
                output: [B, D, T_enc] predicted encoded pattern
        
        Cascade (num_transformer_layers>1):
            Args:
                encoded_input: [B, D, T_enc] encoded input pattern
                encoded_target: [B, D, T_enc] encoded target pattern (required for stage 1)
            Returns:
                output: [B, D, T_enc] predicted encoded pattern after cascade refinement
        """
        B, D, T_enc = encoded_input.shape
        assert D == self.encoding_dim, f"Expected encoding_dim={self.encoding_dim}, got {D}"
        
        if self.num_transformer_layers == 1:
            # ========== SINGLE STAGE MODE ==========
            # Reshape: [B, D, T_enc] -> [B, T_enc, D]
            x = encoded_input.transpose(1, 2)  # [B, T_enc, D]
            x_residual = x  # Save for residual connection
            
            # Apply input normalization
            x = self.input_norm(x)  # [B, T_enc, D]
            
            # Transform
            transformed = self.transformer(x)  # [B, T_enc, D]
            
            # Apply post-transformer normalization
            transformed = self.post_norm(transformed)  # [B, T_enc, D]
            
            # Output projection with residual connection
            output = self.output_proj(transformed) + x_residual  # [B, T_enc, D]
            
            # Reshape back: [B, T_enc, D] -> [B, D, T_enc]
            output = output.transpose(1, 2)
            
            return output, None
        else:
            # ========== CASCADE MODE ==========
            assert encoded_target is not None, "encoded_target required for cascade mode (num_transformer_layers > 1)"
            
            mask_reg_loss = None
            balance_loss = None
            
            # Apply creative agent (compositional or masking-based)
            if self.creative_agent is not None:
                if self.use_compositional:
                    # Compositional agent: Generate NEW pattern through decomposition
                    creative_output, novelty_loss = self.creative_agent(
                        encoded_input, encoded_target
                    )
                    # Use creative output instead of input for cascade
                    encoded_input_use = creative_output
                    encoded_target_use = encoded_target  # Still use target for cascade stage 1
                    mask_reg_loss = novelty_loss  # Novelty loss replaces mask reg loss
                    
                    # Store statistics for monitoring
                    stats = self.creative_agent.get_component_statistics(
                        encoded_input, encoded_target
                    )
                    self._last_input_rhythm_weight = stats['input_rhythm_weight']
                    self._last_input_harmony_weight = stats['input_harmony_weight']
                    self._last_target_rhythm_weight = stats['target_rhythm_weight']
                    self._last_target_harmony_weight = stats['target_harmony_weight']
                else:
                    # Attention-based masking agent (old approach)
                    # generate_creative_masks returns MASKED encodings, not raw masks!
                    # Now also returns balance_loss separately for independent weighting
                    masked_input, masked_target, mask_reg_loss, balance_loss = self.creative_agent.generate_creative_masks(
                        encoded_input, encoded_target, hard=False
                    )
                    
                    # CRITICAL FIX: Restore RMS after masking to prevent signal suppression
                    # Masking reduces signal energy by ~50%, causing quiet output
                    input_rms_before = torch.sqrt(torch.mean(encoded_input ** 2, dim=(1, 2), keepdim=True) + 1e-8)
                    target_rms_before = torch.sqrt(torch.mean(encoded_target ** 2, dim=(1, 2), keepdim=True) + 1e-8)
                    masked_input_rms = torch.sqrt(torch.mean(masked_input ** 2, dim=(1, 2), keepdim=True) + 1e-8)
                    masked_target_rms = torch.sqrt(torch.mean(masked_target ** 2, dim=(1, 2), keepdim=True) + 1e-8)
                    
                    # Restore original RMS while keeping mask structure
                    masked_input = masked_input * (input_rms_before / masked_input_rms)
                    masked_target = masked_target * (target_rms_before / masked_target_rms)
                    
                    encoded_input_use = masked_input
                    encoded_target_use = masked_target
                    
                    # Get raw masks for monitoring (need to call mask_generator directly)
                    # Note: mask_generator now returns 4 values: (input_mask, target_mask, reg_loss, balance_loss)
                    input_mask, target_mask, _, _ = self.creative_agent.mask_generator(
                        encoded_input, encoded_target, hard=False
                    )
                    self._last_input_mask_mean = input_mask.mean().item()
                    self._last_target_mask_mean = target_mask.mean().item()
                    self._last_mask_overlap = (input_mask * target_mask).mean().item()
            else:
                encoded_input_use = encoded_input
                encoded_target_use = encoded_target
            
            # Apply anti-cheating noise to input (prevents memorization)
            if self.training and self.anti_cheating > 0:
                noise_input = torch.randn_like(encoded_input_use) * self.anti_cheating * encoded_input_use.std()
                noisy_input = encoded_input_use + noise_input
            else:
                noisy_input = encoded_input_use
            
            # Compute residual for blending
            x_residual = 0.5 * (encoded_input_use + encoded_target_use)  # [B, D, T_enc]
            x_residual_transposed = x_residual.transpose(1, 2)  # [B, T_enc, D]
            
            output = None
            num_stages = len(self.cascade_stages)
            
            for stage_idx, stage in enumerate(self.cascade_stages):
                is_last_stage = (stage_idx == num_stages - 1)
                
                if DEBUG_NUMERICS:
                    print(f"\n{'='*80}")
                    print(f"CASCADE STAGE {stage_idx}")
                    print(f"{'='*80}")
                
                if stage_idx == 0:
                    # Stage 1: concat(noisy_input, target) -> transformer -> output1
                    check_tensor_health(noisy_input, "noisy_input", f"[stage {stage_idx}]")
                    check_tensor_health(encoded_target_use, "encoded_target_use", f"[stage {stage_idx}]")
                    
                    x_concat = torch.cat([noisy_input, encoded_target_use], dim=1)  # [B, 2*D, T_enc]
                    check_tensor_health(x_concat, "x_concat (after cat)", f"[stage {stage_idx}]")
                else:
                    # Stage 2+: concat(noisy_input, prev_output, noisy_target) -> transformer -> output_i
                    check_tensor_health(output, "prev_output", f"[stage {stage_idx}]")
                    
                    # Apply anti-cheating noise to target (prevents copying)
                    if self.training and self.anti_cheating > 0:
                        noise = torch.randn_like(encoded_target_use) * self.anti_cheating * encoded_target_use.std()
                        noisy_target = encoded_target_use + noise
                    else:
                        noisy_target = encoded_target_use
                    
                    check_tensor_health(noisy_target, "noisy_target", f"[stage {stage_idx}]")
                    
                    # Compute residual for all stages
                    x_residual = (encoded_input_use + output + encoded_target_use) / 3.0  # [B, D, T_enc]
                    x_residual_transposed = x_residual.transpose(1, 2)  # [B, T_enc, D]
                    
                    check_tensor_health(x_residual, "x_residual", f"[stage {stage_idx}]")
                    
                    x_concat = torch.cat([noisy_input, output, noisy_target], dim=1)  # [B, 3*D, T_enc]
                    check_tensor_health(x_concat, "x_concat (after cat)", f"[stage {stage_idx}]")
                
                # Reshape: [B, 2*D or 3*D, T_enc] -> [B, T_enc, 2*D or 3*D]
                x = x_concat.transpose(1, 2)  # [B, T_enc, 2*D] or [B, T_enc, 3*D]
                check_tensor_health(x, "x (after transpose)", f"[stage {stage_idx}]")
                
                # Positional encoding
                x = stage['pos_encoding'](x)  # [B, T_enc, d_model]
                check_tensor_health(x, "x (after pos_encoding)", f"[stage {stage_idx}]")
                
                # Input normalization
                x = stage['input_norm'](x)  # [B, T_enc, d_model]
                check_tensor_health(x, "x (after input_norm)", f"[stage {stage_idx}]")
                
                # Transform
                transformed = stage['transformer'](x)  # [B, T_enc, d_model]
                check_tensor_health(transformed, "transformed (after transformer)", f"[stage {stage_idx}]")
                
                # Post-transformer normalization
                transformed = stage['post_norm'](transformed)  # [B, T_enc, d_model]
                check_tensor_health(transformed, "transformed (after post_norm)", f"[stage {stage_idx}]")
                
                # Output projection: d_model -> D
                output_projected = stage['output_proj'](transformed)  # [B, T_enc, D]
                check_tensor_health(output_projected, "output_projected (after output_proj)", f"[stage {stage_idx}]")
                
                # Add full residual connection for both stages (1.0x)
                # No weak residual needed since we eliminated the problematic stage 2
                output_projected = output_projected + x_residual_transposed  # [B, T_enc, D]
                
                check_tensor_health(output_projected, "output_projected (after residual)", f"[stage {stage_idx}]")
                
                # Reshape to [B, D, T_enc] for RMS scaling
                output_projected_transposed = output_projected.transpose(1, 2)  # [B, D, T_enc]
                
                # Apply FIXED RMS scaling AFTER adding residual
                # This ensures final output (projection + residual) has correct magnitude
                # Using fixed 50/50 weighting of input and target RMS (stable, no learnable parameters)
                input_rms = torch.sqrt(torch.mean(encoded_input_use ** 2, dim=(1, 2), keepdim=True) + 1e-8)  # [B, 1, 1]
                target_rms = torch.sqrt(torch.mean(encoded_target_use ** 2, dim=(1, 2), keepdim=True) + 1e-8)  # [B, 1, 1]
                output_rms = torch.sqrt(torch.mean(output_projected_transposed ** 2, dim=(1, 2), keepdim=True) + 1e-8)  # [B, 1, 1]
                
                if DEBUG_NUMERICS:
                    print(f"   RMS values [stage {stage_idx}]:")
                    print(f"     input_rms: {input_rms.mean().item():.6f}")
                    print(f"     target_rms: {target_rms.mean().item():.6f}")
                    print(f"     output_rms (before scaling): {output_rms.mean().item():.6f}")
                
                # Fixed equal weighting: target_rms = sqrt((input_rms² + target_rms²) / 2)
                target_rms_combined = torch.sqrt((input_rms ** 2 + target_rms ** 2) / 2.0)  # [B, 1, 1]
                
                # CRITICAL: Check for zero/tiny RMS before division
                if DEBUG_NUMERICS:
                    print(f"     target_rms_combined: {target_rms_combined.mean().item():.6f}")
                    if output_rms.min() < 1e-6:
                        print(f"   ⚠️  WARNING: Very small output_rms detected: {output_rms.min().item():.10f}")
                
                # Scale final output to match target RMS (no learnable scale factor)
                # Add safety check for division by zero
                scale_factor = target_rms_combined / (output_rms + 1e-8)
                
                # CRITICAL FIX: Ultra-tight clamping to prevent gradient explosion
                # Even 2× amplification causes exponential growth over multiple stages
                # Range [0.7, 1.5] allows max 50% amplification while preventing instability
                scale_factor = torch.clamp(scale_factor, min=0.7, max=1.5)
                
                if DEBUG_NUMERICS:
                    print(f"     scale_factor (clamped): min={scale_factor.min().item():.6f}, max={scale_factor.max().item():.6f}, mean={scale_factor.mean().item():.6f}")
                    if scale_factor.max() > 1.3:
                        print(f"   ⚠️  WARNING: Large scale_factor detected (>1.3×)")
                    # Check for NaN/Inf after scaling
                    if torch.isnan(scale_factor).any() or torch.isinf(scale_factor).any():
                        print(f"   🔥 ERROR: NaN/Inf in scale_factor!")
                
                output_projected_transposed = output_projected_transposed * scale_factor
                check_tensor_health(output_projected_transposed, "output_projected_transposed (after RMS scaling)", f"[stage {stage_idx}]")
                
                # Reshape back: [B, D, T_enc] -> [B, T_enc, D] -> [B, D, T_enc]
                output = output_projected_transposed  # Already [B, D, T_enc]
            
            # Return tuple: (output, mask_reg_loss, balance_loss)
            # balance_loss is separate for independent weighting (much higher than mask_reg)
            return output, mask_reg_loss, balance_loss


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    # Test model
    print("Testing SimpleTransformer model...")
    print("="*80)
    
    # Test 1: Single stage
    print("\n1. Testing single stage (num_transformer_layers=1):")
    model1 = SimpleTransformer(
        encoding_dim=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        num_transformer_layers=1
    )
    
    trainable, total = count_parameters(model1)
    print(f"\nParameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    
    # Test forward pass
    batch_size = 16
    T_enc = 1126  # ~16s at 24kHz with EnCodec
    
    x = torch.randn(batch_size, 128, T_enc)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        out1 = model1(x)
    
    print(f"Output shape: {out1.shape}")
    print(f"Expected: [{batch_size}, 128, {T_enc}]")
    assert out1.shape == (batch_size, 128, T_enc), "Single stage output shape mismatch!"
    print("✓ Single stage works!")
    
    # Test 2: Cascade with 3 stages
    print("\n" + "="*80)
    print("2. Testing cascade (num_transformer_layers=3):")
    model3 = SimpleTransformer(
        encoding_dim=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        num_transformer_layers=3
    )
    
    trainable3, total3 = count_parameters(model3)
    print(f"\nParameters:")
    print(f"  Trainable: {trainable3:,}")
    print(f"  Total: {total3:,}")
    print(f"  Increase: {trainable3/trainable:.2f}x vs single stage")
    
    # Test cascade forward pass
    x_input = torch.randn(batch_size, 128, T_enc)
    x_target = torch.randn(batch_size, 128, T_enc)
    print(f"\nInput shape: {x_input.shape}")
    print(f"Target shape: {x_target.shape}")
    
    with torch.no_grad():
        out3 = model3(x_input, x_target)
    
    print(f"Output shape: {out3.shape}")
    print(f"Expected: [{batch_size}, 128, {T_enc}]")
    assert out3.shape == (batch_size, 128, T_enc), "Cascade output shape mismatch!"
    print("✓ Cascade works!")
    
    # Test 3: Verify cascade stages
    print("\n" + "="*80)
    print("3. Testing cascade stage architecture:")
    print(f"  Number of cascade stages: {len(model3.cascade_stages)}")
    print(f"  d_model per stage: {model3.d_model_cascade}")
    for i, stage in enumerate(model3.cascade_stages):
        n_params = sum(p.numel() for p in stage['transformer'].parameters())
        print(f"  Stage {i+1}: {n_params:,} transformer parameters")
    print("  ✓ Cascade stages properly initialized!")
    
    # Test 4: Different inputs produce different outputs
    print("\n" + "="*80)
    print("4. Testing output variation with different targets:")
    x_input = torch.randn(2, 128, T_enc)
    x_target1 = torch.randn(2, 128, T_enc)
    x_target2 = torch.randn(2, 128, T_enc)
    
    with torch.no_grad():
        out_t1 = model3(x_input, x_target1)
        out_t2 = model3(x_input, x_target2)
    
    diff = (out_t1 - out_t2).abs().mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")
    assert diff > 0.01, "Outputs should differ with different targets!"
    print("  ✓ Model responds to target variations!")
    
    # Test 5: Single vs cascade parameter comparison
    print("\n" + "="*80)
    print("5. Architecture comparison:")
    print(f"  Single stage:  {trainable:,} params")
    print(f"  3-stage cascade: {trainable3:,} params ({trainable3/trainable:.2f}x)")
    print(f"  Additional parameters: {trainable3-trainable:,}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80 + "\n")
    print("Test passed!")
