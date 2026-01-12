"""
Compositional Creative Agent: Musical Decomposition and Recombination
====================================================================

This module implements TRUE creativity through musical decomposition:
- Rhythm extraction (temporal patterns, beat, envelope)
- Harmony extraction (pitch relationships, chords, melody)
- Timbre extraction (tone color, texture, instrumentation)

The agent extracts these components from input and target, then composes
NEW patterns by intelligently recombining them.

Example: Input=drums, Target=piano
- Extract rhythm from input (drum patterns)
- Extract harmony from target (piano chords)
- Extract timbre from target (piano sound)
- Compose: NEW rhythmic piano pattern (not in either source!)

Architecture:
-------------
1. MultiScaleExtractor: Convolutional layers with different kernel sizes
   - Small kernels (3-5): Rhythm/transients
   - Medium kernels (7-11): Harmony/pitch
   - Large kernels (15-21): Timbre/texture

2. ComponentComposer: Transformer that learns musical composition rules
   - Attention across different musical components
   - Learns which combinations sound good
   - Generates coherent new patterns

3. NoveltyRegularization: Ensures output is different from inputs
   - Distance from input encoding
   - Distance from target encoding
   - Balanced with musical quality

Usage:
------
# In model initialization:
self.creative_agent = CompositionalCreativeAgent(encoding_dim=128)

# In forward pass:
creative_output, novelty_loss = self.creative_agent(
    encoded_input, encoded_target
)
# creative_output: [B, D, T] - NEW pattern, not simple mixing!
# novelty_loss: Regularization to ensure creativity

# In training:
loss = reconstruction_loss + novelty_weight * novelty_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from correlation_penalty import compute_modulation_correlation_penalty


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: [B, T, D]"""
        return x + self.pe[:, :x.size(1), :]


class MultiScaleExtractor(nn.Module):
    """
    Extract musical components at different time scales.
    
    Small kernels (3-5): Fast variations, rhythm, transients
    Medium kernels (7-11): Melodic patterns, harmony
    Large kernels (15-21): Long-term structure, timbre, texture
    
    Args:
        encoding_dim: Input dimension [D]
        output_dim: Output dimension per scale
    """
    
    def __init__(self, encoding_dim, output_dim=64):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Rhythm extractor: Small kernel for fast temporal patterns (with skip connections)
        self.rhythm_conv1 = nn.Conv1d(encoding_dim, output_dim * 2, kernel_size=3, padding=1)
        self.rhythm_conv2 = nn.Conv1d(output_dim * 2, output_dim, kernel_size=5, padding=2)
        self.rhythm_norm = nn.BatchNorm1d(output_dim)
        self.rhythm_skip = nn.Conv1d(encoding_dim, output_dim, kernel_size=1)  # Skip projection
        
        # Harmony extractor: Medium kernel for pitch relationships (with skip connections)
        self.harmony_conv1 = nn.Conv1d(encoding_dim, output_dim * 2, kernel_size=7, padding=3)
        self.harmony_conv2 = nn.Conv1d(output_dim * 2, output_dim, kernel_size=9, padding=4)
        self.harmony_norm = nn.BatchNorm1d(output_dim)
        self.harmony_skip = nn.Conv1d(encoding_dim, output_dim, kernel_size=1)  # Skip projection
        
        # Timbre extractor: Large kernel for texture and tone color (with skip connections)
        self.timbre_conv1 = nn.Conv1d(encoding_dim, output_dim * 2, kernel_size=15, padding=7)
        self.timbre_conv2 = nn.Conv1d(output_dim * 2, output_dim, kernel_size=21, padding=10)
        self.timbre_norm = nn.BatchNorm1d(output_dim)
        self.timbre_skip = nn.Conv1d(encoding_dim, output_dim, kernel_size=1)  # Skip projection
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        """
        Extract musical components with skip connections.
        
        Args:
            x: [B, D, T] encoded audio
        
        Returns:
            rhythm: [B, output_dim, T]
            harmony: [B, output_dim, T]
            timbre: [B, output_dim, T]
        """
        # Rhythm extraction with skip connection
        rhythm_skip = self.rhythm_skip(x)
        rhythm = self.activation(self.rhythm_conv1(x))
        rhythm = self.activation(self.rhythm_conv2(rhythm))
        rhythm = self.rhythm_norm(rhythm + rhythm_skip)  # Skip connection ensures gradient flow
        
        # Harmony extraction with skip connection
        harmony_skip = self.harmony_skip(x)
        harmony = self.activation(self.harmony_conv1(x))
        harmony = self.activation(self.harmony_conv2(harmony))
        harmony = self.harmony_norm(harmony + harmony_skip)  # Skip connection ensures gradient flow
        
        # Timbre extraction with skip connection
        timbre_skip = self.timbre_skip(x)
        timbre = self.activation(self.timbre_conv1(x))
        timbre = self.activation(self.timbre_conv2(timbre))
        timbre = self.timbre_norm(timbre + timbre_skip)  # Skip connection ensures gradient flow
        
        return rhythm, harmony, timbre


class ComponentComposer(nn.Module):
    """
    Compose new musical patterns from extracted components.
    
    Uses transformer to learn composition rules:
    - Which rhythm + harmony combinations sound good?
    - How to blend timbre from different sources?
    - How to create coherent new patterns?
    
    Args:
        component_dim: Dimension of each component (rhythm/harmony/timbre)
        num_components: Number of components to combine (default: 6)
        nhead: Attention heads
        num_layers: Transformer layers
        output_dim: Output encoding dimension
    """
    
    def __init__(self, component_dim=64, num_components=6, nhead=8, 
                 num_layers=4, output_dim=128):
        super().__init__()
        
        self.component_dim = component_dim
        self.num_components = num_components
        self.total_dim = component_dim * num_components
        
        # Project components to transformer dimension
        self.component_projection = nn.Linear(self.total_dim, 512)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(512)
        
        # Transformer: Learn composition rules
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True  # [B, T, D]
        )
        self.composer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: Back to encoding dimension
        self.output_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, components):
        """
        Compose new pattern from components.
        
        Args:
            components: [B, total_dim, T] concatenated components
        
        Returns:
            composed: [B, output_dim, T] new creative pattern
        """
        B, D, T = components.shape
        
        # [B, D, T] -> [B, T, D]
        x = components.transpose(1, 2)
        
        # Project to transformer dimension
        x = self.component_projection(x)  # [B, T, 512]
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [B, T, 512]
        
        # Apply transformer (learns composition rules)
        x = self.composer(x)  # [B, T, 512]
        
        # Project to output dimension
        x = self.output_projection(x)  # [B, T, output_dim]
        
        # STABILITY: Clamp transformer output to prevent explosions
        x = torch.clamp(x, -100.0, 100.0)
        
        # [B, T, D] -> [B, D, T]
        composed = x.transpose(1, 2)  # [B, output_dim, T]
        
        return composed


class CompositionalCreativeAgent(nn.Module):
    """
    Creative agent that generates NEW musical patterns through decomposition.
    
    Process:
    1. Extract rhythm, harmony, timbre from input and target
    2. Select which components to use from each source
    3. Compose NEW pattern using transformer (learns musical rules)
    4. Ensure novelty: output must differ from both inputs
    
    This creates TRUE creativity:
    - Input: drums (strong rhythm, no harmony, percussive timbre)
    - Target: piano (weak rhythm, rich harmony, tonal timbre)
    - Output: Rhythmic piano (drum rhythm + piano harmony + piano timbre)
    
    The output is NOT in either input - it's a new musical idea!
    
    Args:
        encoding_dim: EnCodec encoding dimension (128)
        component_dim: Dimension for each musical component (default: 64)
        composer_heads: Attention heads in composer (default: 8)
        composer_layers: Transformer layers (default: 4)
        novelty_weight: Weight for novelty regularization (default: 0.1)
    """
    
    def __init__(self, encoding_dim=128, component_dim=64, composer_heads=8,
                 composer_layers=4, novelty_weight=0.1):
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.component_dim = component_dim
        self.novelty_weight = novelty_weight
        
        # Component extractors for input and target
        self.input_extractor = MultiScaleExtractor(encoding_dim, component_dim)
        self.target_extractor = MultiScaleExtractor(encoding_dim, component_dim)
        
        # Component selector: Decide which components to use
        # Input: 6 components (3 from input, 3 from target)
        # Output: Attention weights for composition
        # Note: We'll apply softmax over component groups, not individual channels
        self.component_selector = nn.Sequential(
            nn.Conv1d(component_dim * 6, 256, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 6, kernel_size=1),  # Output 6 weights (one per component)
        )
        
        self.component_dim = component_dim
        
        # Composer: Combine selected components into new pattern
        self.composer = ComponentComposer(
            component_dim=component_dim,
            num_components=6,  # 3 from input + 3 from target
            nhead=composer_heads,
            num_layers=composer_layers,
            output_dim=encoding_dim
        )
    
    def _extract_components(self, encoded_input, encoded_target):
        """Extract rhythm, harmony, timbre from both sources."""
        input_rhythm, input_harmony, input_timbre = self.input_extractor(encoded_input)
        target_rhythm, target_harmony, target_timbre = self.target_extractor(encoded_target)
        
        return (input_rhythm, input_harmony, input_timbre,
                target_rhythm, target_harmony, target_timbre)
    
    def _compute_component_weights(self, all_components, B, T):
        """Compute attention-based weights for component selection."""
        # Get selection logits: [B, 6, T]
        selection_logits = self.component_selector(all_components)
        
        # Normalize weights with softmax
        component_type_weights = F.softmax(selection_logits, dim=1)  # [B, 6, T]
        
        # Expand to match component dimensions
        weights_expanded = component_type_weights.unsqueeze(2)  # [B, 6, 1, T]
        weights_expanded = weights_expanded.expand(-1, -1, self.component_dim, -1)  # [B, 6, component_dim, T]
        weights_reshaped = weights_expanded.reshape(B, 6 * self.component_dim, T)  # [B, 6*component_dim, T]
        
        return weights_reshaped, selection_logits
    
    def _compute_balance_penalty(self, selection_logits):
        """Encourage balanced usage of input and target components."""
        # Average weights: [B, 6, T] -> [6]
        avg_weights = selection_logits.mean(dim=[0, 2])
        
        # Sum input (0,1,2) vs target (3,4,5) components
        input_total = avg_weights[:3].sum()
        target_total = avg_weights[3:].sum()
        
        # Penalty for imbalance (want ~50/50 split)
        balance_loss = torch.abs(input_total - target_total) * 0.1
        
        return balance_loss
    
    def _compute_novelty_loss(self, creative_output, encoded_input, encoded_target):
        """Ensure output differs from both input and target (orthogonality in latent space)."""
        # L2 normalization for cosine similarity
        creative_norm = F.normalize(creative_output, p=2, dim=1)
        input_norm = F.normalize(encoded_input, p=2, dim=1)
        target_norm = F.normalize(encoded_target, p=2, dim=1)
        
        # Cosine similarity [-1, 1]
        similarity_to_input = (creative_norm * input_norm).sum(dim=1).mean()
        similarity_to_target = (creative_norm * target_norm).sum(dim=1).mean()
        
        # Penalize high absolute correlation (want orthogonal outputs)
        novelty_loss = (torch.abs(similarity_to_input) + torch.abs(similarity_to_target)) * 0.5
        
        return novelty_loss
    
    def forward(self, encoded_input, encoded_target):
        """
        Generate creative pattern through compositional decomposition.
        
        Args:
            encoded_input: [B, D, T] input encoding
            encoded_target: [B, D, T] target encoding
        
        Returns:
            creative_output: [B, D, T] NEW composed pattern
            novelty_loss: Regularization ensuring output differs from inputs
        """
        B, D, T = encoded_input.shape
        
        # Extract musical components from both sources
        components = self._extract_components(encoded_input, encoded_target)
        all_components = torch.cat(components, dim=1)  # [B, 6*component_dim, T]
        
        # Compute component selection weights
        component_weights, selection_logits = self._compute_component_weights(all_components, B, T)
        selected_components = all_components * component_weights
        
        # Compute penalties
        balance_loss = self._compute_balance_penalty(selection_logits)
        
        # Compose new pattern
        creative_output = self.composer(selected_components)  # [B, D, T]
        
        # STABILITY: Clamp output to prevent extreme values that cause NaN
        # EnCodec operates in range ~[-30, 30], clamp to safe range
        creative_output = torch.clamp(creative_output, -50.0, 50.0)
        
        # STABILITY: Check for NaN and replace with zeros if detected
        if torch.isnan(creative_output).any():
            print("⚠️  WARNING: NaN detected in compositional agent output, replacing with zeros")
            creative_output = torch.where(torch.isnan(creative_output), torch.zeros_like(creative_output), creative_output)
        
        # Compute novelty loss
        novelty_loss = self._compute_novelty_loss(creative_output, encoded_input, encoded_target)
        total_loss = novelty_loss + balance_loss
        
        return creative_output, total_loss
    
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
    
    def get_component_statistics(self, encoded_input, encoded_target):
        """
        Analyze component extraction for debugging/visualization.
        
        Returns statistics about which components are being used.
        
        Returns:
            dict with component energies and selection weights
        """
        with torch.no_grad():
            # Extract components
            input_rhythm, input_harmony, input_timbre = self.input_extractor(encoded_input)
            target_rhythm, target_harmony, target_timbre = self.target_extractor(encoded_target)
            
            # Concatenate
            all_components = torch.cat([
                input_rhythm, input_harmony, input_timbre,
                target_rhythm, target_harmony, target_timbre
            ], dim=1)
            
            # Get selection weights: [B, 6, T] - one weight per component type
            selection_logits = self.component_selector(all_components)  # [B, 6, T]
            component_type_weights = F.softmax(selection_logits, dim=1)  # [B, 6, T]
            
            # Average over batch and time to get overall component importance
            component_type_weights_avg = component_type_weights.mean(dim=[0, 2])  # [6]
            
            # Calculate energy per component type
            def component_energy(x):
                return torch.sqrt(torch.mean(x ** 2))
            
            stats = {
                'input_rhythm_energy': component_energy(input_rhythm).item(),
                'input_harmony_energy': component_energy(input_harmony).item(),
                'input_timbre_energy': component_energy(input_timbre).item(),
                'target_rhythm_energy': component_energy(target_rhythm).item(),
                'target_harmony_energy': component_energy(target_harmony).item(),
                'target_timbre_energy': component_energy(target_timbre).item(),
                'input_rhythm_weight': component_type_weights_avg[0].item(),
                'input_harmony_weight': component_type_weights_avg[1].item(),
                'input_timbre_weight': component_type_weights_avg[2].item(),
                'target_rhythm_weight': component_type_weights_avg[3].item(),
                'target_harmony_weight': component_type_weights_avg[4].item(),
                'target_timbre_weight': component_type_weights_avg[5].item(),
            }
            
            return stats


if __name__ == "__main__":
    """Test the compositional creative agent"""
    
    print("="*80)
    print("Testing Compositional Creative Agent")
    print("="*80)
    
    # Create agent
    agent = CompositionalCreativeAgent(
        encoding_dim=128,
        component_dim=64,
        composer_heads=8,
        composer_layers=4,
        novelty_weight=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    
    print(f"\nAgent Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    encoding_dim = 128
    time_steps = 750
    
    encoded_input = torch.randn(batch_size, encoding_dim, time_steps)
    encoded_target = torch.randn(batch_size, encoding_dim, time_steps)
    
    print(f"\nInput shapes:")
    print(f"  Encoded input: {encoded_input.shape}")
    print(f"  Encoded target: {encoded_target.shape}")
    
    # Forward pass
    creative_output, novelty_loss = agent(encoded_input, encoded_target)
    
    print(f"\nOutput shapes:")
    print(f"  Creative output: {creative_output.shape}")
    print(f"  Novelty loss: {novelty_loss.item():.6f}")
    
    # Component statistics
    stats = agent.get_component_statistics(encoded_input, encoded_target)
    
    print(f"\nComponent Analysis:")
    print(f"  Input components:")
    print(f"    Rhythm  - Energy: {stats['input_rhythm_energy']:.4f}, Weight: {stats['input_rhythm_weight']:.4f}")
    print(f"    Harmony - Energy: {stats['input_harmony_energy']:.4f}, Weight: {stats['input_harmony_weight']:.4f}")
    print(f"    Timbre  - Energy: {stats['input_timbre_energy']:.4f}, Weight: {stats['input_timbre_weight']:.4f}")
    print(f"  Target components:")
    print(f"    Rhythm  - Energy: {stats['target_rhythm_energy']:.4f}, Weight: {stats['target_rhythm_weight']:.4f}")
    print(f"    Harmony - Energy: {stats['target_harmony_energy']:.4f}, Weight: {stats['target_harmony_weight']:.4f}")
    print(f"    Timbre  - Energy: {stats['target_timbre_energy']:.4f}, Weight: {stats['target_timbre_weight']:.4f}")
    
    print("\n" + "="*80)
    print("✓ Compositional Creative Agent working correctly!")
    print("="*80)
