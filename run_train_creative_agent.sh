#!/bin/bash

# Training script with FIXED loss configuration to prevent target copying
# Changes from original:
# 1. loss_weight_input: 0.0 → 0.3 (encourage using input rhythm)
# 2. loss_weight_target: 1.0 → 0.7 (reduce target copying bias)
# 3. balance_loss_weight: 5.0 → 10.0 (stronger mask balance)
# 4. novelty_weight: 0.85 → 0.5 (moderate creativity, was too high)

set -e

echo "=================================================="
echo "Training Creative Agent with FIXED Loss Weights"
echo "=================================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "=================================================="
echo ""
echo "🔧 CHANGES FROM ORIGINAL:"
echo "  - loss_weight_input: 0.0 (no change - using GAN instead)"
echo "  - loss_weight_target: 1.0 → 0.0 (removed - using GAN instead)"
echo "  - balance_loss_weight: 5.0 → 10.0 (stronger 50/50 enforcement)"
echo "  - novelty_weight: 0.85 → 0.5 (moderate creativity)"
echo "  - GAN weight: 0.0 → 0.1 (adversarial training for quality)"
echo ""
echo "📊 NEW DIAGNOSTIC METRICS:"
echo "  - Output→Input correlation (should be 0.4-0.6)"
echo "  - Output→Target correlation (should be 0.5-0.7)"
echo "  - Alerts if output copying one source"
echo ""
echo "=================================================="
echo ""

# Environment
PYTHON_BIN=your/python/path
WORKING_DIR=your/working/directory/path

# Set DDP port (use random port to avoid conflicts)
export MASTER_PORT=$((29500 + RANDOM % 500))

# NCCL configuration to prevent timeouts and improve stability
export NCCL_TIMEOUT=1800  # 30 minutes (was 10 minutes)
export NCCL_DEBUG=WARN    # Only show warnings/errors (was INFO)
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Detect errors earlier (updated from deprecated NCCL_ASYNC_ERROR_HANDLING)

echo "Environment:"
echo "  Python: ${PYTHON_BIN}"
echo "  Working directory: ${WORKING_DIR}"
echo "  CUDA devices: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "  DDP Master Port: ${MASTER_PORT}"
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Configuration
DATASET_ROOT=dataset_pairs_wav
CHECKPOINT_DIR=checkpoints_creative_agent
ENCODING_DIM=128
ATTENTION_HEADS=8
TRANSFORMER_LAYERS=6
CASCADE_STAGES=3
DROPOUT=0.1
ENCODEC_BW=6.0
ENCODEC_SR=24000
EPOCHS=200
BATCH_SIZE=8
LR=1e-4
WEIGHT_DECAY=0.01
NUM_WORKERS=0
PATIENCE=20
SEED=42
SAVE_EVERY=10

# Creative Agent Configuration
USE_CREATIVE_AGENT=true
MASK_REG_WEIGHT=0.1
BALANCE_LOSS_WEIGHT=15.0  # INCREASED to 15.0 (strong 50/50 enforcement)

# Loss Configuration (FIXED v2 - small reconstruction guidance)
LOSS_WEIGHT_INPUT=0.3      # Small guidance from input
LOSS_WEIGHT_TARGET=0.3     # Small guidance from target
LOSS_WEIGHT_SPECTRAL=0.0
LOSS_WEIGHT_MEL=0.0

# GAN Configuration
GAN_WEIGHT=0.1
DISC_LR=5e-5
DISC_UPDATE_FREQ=1

# Correlation (anti-modulation)
CORR_WEIGHT=0.5  # Quadratic penalty (corr²) is stable - can use higher weight

# Gradient clipping (increased to handle RMS scaling spikes)
GRAD_CLIP=10.0  # Increased from 5.0 to handle stage 2 RMS amplification

# Other
UNITY_TEST=false
SHUFFLE_TARGETS=true   # CHANGED: Random pairing forces true mixing
ANTI_CHEATING=0.1
MASK_TYPE=none

# World size (number of GPUs)
WORLD_SIZE=4

echo "=================================================="
echo "Starting DDP training with Creative Agent (FIXED)..."
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - Model: SimpleTransformer with Creative Agent"
echo "  - Encoding dim: ${ENCODING_DIM}"
echo "  - Attention heads: ${ATTENTION_HEADS}"
echo "  - Transformer layers: ${TRANSFORMER_LAYERS}"
echo "  - Transformer cascade stages: ${CASCADE_STAGES}"
echo "  - Dropout: ${DROPOUT}"
echo ""
echo "  - Creative Agent: ENABLED 🎨"
echo "    * Learnable attention-based masking"
echo "    * Mask generator: ~500K params"
echo "    * Discriminator: ~200K params"
echo "    * Mask regularization weight: ${MASK_REG_WEIGHT}"
echo "    * Balance loss weight: ${BALANCE_LOSS_WEIGHT} (INCREASED from 5.0)"
echo ""
echo "  - EnCodec: ${ENCODEC_SR}Hz, bandwidth=${ENCODEC_BW} (FROZEN)"
echo ""
echo "  - Dataset: ${DATASET_ROOT}/"
echo "  - Batch size: ${BATCH_SIZE} per GPU × ${WORLD_SIZE} GPUs = $((BATCH_SIZE * WORLD_SIZE))"
echo "  - Learning rate: ${LR}"
echo "  - Optimizer: AdamW (weight_decay=${WEIGHT_DECAY})"
echo "  - Loss: Combined perceptual + mask regularization"
echo "  - Loss weights (FIXED):"
echo "    * input=${LOSS_WEIGHT_INPUT} (no reconstruction loss)"
echo "    * target=${LOSS_WEIGHT_TARGET} (no reconstruction loss)"
echo "    * spectral=${LOSS_WEIGHT_SPECTRAL}"
echo "    * mel=${LOSS_WEIGHT_MEL}"
echo "    * GAN=${GAN_WEIGHT} (adversarial training)"
echo "  - Novelty weight: ${NOVELTY_WEIGHT} (MODERATE: was 0.85)"
echo "  - Correlation weight: ${CORR_WEIGHT}"
echo "  - Gradient clipping: ${GRAD_CLIP} (max norm)"
echo "  - Shuffle targets: ${SHUFFLE_TARGETS} (continuation pairs)"
echo "  - Anti-cheating noise: ${ANTI_CHEATING} (stages 2+)"
echo "  - Fixed masking: DISABLED (creative agent replaces it)"
echo "  - GAN Training: ENABLED"
echo "    * Discriminator LR: ${DISC_LR}"
echo "    * Update frequency: ${DISC_UPDATE_FREQ} batch"
echo "  - Epochs: ${EPOCHS} (with early stopping)"
echo ""
echo "Expected Behavior (with FIXED losses):"
echo "  - Complementarity: ~75% (untrained) → 85-95% (after 50+ epochs)"
echo "  - Mask reg loss: ~0.25 → 0.05-0.10"
echo "  - Output→Input correlation: 0.2 → 0.4-0.6 (using input rhythm)"
echo "  - Output→Target correlation: 0.8 → 0.5-0.7 (balanced)"
echo "  - Alert: ⚠️ if output copying one source (ratio > 2×)"
echo ""
echo "=================================================="
echo ""

# Check for SLURM job ID
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Detected SLURM job: $SLURM_JOB_ID"
    # GPUs already allocated by SLURM
    echo "✓ GPUs already allocated"
else
    echo "Not running under SLURM"
    echo "Assuming GPUs are available via CUDA_VISIBLE_DEVICES"
fi

echo ""
echo "================================================================================";
echo "SIMPLE TRANSFORMER TRAINING ON WAV PAIRS (FIXED LOSS CONFIGURATION)";
echo "================================================================================";
echo "Configuration:"
echo "  Dataset: ${DATASET_ROOT}"
echo "  Model:"
echo "    - Encoding dim: ${ENCODING_DIM}"
echo "    - Attention heads: ${ATTENTION_HEADS}"
echo "    - Internal transformer layers: ${TRANSFORMER_LAYERS}"
echo "    - Cascade stages: ${CASCADE_STAGES}"
echo "    - Dropout: ${DROPOUT}"
echo "  EnCodec:"
echo "    - Sample rate: ${ENCODEC_SR} Hz"
echo "    - Bandwidth: ${ENCODEC_BW}"
echo "    - Status: FROZEN (encoder + decoder)"
echo "  Training:"
echo "    - Epochs: ${EPOCHS}"
echo "    - Batch size: ${BATCH_SIZE} per GPU × ${WORLD_SIZE} GPUs = $((BATCH_SIZE * WORLD_SIZE * 2))"
echo "    - Learning rate: ${LR}"
echo "    - Optimizer: AdamW (weight_decay=${WEIGHT_DECAY})"
echo "    - Loss: Combined perceptual loss (FIXED)"
echo "      * Input weight: ${LOSS_WEIGHT_INPUT} (no reconstruction)"
echo "      * Target weight: ${LOSS_WEIGHT_TARGET} (no reconstruction)"
echo "      * Spectral weight: ${LOSS_WEIGHT_SPECTRAL}"
echo "      * Mel weight: ${LOSS_WEIGHT_MEL}"
echo "      * GAN weight: ${GAN_WEIGHT} (adversarial training)"
echo "    - Unity test: ${UNITY_TEST}"
echo "    - Shuffle targets: ${SHUFFLE_TARGETS}"
echo "  🎨 Attention-Based Creative Agent:"
echo "    - Enabled: ${USE_CREATIVE_AGENT} (learnable masking)"
echo "    - Mask regularization weight: ${MASK_REG_WEIGHT}"
echo "    - Balance loss weight: ${BALANCE_LOSS_WEIGHT} (INCREASED)"
echo "  GAN Training:"
echo "    - Enabled: True (adversarial training)"
echo "    - Discriminator LR: ${DISC_LR}"
echo "    - Discriminator update frequency: every ${DISC_UPDATE_FREQ} batch(es)"
echo "  Checkpoints: ${CHECKPOINT_DIR}/"
echo "================================================================================";
echo "================================================================================";

# Launch training (train_simple_ddp.py internally spawns DDP workers)
${PYTHON_BIN} train_simple_ddp.py \
    --dataset_folder ${DATASET_ROOT} \
    --encoding_dim ${ENCODING_DIM} \
    --nhead ${ATTENTION_HEADS} \
    --num_layers ${TRANSFORMER_LAYERS} \
    --num_transformer_layers ${CASCADE_STAGES} \
    --dropout ${DROPOUT} \
    --encodec_bandwidth ${ENCODEC_BW} \
    --encodec_sr ${ENCODEC_SR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_workers ${NUM_WORKERS} \
    --patience ${PATIENCE} \
    --seed ${SEED} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --save_every ${SAVE_EVERY} \
    --world_size ${WORLD_SIZE} \
    --unity_test ${UNITY_TEST} \
    --shuffle_targets ${SHUFFLE_TARGETS} \
    --anti_cheating ${ANTI_CHEATING} \
    --loss_weight_input ${LOSS_WEIGHT_INPUT} \
    --loss_weight_target ${LOSS_WEIGHT_TARGET} \
    --loss_weight_spectral ${LOSS_WEIGHT_SPECTRAL} \
    --loss_weight_mel ${LOSS_WEIGHT_MEL} \
    --mask_type ${MASK_TYPE} \
    --use_creative_agent ${USE_CREATIVE_AGENT} \
    --mask_reg_weight ${MASK_REG_WEIGHT} \
    --balance_loss_weight ${BALANCE_LOSS_WEIGHT} \
    --gan_weight ${GAN_WEIGHT} \
    --disc_lr ${DISC_LR} \
    --disc_update_freq ${DISC_UPDATE_FREQ} \
    --corr_weight ${CORR_WEIGHT}

echo ""
echo "=================================================="
echo "Next Steps:"
echo "  1. Monitor Output→Input correlation (should increase to 0.4-0.6)"
echo "  2. Monitor Output→Target correlation (should decrease to 0.5-0.7)"
echo "  3. Watch for alerts: ⚠️ OUTPUT IS COPYING TARGET/INPUT"
echo "  4. After 20 epochs, check if mixing is balanced"
echo "  5. Run inference: python inference_cascade.py --checkpoint ${CHECKPOINT_DIR}/best_model.pt"
echo "=================================================="
echo ""
echo "Training finished"
echo "=================================================="
