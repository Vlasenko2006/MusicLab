# MusicLab

Meet my MusicLab — a powerful end-to-end GAN and GenAI free production-ready music experimentation platform for comercial and private use. Upload your favorite 2 music patterns, choose the parts you like the most and let the MusicLab generate a ginle of those combining them together. 
MusicLab equipped with interactive multilingual Chat-bot assistant “Rita”, and on-prem/cloud model inference
Inside Music Lab (the backend) is a creative agent NN that masks out important music parts, Microsoft EnCodec encoder followeed by cascade of 12-layer transformer blocks (similar to modern LLMs)   - they do the fusion magic, and Microsoft EnCodec decoder.
How to use it on your own? You can use it anytime, just  follow this link and enjoy.

You want to use it locally or deploy on AWS for your own needs? Just download the code and ask me for a check point and be ready to use. The MusicLab docker scripts and and fits on t2.micro iAWS Free Tier instances. See the deployment instructions 


## Overview

Neural audio continuation system that learns to creatively blend input audio (rhythm) with target characteristics (harmony/melody). The model uses learnable complementary masks to separate and intelligently mix different musical components.

## Architecture

```
Input Audio (16s, 24kHz)
    ↓
EnCodec Encoder (frozen)
    ↓
Stage 0 Transformer (256→1024 dim, 6 layers)
    ↓
Stage 1 Transformer (384→1536 dim, 6 layers)
    ↓
Creative Agent (generates complementary masks via attention)
    ├─ Input Mask (extracts rhythm)
    └─ Target Mask (extracts harmony/melody)
    ↓
Masked combination → EnCodec Decoder (frozen)
    ↓
Output Audio
```

**Total Parameters**: 24.9M (16.7M transformer + 700K creative agent + 3.8M discriminator)

### Key Components

- **2-Stage Cascade**: Progressive refinement with spectral normalization
- **Creative Agent**: Cross-attention based mask generator with complementarity loss
- **Audio Discriminator**: Adversarial training for realistic audio generation
- **EnCodec Integration**: 24kHz, 6.0 bandwidth (frozen encoder/decoder)

## Requirements

```bash
# Python 3.10+
torch>=2.0.0
torchaudio>=2.0.0
encodec
mlflow
tqdm
numpy
```

## Quick Start

### Training

```bash
# Single-node multi-GPU training (DDP)
python train_simple_ddp.py \
  --train_dir dataset_pairs_wav/train \
  --val_dir dataset_pairs_wav/val \
  --checkpoint_dir checkpoints \
  --batch_size 8 \
  --epochs 200

# Or use launch script
bash run_train_creative_agent_fixed.sh
```

### Resume from Checkpoint

```bash
bash run_train_creative_agent_resume.sh
```

### Inference

```bash
# Generate audio from trained model
python inference_cascade.py \
  --checkpoint checkpoints/best_model.pt \
  --input_audio input.wav \
  --target_audio target.wav \
  --output output.wav \
  --shuffle_targets  # Random target pairing for creativity
```

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 32 (8×4 GPUs) | Per-device: 8 |
| Learning rate | 1e-4 | AdamW optimizer |
| Weight decay | 0.01 | L2 regularization |
| Epochs | 200 | Full training |
| Input loss weight | 0.3 | RMS reconstruction |
| Target loss weight | 0.3 | RMS continuation |
| Mask reg weight | 0.1 | Complementarity penalty |
| Balance loss weight | 15.0 | 50/50 mask balance |
| GAN weight | 0.15 | Adversarial loss |
| Correlation penalty | 0.5 | Anti-modulation |


## Project Structure

```
PowrMuse/
├── model_simple_transformer.py    # 2-stage cascade architecture
├── creative_agent.py              # Attention-based mask generator
├── audio_discriminator.py         # GAN discriminator
├── correlation_penalty.py         # Anti-modulation loss
│
├── train_simple_worker.py         # DDP training worker
├── train_simple_ddp.py            # Multi-GPU launcher
├── training/losses.py             # Loss functions
│
├── inference_cascade.py           # Audio generation
├── inference_sample.py            # Single sample inference
│
├── dataset_wav_pairs.py           # Dataset loader
├── create_dataset_pairs_wav.py    # Dataset creation
│
├── run_train_creative_agent.sh           # Main training script
├── view_mlflow_data.py            # View training metrics
│
└── README.md                      # This file
```

## Dataset Format

Paired WAV files organized as:

```
dataset_pairs_wav/
├── train/
│   ├── pair_0000_input.wav    # 16 seconds, 24kHz, mono
│   ├── pair_0000_target.wav
│   ├── pair_0001_input.wav
│   └── ...
└── val/
    ├── pair_0000_input.wav
    └── ...
```

Create dataset using:

```bash
python create_dataset_pairs_wav.py \
  --input_dir path/to/audio \
  --output_dir dataset_pairs_wav \
  --duration 16 \
  --sample_rate 24000
```

## Hardware Requirements

- **Training**: 4× NVIDIA A100-80GB (or similar)
- **Inference**: 1× GPU with 16GB+ VRAM (or CPU)
- **Storage**: ~50GB for dataset, ~5GB for checkpoints

## Citation

```bibtex
@software{jingle_d_2025,
  title={Jingle_D: Creative Audio Mixing with Cascade Transformers},
  author={Training conducted on DKRZ Levante supercomputer},
  year={2025},
  url={https://github.com/YOUR_USERNAME/Jingle_D}
}
```

## License

Research project - refer to institution policies for usage terms.
