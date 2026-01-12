#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset for WAV Pairs with EnCodec Encoding

Loads WAV file pairs and encodes them with EnCodec on-the-fly
"""

import os
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np


class WavPairsDataset(Dataset):
    """
    Dataset that loads WAV pairs and encodes them with EnCodec
    
    Each sample is a pair:
    - input: pair_XXXX_input.wav
    - output: pair_XXXX_output.wav
    """
    
    def __init__(self, data_folder, encodec_model, device='cuda', shuffle_targets=False):
        """
        Args:
            data_folder: Path to train/ or val/ folder containing pairs
            encodec_model: Pre-loaded EnCodec model (frozen)
            device: Device for encoding
            shuffle_targets: If True, randomly shuffle targets (not matched to inputs)
        """
        self.data_folder = data_folder
        self.encodec_model = encodec_model
        self.device = device
        self.shuffle_targets = shuffle_targets
        
        # Find all input files
        all_files = os.listdir(data_folder)
        self.input_files = sorted([f for f in all_files if f.endswith('_input.wav')])
        
        # Also collect all output files for shuffling
        self.output_files = sorted([f for f in all_files if f.endswith('_output.wav')])
        
        print(f"WavPairsDataset initialized:")
        print(f"  Folder: {data_folder}")
        print(f"  Pairs found: {len(self.input_files)}")
        print(f"  Shuffle targets: {shuffle_targets}")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        """
        Load and encode input, return raw target audio
        
        Returns:
            input_encoded: [D, T_enc] encoded input
            output_audio: [1, samples] raw target audio (NOT encoded)
        """
        # Get file paths
        input_file = self.input_files[idx]
        
        if self.shuffle_targets:
            # Random target: pick any output file
            random_idx = np.random.randint(0, len(self.output_files))
            output_file = self.output_files[random_idx]
        else:
            # Matched pair: continuation of this input
            output_file = input_file.replace('_input.wav', '_output.wav')
        
        input_path = os.path.join(self.data_folder, input_file)
        output_path = os.path.join(self.data_folder, output_file)
        
        # Load audio files (stereo [samples, 2])
        input_audio, sr_input = sf.read(input_path)
        output_audio, sr_output = sf.read(output_path)
        
        # Convert stereo to mono by averaging channels
        if input_audio.ndim == 2:
            input_audio = input_audio.mean(axis=1)  # [samples]
        if output_audio.ndim == 2:
            output_audio = output_audio.mean(axis=1)  # [samples]
        
        # Convert to torch tensor and add channel dimension
        input_audio = torch.from_numpy(input_audio).float()  # [samples]
        output_audio = torch.from_numpy(output_audio).float()  # [samples]
        
        # Add batch and channel dimensions for EnCodec: [1, 1, samples]
        input_audio = input_audio.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # For output, add channel dimension for consistency: [1, samples]
        output_audio = output_audio.unsqueeze(0)  # [1, samples]
        
        # Encode ONLY the input with EnCodec (frozen)
        with torch.no_grad():
            input_encoded = self.encodec_model.encoder(input_audio)  # [1, 128, T]
            input_encoded = input_encoded.squeeze(0)  # [128, T]
        
        # Return encoded input and RAW audio output
        return input_encoded, output_audio


def create_dataloaders(train_folder, val_folder, encodec_model, batch_size, 
                       num_workers=4, device='cuda', shuffle_targets=False):
    """
    Create train and validation dataloaders
    
    Args:
        train_folder: Path to train/ folder
        val_folder: Path to val/ folder
        encodec_model: Pre-loaded EnCodec model
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        device: Device for encoding
        shuffle_targets: If True, randomly shuffle targets (not matched to inputs)
        
    Returns:
        train_loader, val_loader
    """
    print("\nCreating dataloaders...")
    
    train_dataset = WavPairsDataset(train_folder, encodec_model, device, shuffle_targets)
    val_dataset = WavPairsDataset(val_folder, encodec_model, device, shuffle_targets=False)  # Always matched for validation
    
    # Note: num_workers=0 because we're encoding on GPU
    # If using CPU encoding, can increase num_workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # GPU encoding requires 0
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Dataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing WavPairsDataset...")
    print("="*60)
    
    # This is a test - would need actual EnCodec model
    print("Note: This test requires EnCodec model to be loaded")
    print("Run from training script where EnCodec is available")
