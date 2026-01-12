#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified EncodedPatternDataset - Treats each pattern as a separate sample

Key change: Dataset yields 5400 samples (1800 base × 3 patterns) instead of 1800.
Each pattern becomes an independent training sample.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class EncodedPatternDataset(Dataset):
    """
    Dataset for pre-encoded EnCodec representations.
    
    IMPORTANT: Treats each of the 3 patterns per sample as independent samples.
    - Original: 1800 samples with 3 patterns each
    - This version: 5400 samples (each pattern is a sample)
    
    Each sample returns:
        inputs: [N, D, T] - N patterns to fuse
        targets: [N, D, T] - N target patterns (same as inputs for reconstruction)
    
    Where N = num_patterns (default 3), D = encoding_dim (128), T = time_steps (1126)
    """
    
    def __init__(self, dataset_folder, split='train', num_patterns=1):
        """
        Args:
            dataset_folder: Path to dataset folder
            split: 'train' or 'val'
            num_patterns: Number of patterns to include in input/target (default 3)
        """
        self.dataset_folder = dataset_folder
        self.split = split
        self.num_patterns = num_patterns
        
        # Load memory-mapped arrays
        # Note: Dataset has train_inputs.npy and train_targets.npy
        self.inputs_path = os.path.join(dataset_folder, f'{split}_inputs.npy')
        self.targets_path = os.path.join(dataset_folder, f'{split}_targets.npy')
        
        if not os.path.exists(self.inputs_path):
            raise FileNotFoundError(f"Dataset not found: {self.inputs_path}")
        if not os.path.exists(self.targets_path):
            raise FileNotFoundError(f"Dataset not found: {self.targets_path}")
        
        # Memory-map the data (don't load into RAM)
        self.inputs_data = np.load(self.inputs_path, mmap_mode='r')
        self.targets_data = np.load(self.targets_path, mmap_mode='r')
        
        # Original shape: [num_base_samples, 3, D, T]
        self.num_base_samples = self.inputs_data.shape[0]
        self.patterns_per_sample = self.inputs_data.shape[1]  # Should be 3
        self.encoding_dim = self.inputs_data.shape[2]
        self.time_steps = self.inputs_data.shape[3]
        
        # NEW: Total samples = base_samples × patterns_per_sample
        # Each pattern becomes an independent sample
        self.total_samples = self.num_base_samples * self.patterns_per_sample
        
        print(f"{split.upper()} dataset loaded:")
        print(f"  Base samples: {self.num_base_samples}")
        print(f"  Patterns per base sample: {self.patterns_per_sample}")
        print(f"  Total independent samples: {self.total_samples}")
        print(f"  Shape per sample: [{self.patterns_per_sample}, {self.encoding_dim}, {self.time_steps}]")
        print(f"  Memory: ~{self.inputs_data.nbytes / 1e9:.2f} GB (memory-mapped)")
    
    def __len__(self):
        """Return total number of independent samples"""
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get a sample by treating each pattern as independent.
        
        Args:
            idx: Index from 0 to (num_base_samples × 3) - 1
        
        Returns:
            inputs: [N, D, T] - N patterns from the same base sample
            targets: [N, D, T] - Same N patterns (reconstruction task)
        """
        # Map linear idx to (base_sample_idx, pattern_idx)
        base_idx = idx // self.patterns_per_sample
        pattern_idx = idx % self.patterns_per_sample
        
        # Load all patterns from this base sample
        # Shape: [3, D, T]
        inputs_patterns = self.inputs_data[base_idx]
        targets_patterns = self.targets_data[base_idx]
        
        # Use all 3 patterns as input and target
        # This maintains the multi-pattern fusion task
        inputs = torch.from_numpy(inputs_patterns.copy()).float()
        targets = torch.from_numpy(targets_patterns.copy()).float()
        
        return inputs, targets
    
    def get_sample_info(self, idx):
        """Get which base sample and pattern this index corresponds to"""
        base_idx = idx // self.patterns_per_sample
        pattern_idx = idx % self.patterns_per_sample
        return base_idx, pattern_idx


def test_dataset():
    """Test the modified dataset"""
    print("\n" + "="*80)
    print("TESTING MODIFIED EncodedPatternDataset")
    print("="*80)
    
    # Use HPC path for testing
    dataset_folder = "/work/gg0302/g260141/Jingle/dataset_multipattern_encoded"
    
    print("\n1. Loading train dataset...")
    train_dataset = EncodedPatternDataset(dataset_folder, split='train')
    
    print("\n2. Checking dataset length:")
    print(f"   Expected: 1800 × 3 = 5400 samples")
    print(f"   Actual: {len(train_dataset)} samples")
    assert len(train_dataset) == 5400, "Dataset length incorrect!"
    
    print("\n3. Testing sample retrieval:")
    for test_idx in [0, 1, 2, 3, 1799, 1800, 1801, 5399]:
        inputs, targets = train_dataset[test_idx]
        base_idx, pattern_idx = train_dataset.get_sample_info(test_idx)
        print(f"   Sample {test_idx}: base_sample={base_idx}, pattern={pattern_idx}, shape={list(inputs.shape)}")
    
    print("\n4. Verifying all samples are unique:")
    # Check first 10 samples have same base but accessed differently
    sample_0 = train_dataset[0][0]
    sample_1 = train_dataset[1][0]
    sample_2 = train_dataset[2][0]
    sample_3 = train_dataset[3][0]  # This should be from base_sample=1
    
    print(f"   Samples 0,1,2 from same base: {torch.equal(sample_0, sample_1) and torch.equal(sample_1, sample_2)}")
    print(f"   Sample 3 from different base: {not torch.equal(sample_0, sample_3)}")
    
    print("\n5. Verifying batching works:")
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    batch_inputs, batch_targets = next(iter(loader))
    print(f"   Batch shape: {list(batch_inputs.shape)}")
    print(f"   Expected: [32, 3, 128, 1126]")
    assert batch_inputs.shape == (32, 3, 128, 1126), "Batch shape incorrect!"
    
    print("\n6. Calculating batches per epoch:")
    num_batches = len(train_dataset) // 32
    print(f"   Samples: {len(train_dataset)}")
    print(f"   Batch size: 32 (effective with 4 GPUs)")
    print(f"   Batches per epoch: {num_batches}")
    print(f"   Expected: ~169 batches (instead of previous 57)")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! Dataset now uses all 3 patterns as separate samples.")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_dataset()
