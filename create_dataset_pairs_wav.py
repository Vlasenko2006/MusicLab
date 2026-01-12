#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Simple Pairs Dataset for Audio Continuation Task

Creates pairs of consecutive audio segments:
- Input: song[0-16sec]
- Output: song[16-32sec]

OR

- Input: song[16-32sec]  
- Output: song[32-48sec]

No encoding, just raw WAV format. No triplets, just pairs.
"""

import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random

def create_pairs_dataset(
    input_folder="output",
    output_folder="dataset_pairs_wav",
    segment_duration=16.0,  # Duration in seconds
    target_sr=24000,
    train_split=0.9,
    max_pairs_per_song=10,
    max_total_pairs=2000
):
    """
    Create dataset of audio segment pairs in WAV format.
    
    Args:
        input_folder: Folder with .npy audio files
        output_folder: Where to save the dataset
        segment_duration: Length of each segment in seconds
        target_sr: Target sample rate
        train_split: Fraction for training (rest is validation)
        max_pairs_per_song: Maximum pairs to extract from each song
        max_total_pairs: Maximum total pairs in dataset
    """
    
    print("\n" + "="*80)
    print("CREATING AUDIO PAIRS DATASET (WAV FORMAT)")
    print("="*80)
    
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Configuration
    samples_per_segment = int(segment_duration * target_sr)
    print(f"\nConfiguration:")
    print(f"  Segment duration: {segment_duration} seconds")
    print(f"  Sample rate: {target_sr} Hz")
    print(f"  Samples per segment: {samples_per_segment}")
    print(f"  Train/Val split: {train_split:.1%} / {1-train_split:.1%}")
    print(f"  Max pairs per song: {max_pairs_per_song}")
    print(f"  Max total pairs: {max_total_pairs}")
    
    # Find all .npy files
    npy_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
    print(f"\nFound {len(npy_files)} .npy files")
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {input_folder}")
    
    # Phase 1: Scan files and find valid pairs
    print("\nPhase 1: Scanning audio files for valid pairs...")
    all_pairs = []
    
    for npy_file in tqdm(npy_files, desc="Scanning"):
        file_path = os.path.join(input_folder, npy_file)
        
        try:
            # Load audio (memory-mapped for efficiency)
            audio = np.load(file_path, mmap_mode='r')
            
            # Check format: should be [2, samples] (stereo)
            if audio.ndim != 2 or audio.shape[0] != 2:
                print(f"  Skipping {npy_file}: wrong shape {audio.shape}")
                continue
            
            total_samples = audio.shape[1]
            
            # Need at least 2 consecutive segments
            if total_samples < 2 * samples_per_segment:
                print(f"  Skipping {npy_file}: too short ({total_samples} samples)")
                continue
            
            # Find all possible consecutive pairs
            num_possible_pairs = total_samples // samples_per_segment - 1
            num_pairs_to_use = min(num_possible_pairs, max_pairs_per_song)
            
            # Extract pair info (don't load audio yet)
            for pair_idx in range(num_pairs_to_use):
                start_sample = pair_idx * samples_per_segment
                all_pairs.append({
                    'file': npy_file,
                    'start_sample': start_sample,
                    'pair_idx': pair_idx
                })
                
        except Exception as e:
            print(f"  Error with {npy_file}: {e}")
            continue
    
    print(f"\nFound {len(all_pairs)} valid pairs")
    
    # Limit total pairs
    if len(all_pairs) > max_total_pairs:
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_total_pairs]
        print(f"Limited to {max_total_pairs} pairs")
    
    # Phase 2: Split into train/val
    random.shuffle(all_pairs)
    num_train = int(len(all_pairs) * train_split)
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val: {len(val_pairs)} pairs")
    
    # Phase 3: Extract and save pairs
    print("\nPhase 3: Extracting and saving audio pairs...")
    
    def save_pairs(pairs, folder, split_name):
        """Extract and save pairs to WAV files"""
        for idx, pair_info in enumerate(tqdm(pairs, desc=f"Saving {split_name}")):
            file_path = os.path.join(input_folder, pair_info['file'])
            start_sample = pair_info['start_sample']
            
            try:
                # Load full audio
                audio = np.load(file_path, mmap_mode='r')
                
                # Extract input segment (first segment)
                input_start = start_sample
                input_end = start_sample + samples_per_segment
                input_segment = audio[:, input_start:input_end].copy()
                
                # Extract output segment (next segment)
                output_start = input_end
                output_end = output_start + samples_per_segment
                output_segment = audio[:, output_start:output_end].copy()
                
                # Verify shapes
                if input_segment.shape[1] != samples_per_segment:
                    continue
                if output_segment.shape[1] != samples_per_segment:
                    continue
                
                # Save as WAV files
                # Transpose to [samples, channels] for soundfile
                input_path = os.path.join(folder, f'pair_{idx:04d}_input.wav')
                output_path = os.path.join(folder, f'pair_{idx:04d}_output.wav')
                
                sf.write(input_path, input_segment.T, target_sr)
                sf.write(output_path, output_segment.T, target_sr)
                
            except Exception as e:
                print(f"\n  Error saving pair {idx}: {e}")
                continue
    
    # Save train and val sets
    save_pairs(train_pairs, train_folder, 'train')
    save_pairs(val_pairs, val_folder, 'val')
    
    # Create metadata file
    metadata = {
        'segment_duration': segment_duration,
        'sample_rate': target_sr,
        'samples_per_segment': samples_per_segment,
        'num_train': len(train_pairs),
        'num_val': len(val_pairs),
        'total_pairs': len(all_pairs)
    }
    
    metadata_path = os.path.join(output_folder, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE")
    print("="*80)
    print(f"\nDataset saved to: {output_folder}/")
    print(f"  Train: {len(train_pairs)} pairs in {train_folder}/")
    print(f"  Val: {len(val_pairs)} pairs in {val_folder}/")
    print(f"\nEach pair consists of:")
    print(f"  - pair_XXXX_input.wav: segment [0-16sec]")
    print(f"  - pair_XXXX_output.wav: segment [16-32sec]")
    print(f"\nAll files are stereo WAV at {target_sr} Hz")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Default configuration
    create_pairs_dataset(
        input_folder="output",
        output_folder="dataset_pairs_wav",
        segment_duration=16.0,
        target_sr=24000,
        train_split=0.9,
        max_pairs_per_song=10,
        max_total_pairs=2000
    )
