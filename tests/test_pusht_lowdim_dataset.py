#!/usr/bin/env python3

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import numpy as np
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset

def test_pusht_lowdim_dataset():
    """Test PushTLowdimDataset loading and data retrieval"""
    
    # Try to find a suitable dataset in the data directory
    possible_paths = [
        "data/demo_move_peg/replay_buffer.zarr",  # Available peg demo
    ]
    
    zarr_path = None
    for path in possible_paths:
        if os.path.exists(path):
            zarr_path = path
            print(f"Using dataset: {path}")
            break
    
    if zarr_path is None:
        print("No suitable dataset found. Available datasets:")
        data_dir = "data"
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    replay_path = os.path.join(item_path, "replay_buffer.zarr")
                    if os.path.exists(replay_path):
                        print(f"  - {replay_path}")
        print("Please update the test with a valid dataset path.")
        print("Note: This test requires a PushT-compatible dataset.")
        return
    
    print("Testing PushTLowdimDataset")
    print("=" * 50)
    
    try:
        # Initialize dataset
        dataset = PushTLowdimDataset(
            zarr_path=zarr_path,
            horizon=16,
            pad_before=1,
            pad_after=7,
            obs_key='robot_eef_force',
            state_key='robot_eef_pose',
            action_key='action',
            seed=42,
            val_ratio=0.1,
            max_train_episodes=100
        )
        
        print("Dataset loaded successfully!")
        print(f"Dataset length: {len(dataset)}")
        print(f"Horizon: {dataset.horizon}")
        print(f"Obs key: {dataset.obs_key}")
        print(f"State key: {dataset.state_key}")
        print(f"Action key: {dataset.action_key}")
        print()
        
        # Get replay buffer info
        replay_buffer = dataset.replay_buffer
        print("Replay Buffer Info:")
        print(f"  Episodes: {replay_buffer.n_episodes}")
        print(f"  Total steps: {replay_buffer.n_steps}")
        if dataset.obs_key in replay_buffer:
            print(f"  Obs shape: {replay_buffer[dataset.obs_key].shape}")
        if dataset.state_key in replay_buffer:
            print(f"  State shape: {replay_buffer[dataset.state_key].shape}")
        if dataset.action_key in replay_buffer:
            print(f"  Action shape: {replay_buffer[dataset.action_key].shape}")
        print()
        
        # Test __getitem__ with multiple examples
        print("Sample Data from __getitem__:")
        print("-" * 30)
        
        for i in [0, len(dataset)//2, len(dataset)-1]:
            print(f"Example {i}:")
            sample = dataset[i]
            
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"    min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")
                    
                    # Show first few values for small tensors
                    if value.numel() <= 20:
                        print(f"    values: {value.flatten()[:10].tolist()}")
                    else:
                        print(f"    first 5 values: {value.flatten()[:5].tolist()}")
                        print(f"    last 5 values: {value.flatten()[-5:].tolist()}")
                else:
                    print(f"  {key}: {type(value)} = {value}")
            print()
        
        print("=" * 50)
        
        # Test validation dataset
        val_dataset = dataset.get_validation_dataset()
        print("Validation dataset created!")
        print(f"Validation dataset length: {len(val_dataset)}")
        
        if len(val_dataset) > 0:
            val_sample = val_dataset[0]
            print("Validation sample shapes:")
            for key, value in val_sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
        print()
        
        # Test get_all_actions
        all_actions = dataset.get_all_actions()
        print("All actions retrieved!")
        print(f"All actions shape: {all_actions.shape}")
        print(f"Actions range: [{all_actions.min().item():.4f}, {all_actions.max().item():.4f}]")
        
        print("All tests passed!")
        
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        print("Please ensure you have a PushT dataset available.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pusht_lowdim_dataset()
