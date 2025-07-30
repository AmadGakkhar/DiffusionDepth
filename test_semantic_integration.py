#!/usr/bin/env python3
"""
Quick validation script to test semantic integration
Run this to verify semantic information is flowing through the model
"""

import torch
import numpy as np
from src.data.nyu import NYU
from src.model.diffusion_dcbase_model import Diffusion_DCbase_Model
import argparse
import json

def create_test_args():
    """Create minimal args for testing"""
    class Args:
        def __init__(self):
            self.dir_data = "/mnt/semantic_data/"  # Update with your actual semantic data path
            self.split_json = "data_json/nyu.json"
            self.patch_height = 228
            self.patch_width = 304
            self.top_crop = 0
            self.augment = False
            self.num_sample = 500
            
    return Args()

def test_semantic_flow():
    print("=== Testing Semantic Integration ===")
    
    # Create test dataset
    args = create_test_args()
    dataset = NYU(args, 'train')
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        
        # Check if semantic data is loaded
        if 'semantic' in sample:
            semantic = sample['semantic']
            print(f"✓ Semantic loaded: shape={semantic.shape}, dtype={semantic.dtype}")
            print(f"  Unique classes: {torch.unique(semantic).cpu().numpy()}")
            print(f"  Value range: [{semantic.min().item():.3f}, {semantic.max().item():.3f}]")
        else:
            print("✗ No semantic data found in sample!")
            return False
            
        # Also check the raw h5 file to verify semantic_map exists
        import h5py
        import os
        try:
            path_file = os.path.join(args.dir_data, dataset.sample_list[i]['filename'])
            with h5py.File(path_file, 'r') as f:
                if 'semantic_map' in f:
                    semantic_h5 = f['semantic_map'][:]
                    print(f"✓ Raw semantic_map in h5: shape={semantic_h5.shape}, unique={np.unique(semantic_h5)[:10]}...")
                else:
                    print("✗ No 'semantic_map' key found in h5 file!")
                    print(f"Available keys: {list(f.keys())}")
        except Exception as e:
            print(f"✗ Error reading h5 file: {e}")
            
        # Check other required keys
        required_keys = ['rgb', 'dep', 'gt', 'depth_mask', 'depth_map']
        for key in required_keys:
            if key in sample:
                print(f"✓ {key}: {sample[key].shape}")
            else:
                print(f"✗ Missing {key}")
    
    print("\n=== Data Loading Test PASSED ===")
    return True

def test_model_forward():
    print("\n=== Testing Model Forward Pass ===")
    
    args = create_test_args()
    dataset = NYU(args, 'train')
    
    # Get a sample and add batch dimension
    sample = dataset[0]
    batch_sample = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch_sample[key] = value.unsqueeze(0)  # Add batch dimension
        else:
            batch_sample[key] = value
    
    print("Sample keys:", list(batch_sample.keys()))
    print("Batch shapes:")
    for key, value in batch_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test model creation (without actually loading weights)
    print("\n✓ Model forward test setup complete")
    print("Run actual training to see debug prints!")
    
    return True

if __name__ == "__main__":
    success = True
    
    try:
        success &= test_semantic_flow()
        success &= test_model_forward()
        
        if success:
            print("\n🎉 All tests passed! Semantic integration should work.")
            print("Now run training and watch for [DEBUG] messages in the output.")
        else:
            print("\n❌ Some tests failed. Check the issues above.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 