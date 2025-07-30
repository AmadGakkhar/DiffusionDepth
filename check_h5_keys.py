#!/usr/bin/env python3
"""
Quick script to check if h5 files contain semantic_map key
"""

import h5py
import json
import os
import numpy as np

def check_h5_files():
    # Load the dataset split
    with open('data_json/nyu.json', 'r') as f:
        nyu_data = json.load(f)
    
    # Check a few training files
    data_dir = "/mnt/semantic_data/"  # Update this path
    
    print("=== Checking H5 Files for semantic_map ===")
    
    for i, sample in enumerate(nyu_data['train'][:5]):  # Check first 5 files
        filepath = os.path.join(data_dir, sample['filename'])
        
        print(f"\n--- File {i+1}: {sample['filename']} ---")
        
        if not os.path.exists(filepath):
            print(f"✗ File does not exist: {filepath}")
            continue
            
        try:
            with h5py.File(filepath, 'r') as f:
                print(f"Available keys: {list(f.keys())}")
                
                if 'semantic_map' in f:
                    semantic_data = f['semantic_map'][:]
                    print(f"✓ semantic_map found: shape={semantic_data.shape}, dtype={semantic_data.dtype}")
                    print(f"  Unique values: {np.unique(semantic_data)[:10]}...")  # First 10 unique values
                    print(f"  Value range: [{semantic_data.min()}, {semantic_data.max()}]")
                else:
                    print("✗ No 'semantic_map' key found!")
                    
        except Exception as e:
            print(f"✗ Error reading file: {e}")
    
    print("\n=== Summary ===")
    print("If you see 'semantic_map found' messages above, your data is ready!")
    print("If not, check that:")
    print("1. The data_dir path is correct")
    print("2. Your h5 files actually contain semantic_map data")
    print("3. The files are accessible")

if __name__ == "__main__":
    check_h5_files() 