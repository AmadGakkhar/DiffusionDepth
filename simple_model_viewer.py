#!/usr/bin/env python3
"""
Simple script to quickly view the Diffusion_DCbase_Model architecture
Usage: python simple_model_viewer.py
"""

import sys
sys.path.append('src')

import torch
from model.diffusion_dcbase_model import Diffusion_DCbase_Model

def create_args():
    class Args:
        def __init__(self):
            # Backbone configuration
            self.backbone_name = "mmbev_res18"  # Fixed: use the function name, not class name
            self.backbone_module = "mmbev_resnet"
            
            # Head configuration
            self.head_specify = "DDIMDepthEstimate_Res"
            self.inference_steps = 20
            self.num_train_timesteps = 1000
            
            # Additional required parameters
            self.model_name = "Diffusion_DCbase_Model"
            self.network = "resnet18"
            self.prop_kernel = 3
            self.affinity = "TGASS"
            self.conf_prop = True
    return Args()

def main():
    print("Loading Diffusion_DCbase_Model...")
    
    # Create model
    args = create_args()
    model = Diffusion_DCbase_Model(args)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    
    print("\n" + "="*60)
    print("MODEL COMPONENTS")
    print("="*60)
    print(f"Backbone: {model.depth_backbone.__class__.__name__}")
    print(f"Head: {model.depth_head.__class__.__name__}")
    
    print("\n" + "="*60)
    print("PARAMETER COUNT")
    print("="*60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    print("\n" + "="*60)
    print("INPUT/OUTPUT SHAPES")
    print("="*60)
    
    # Test with dummy input
    sample = {
        'rgb': torch.randn(2, 3, 228, 304),
        'dep': torch.randn(2, 1, 228, 304),
        'gt': torch.randn(2, 1, 228, 304),
        'depth_map': torch.randn(2, 1, 228, 304),
        'depth_mask': torch.ones(2, 1, 228, 304),
    }
    
    print("Input shapes:")
    for key, tensor in sample.items():
        print(f"  {key}: {tensor.shape}")
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample)
            print("\nOutput shapes:")
            if isinstance(output, dict):
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
    except Exception as e:
        print(f"\nError during forward pass: {e}")

if __name__ == "__main__":
    main() 