#!/usr/bin/env python3
"""
Script to visualize the architecture of Diffusion_DCbase_Model
This script provides multiple ways to display and understand the model structure.
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
from collections import OrderedDict
import argparse

# Import the model
from model.diffusion_dcbase_model import Diffusion_DCbase_Model

def create_dummy_args():
    """Create dummy arguments for model initialization"""
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

def print_model_structure(model, show_parameters=True):
    """Print the basic model structure"""
    print("="*80)
    print("MODEL STRUCTURE")
    print("="*80)
    print(model)
    print()
    
    if show_parameters:
        print("="*80)
        print("MODEL PARAMETERS")
        print("="*80)
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            print(f"{name:50} | Shape: {str(param.shape):20} | Trainable: {param.requires_grad}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print()

def print_module_hierarchy(model, indent=0):
    """Print the module hierarchy in detail"""
    print("="*80)
    print("DETAILED MODULE HIERARCHY")
    print("="*80)
    
    def _print_modules(module, name="", indent=0):
        indent_str = "  " * indent
        if hasattr(module, '_modules') and module._modules:
            print(f"{indent_str}{name}: {module.__class__.__name__}")
            for child_name, child_module in module._modules.items():
                if child_module is not None:
                    _print_modules(child_module, child_name, indent + 1)
        else:
            param_count = sum(p.numel() for p in module.parameters())
            print(f"{indent_str}{name}: {module.__class__.__name__} (params: {param_count:,})")
    
    _print_modules(model, "Diffusion_DCbase_Model")
    print()

def analyze_forward_flow(model, input_shape=(1, 3, 228, 304)):
    """Analyze the forward flow with dummy input"""
    print("="*80)
    print("FORWARD FLOW ANALYSIS")
    print("="*80)
    
    # Create dummy sample input
    sample = {
        'rgb': torch.randn(input_shape),
        'dep': torch.randn(input_shape[0], 1, input_shape[2], input_shape[3]),
        'gt': torch.randn(input_shape[0], 1, input_shape[2], input_shape[3]),
        'depth_map': torch.randn(input_shape[0], 1, input_shape[2], input_shape[3]),
        'depth_mask': torch.ones(input_shape[0], 1, input_shape[2], input_shape[3]),
    }
    
    print(f"Input shapes:")
    for key, tensor in sample.items():
        print(f"  {key}: {tensor.shape}")
    
    print(f"\nModel components:")
    print(f"  Backbone: {model.depth_backbone.__class__.__name__}")
    print(f"  Head: {model.depth_head.__class__.__name__}")
    
    # Try to get intermediate shapes
    try:
        model.eval()
        with torch.no_grad():
            # Get backbone features
            img = sample['rgb']
            print(f"\nBackbone forward:")
            print(f"  Input shape: {img.shape}")
            
            features = model.depth_backbone(img)
            if isinstance(features, (list, tuple)):
                for i, feat in enumerate(features):
                    print(f"  Feature {i} shape: {feat.shape}")
            else:
                print(f"  Output shape: {features.shape}")
            
            print(f"\nFull model forward:")
            try:
                output = model(sample)
                if isinstance(output, dict):
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {value.shape}")
                        else:
                            print(f"  {key}: {type(value)}")
                else:
                    print(f"  Output type: {type(output)}")
            except Exception as e:
                print(f"  Error in full forward: {e}")
                
    except Exception as e:
        print(f"Error in forward analysis: {e}")
    
    print()

def try_torchsummary(model, input_shape=(3, 228, 304)):
    """Try to use torchsummary if available"""
    try:
        from torchsummary import summary
        print("="*80)
        print("TORCH SUMMARY")
        print("="*80)
        
        # Create a wrapper for the model since it expects a dict input
        class ModelWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model
            
            def forward(self, x):
                sample = {
                    'rgb': x,
                    'dep': torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device),
                    'gt': torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device),
                    'depth_map': torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device),
                    'depth_mask': torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device),
                }
                return self.model(sample)
        
        wrapped_model = ModelWrapper(model)
        summary(wrapped_model, input_shape)
        print()
        
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        print()

def try_torchviz(model, input_shape=(1, 3, 228, 304)):
    """Try to create a visualization graph using torchviz if available"""
    try:
        from torchviz import make_dot
        print("="*80)
        print("COMPUTATIONAL GRAPH VISUALIZATION")
        print("="*80)
        
        # Create dummy input
        sample = {
            'rgb': torch.randn(input_shape, requires_grad=True),
            'dep': torch.randn(input_shape[0], 1, input_shape[2], input_shape[3]),
            'gt': torch.randn(input_shape[0], 1, input_shape[2], input_shape[3]),
            'depth_map': torch.randn(input_shape[0], 1, input_shape[2], input_shape[3]),
            'depth_mask': torch.ones(input_shape[0], 1, input_shape[2], input_shape[3]),
        }
        
        model.eval()
        output = model(sample)
        
        # Create visualization
        if isinstance(output, dict) and 'pred' in output:
            dot = make_dot(output['pred'], params=dict(model.named_parameters()))
        elif isinstance(output, dict):
            # Use the first tensor output
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    dot = make_dot(value, params=dict(model.named_parameters()))
                    break
        else:
            dot = make_dot(output, params=dict(model.named_parameters()))
        
        # Save the graph
        dot.render('model_architecture_graph', format='png', cleanup=True)
        print("Computational graph saved as 'model_architecture_graph.png'")
        print()
        
    except ImportError:
        print("torchviz not available. Install with: pip install torchviz")
        print()
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Visualize Diffusion_DCbase_Model architecture')
    parser.add_argument('--input-height', type=int, default=228, help='Input height')
    parser.add_argument('--input-width', type=int, default=304, help='Input width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--show-params', action='store_true', help='Show detailed parameters')
    parser.add_argument('--create-graph', action='store_true', help='Create computational graph')
    
    args = parser.parse_args()
    
    try:
        # Create model
        print("Initializing Diffusion_DCbase_Model...")
        model_args = create_dummy_args()
        model = Diffusion_DCbase_Model(model_args)
        model.eval()
        
        input_shape = (args.batch_size, 3, args.input_height, args.input_width)
        
        # Basic structure
        print_model_structure(model, show_parameters=args.show_params)
        
        # Detailed hierarchy
        print_module_hierarchy(model)
        
        # Forward flow analysis
        analyze_forward_flow(model, input_shape)
        
        # Try external tools
        try_torchsummary(model, input_shape[1:])  # Remove batch dimension for torchsummary
        
        if args.create_graph:
            try_torchviz(model, input_shape)
        
        print("="*80)
        print("ARCHITECTURE ANALYSIS COMPLETE")
        print("="*80)
        print("Model components:")
        print(f"  - Backbone: {model.depth_backbone.__class__.__name__}")
        print(f"  - Head: {model.depth_head.__class__.__name__}")
        print(f"  - IP Basic: {model.ip_basic}")
        print(f"  - Depth Keys: {model.depth_keys}")
        print("="*80)
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed.")

if __name__ == "__main__":
    main() 