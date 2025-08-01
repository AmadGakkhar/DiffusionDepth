#!/usr/bin/env python3
"""
Experiment Results Plotting Script

This script plots losses and metrics from DiffusionDepth experiment folders.
The script creates separate plots for training, validation, and test data.

USAGE:
1. Modify the EXPERIMENTS dictionary below to specify your experiment folders and custom names
2. Adjust BASE_DIR if your experiments are in a different location
3. Run: python plot_experiments.py

FEATURES:
- Separate plots for train/validation/test data
- Custom experiment names
- Automatic data parsing from loss_*.txt and metric_*.txt files
- Saves plots as PNG files in experiment_plots/ directory
- Console summary with final epoch values

CONFIGURATION:
- EXPERIMENTS: Dictionary mapping folder paths to custom display names
- BASE_DIR: Base directory containing experiment folders
- SAVE_PLOTS: Set to True to save plots, False to only display
"""

import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ==================== CONFIGURATION ====================
# Modify these experiment folders and their custom names as needed
EXPERIMENTS = {
    "experiments/250731_034741_semantic_depth_mid_fusion": "Late Fusion",
    "experiments/250729_071011_NYU_Res50_Semantic_train": "Early Fusion",
    "experiments/250716_060541_NYU_Res50_train": "Baseline"
}

# Base directory (change if needed)
BASE_DIR = "/home/ubuntu/code/dd/DiffusionDepth-Orig"

# Plot settings
FIGURE_SIZE = (15, 10)
DPI = 100
SAVE_PLOTS = True  # Set to False if you don't want to save plots
OUTPUT_DIR = "experiment_plots"

# ==================== DATA PARSING FUNCTIONS ====================

def parse_metric_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a single metric line and return dictionary of metric values."""
    if not line.strip() or '|' not in line:
        return None
    
    parts = line.strip().split('|')
    if len(parts) < 3:
        return None
    
    epoch_str = parts[0].strip()
    metric_part = parts[2].strip()
    
    try:
        epoch = int(epoch_str)
    except ValueError:
        return None
    
    # Parse metrics using regex
    metrics = {'epoch': epoch}
    metric_patterns = {
        'RMSE': r'RMSE:\s*([\d.]+)',
        'MAE': r'MAE:\s*([\d.]+)',
        'iRMSE': r'iRMSE:\s*([\d.]+)',
        'iMAE': r'iMAE:\s*([\d.]+)',
        'REL': r'REL:\s*([\d.]+)',
        'D1': r'D\^1:\s*([\d.]+)',
        'D2': r'D\^2:\s*([\d.]+)',
        'D3': r'D\^3:\s*([\d.]+)'
    }
    
    for metric_name, pattern in metric_patterns.items():
        match = re.search(pattern, metric_part)
        if match:
            metrics[metric_name] = float(match.group(1))
    
    return metrics

def parse_loss_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a single loss line and return dictionary of loss values."""
    if not line.strip() or '|' not in line:
        return None
    
    parts = line.strip().split('|')
    if len(parts) < 3:
        return None
    
    epoch_str = parts[0].strip()
    loss_part = parts[2].strip()
    
    try:
        epoch = int(epoch_str)
    except ValueError:
        return None
    
    # Parse losses using regex
    losses = {'epoch': epoch}
    loss_patterns = {
        'L1': r'L1:\s*([\d.]+)',
        'L2': r'L2:\s*([\d.]+)',
        'DDIM': r'DDIM:\s*([\d.]+)',
        'Total': r'Total:\s*([\d.]+)'
    }
    
    for loss_name, pattern in loss_patterns.items():
        match = re.search(pattern, loss_part)
        if match:
            losses[loss_name] = float(match.group(1))
    
    return losses

def load_data_file(filepath: str, parser_func) -> List[Dict[str, float]]:
    """Load and parse data from a file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return []
    
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parsed = parser_func(line)
                if parsed:
                    data.append(parsed)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    return data

def get_experiment_name(folder_path: str) -> str:
    """Extract experiment name from args.json or use folder name."""
    args_path = os.path.join(folder_path, 'args.json')
    if os.path.exists(args_path):
        try:
            with open(args_path, 'r') as f:
                args = json.load(f)
                if 'save' in args:
                    return args['save']
        except Exception:
            pass
    
    # Fallback to folder name
    return os.path.basename(folder_path)

# ==================== PLOTTING FUNCTIONS ====================

def plot_losses_by_split(all_data: Dict, save_dir: Optional[str] = None):
    """Plot loss curves for all experiments, separated by train/val/test."""
    splits = ['train', 'val', 'test']
    loss_types = ['L1', 'L2', 'DDIM', 'Total']
    
    for split in splits:
        # Check if any experiment has data for this split
        has_data = any(data.get(f'loss_{split}') for data in all_data.values())
        if not has_data:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, dpi=DPI)
        fig.suptitle(f'{split.capitalize()} Losses Comparison', fontsize=16, fontweight='bold')
        
        for idx, loss_type in enumerate(loss_types):
            ax = axes[idx // 2, idx % 2]
            
            # Plot losses for this split
            for exp_name, data in all_data.items():
                loss_data = data.get(f'loss_{split}', [])
                if loss_data:
                    epochs = [d['epoch'] for d in loss_data if loss_type in d]
                    values = [d[loss_type] for d in loss_data if loss_type in d]
                    if epochs and values:
                        ax.plot(epochs, values, 'o-', label=exp_name, linewidth=2, markersize=4)
            
            ax.set_title(f'{loss_type} Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Value')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_dir and SAVE_PLOTS:
            os.makedirs(save_dir, exist_ok=True)
            filename = f'losses_{split}_comparison.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=DPI, bbox_inches='tight')
            print(f"Saved {split} losses plot to {os.path.join(save_dir, filename)}")
        
        plt.show()

def plot_metrics_by_split(all_data: Dict, save_dir: Optional[str] = None):
    """Plot metric curves for all experiments, separated by train/val/test."""
    splits = ['train', 'val', 'test']
    metrics = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D1', 'D2', 'D3']
    
    for split in splits:
        # Check if any experiment has data for this split
        has_data = any(data.get(f'metric_{split}') for data in all_data.values())
        if not has_data:
            continue
            
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=DPI)
        fig.suptitle(f'{split.capitalize()} Metrics Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 4, idx % 4]
            
            # Plot metrics for this split
            for exp_name, data in all_data.items():
                metric_data = data.get(f'metric_{split}', [])
                if metric_data:
                    epochs = [d['epoch'] for d in metric_data if metric in d]
                    values = [d[metric] for d in metric_data if metric in d]
                    if epochs and values:
                        ax.plot(epochs, values, 'o-', label=exp_name, linewidth=2, markersize=4)
            
            ax.set_title(f'{metric} Metric', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metric Value')
            ax.grid(True, alpha=0.3)
            if idx == 0:  # Only show legend on first subplot to avoid clutter
                ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_dir and SAVE_PLOTS:
            os.makedirs(save_dir, exist_ok=True)
            filename = f'metrics_{split}_comparison.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=DPI, bbox_inches='tight')
            print(f"Saved {split} metrics plot to {os.path.join(save_dir, filename)}")
        
        plt.show()

def plot_summary_table(all_data: Dict, save_dir: Optional[str] = None):
    """Create a summary table with final epoch values."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY - FINAL EPOCH VALUES")
    print("="*80)
    
    for exp_name, data in all_data.items():
        print(f"\n📊 {exp_name}")
        print("-" * len(exp_name))
        
        # Training metrics and loss
        if data['metric_train']:
            final_train = data['metric_train'][-1]
            print(f"  🏋️  Training (Epoch {final_train['epoch']}): RMSE={final_train.get('RMSE', 'N/A'):.4f}, "
                  f"MAE={final_train.get('MAE', 'N/A'):.4f}, REL={final_train.get('REL', 'N/A'):.4f}")
        if data['loss_train']:
            final_loss = data['loss_train'][-1]
            print(f"      Training Loss: Total={final_loss.get('Total', 'N/A'):.4f}")
        
        # Validation metrics and loss
        if data['metric_val']:
            final_val = data['metric_val'][-1]
            print(f"  🔍 Validation (Epoch {final_val['epoch']}): RMSE={final_val.get('RMSE', 'N/A'):.4f}, "
                  f"MAE={final_val.get('MAE', 'N/A'):.4f}, REL={final_val.get('REL', 'N/A'):.4f}")
        if data['loss_val']:
            final_val_loss = data['loss_val'][-1]
            print(f"      Validation Loss: Total={final_val_loss.get('Total', 'N/A'):.4f}")
        
        # Test metrics and loss
        if data['metric_test']:
            final_test = data['metric_test'][-1]
            print(f"  🧪 Test (Epoch {final_test['epoch']}): RMSE={final_test.get('RMSE', 'N/A'):.4f}, "
                  f"MAE={final_test.get('MAE', 'N/A'):.4f}, REL={final_test.get('REL', 'N/A'):.4f}")
        if data['loss_test']:
            final_test_loss = data['loss_test'][-1]
            print(f"      Test Loss: Total={final_test_loss.get('Total', 'N/A'):.4f}")

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to load data and create plots."""
    print("🚀 Starting Experiment Analysis...")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📊 Analyzing {len(EXPERIMENTS)} experiments")
    
    all_data = {}
    
    # Load data from all experiments
    for folder, custom_name in EXPERIMENTS.items():
        full_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(full_path):
            print(f"⚠️  Warning: Experiment folder not found: {full_path}")
            continue
        
        exp_name = custom_name  # Use the custom name instead of extracting from args.json
        print(f"\n📈 Loading data for: {exp_name}")
        
        data = {}
        
        # Load loss files
        data['loss_train'] = load_data_file(os.path.join(full_path, 'loss_train.txt'), parse_loss_line)
        data['loss_val'] = load_data_file(os.path.join(full_path, 'loss_val.txt'), parse_loss_line)
        data['loss_test'] = load_data_file(os.path.join(full_path, 'loss_test.txt'), parse_loss_line)
        
        # Load metric files
        data['metric_train'] = load_data_file(os.path.join(full_path, 'metric_train.txt'), parse_metric_line)
        data['metric_val'] = load_data_file(os.path.join(full_path, 'metric_val.txt'), parse_metric_line)
        data['metric_test'] = load_data_file(os.path.join(full_path, 'metric_test.txt'), parse_metric_line)
        
        all_data[exp_name] = data
        
        # Print data summary
        train_epochs = len(data['loss_train'])
        val_epochs = len(data['loss_val'])
        test_epochs = len(data['loss_test'])
        print(f"  ✅ Loaded {train_epochs} training epochs, {val_epochs} validation epochs, {test_epochs} test epochs")
    
    if not all_data:
        print("❌ No valid experiment data found!")
        return
    
    # Create output directory if saving plots
    output_dir = None
    if SAVE_PLOTS:
        output_dir = os.path.join(BASE_DIR, OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Plots will be saved to: {output_dir}")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    plot_losses_by_split(all_data, output_dir)
    plot_metrics_by_split(all_data, output_dir)
    
    # Print summary
    plot_summary_table(all_data)
    
    print(f"\n✅ Analysis complete! Analyzed {len(all_data)} experiments.")
    if SAVE_PLOTS:
        print(f"📁 Plots saved to: {output_dir}")

if __name__ == "__main__":
    main() 