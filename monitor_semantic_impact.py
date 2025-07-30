#!/usr/bin/env python3
"""
Monitor semantic impact on training
This script compares loss behavior with/without semantic information
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def create_loss_monitor():
    """Create a simple loss monitoring system"""
    
    class LossMonitor:
        def __init__(self):
            self.losses_with_semantic = []
            self.losses_without_semantic = []
            self.iteration = 0
            self.start_time = time.time()
            
        def log_loss(self, loss_value, has_semantic=True):
            if has_semantic:
                self.losses_with_semantic.append((self.iteration, loss_value))
            else:
                self.losses_without_semantic.append((self.iteration, loss_value))
            self.iteration += 1
            
        def print_stats(self, every_n=10):
            if self.iteration % every_n == 0:
                elapsed = time.time() - self.start_time
                
                if self.losses_with_semantic:
                    recent_with = np.mean([l[1] for l in self.losses_with_semantic[-10:]])
                else:
                    recent_with = float('nan')
                    
                if self.losses_without_semantic:
                    recent_without = np.mean([l[1] for l in self.losses_without_semantic[-10:]])
                else:
                    recent_without = float('nan')
                
                print(f"[MONITOR] Iter {self.iteration:4d} | "
                      f"Loss w/ semantic: {recent_with:.4f} | "
                      f"Loss w/o semantic: {recent_without:.4f} | "
                      f"Time: {elapsed:.1f}s")
                
        def save_plot(self, filename="semantic_loss_comparison.png"):
            plt.figure(figsize=(12, 6))
            
            if self.losses_with_semantic:
                iters, losses = zip(*self.losses_with_semantic)
                plt.plot(iters, losses, 'b-', label='With Semantic', alpha=0.7)
                
            if self.losses_without_semantic:
                iters, losses = zip(*self.losses_without_semantic)
                plt.plot(iters, losses, 'r-', label='Without Semantic', alpha=0.7)
                
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss: With vs Without Semantic Information')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filename}")
    
    return LossMonitor()

# Global monitor instance
loss_monitor = create_loss_monitor()

def add_loss_monitoring_to_training():
    """
    Instructions to add to your training script:
    
    1. Import this module:
       from monitor_semantic_impact import loss_monitor
    
    2. In your training loop, after computing loss:
       has_semantic = sample.get('semantic') is not None
       loss_monitor.log_loss(loss.item(), has_semantic)
       loss_monitor.print_stats()
    
    3. At the end of training:
       loss_monitor.save_plot()
    """
    pass

if __name__ == "__main__":
    print("=== Semantic Impact Monitor ===")
    print("This module provides loss monitoring capabilities.")
    print("Add the following to your training script:")
    print()
    print("1. Import: from monitor_semantic_impact import loss_monitor")
    print("2. In training loop:")
    print("   has_semantic = sample.get('semantic') is not None")
    print("   loss_monitor.log_loss(loss.item(), has_semantic)")
    print("   loss_monitor.print_stats()")
    print("3. At end: loss_monitor.save_plot()")
    print()
    print("This will show you immediately if semantic info affects loss!") 