import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import random

def print_h5_info(f, indent=''):
    """Recursively print information about HDF5 file/group structure."""
    for key, item in f.items():
        if isinstance(item, h5py.Dataset):
            print(f"{indent}Key: '{key}'")
            print(f"{indent}  Shape: {item.shape}")
            print(f"{indent}  Type: {item.dtype}")
            print(f"{indent}  Storage size: {item.nbytes / 1024:.2f} KB")
            if len(item.shape) > 0:
                data = item[:]
                print(f"{indent}  Value range: [{np.min(data)}, {np.max(data)}]")
        elif isinstance(item, h5py.Group):
            print(f"{indent}Group: '{key}'")
            print_h5_info(item, indent + '  ')

def save_visualizations(h5_path, output_dir, idx):
    """Save RGB, depth, and semantic visualizations from an H5 file."""
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nExamining sample {idx} - File: {h5_path}")
        print("=" * 50)
        print("H5 File Structure:")
        print_h5_info(f)
        print("=" * 50)
            
        # Save RGB
        if 'rgb' in f:
            rgb = f['rgb'][:]
            rgb = rgb.transpose(1, 2, 0)  # CHW to HWC
            rgb = rgb.astype(np.uint8)
            rgb_img = Image.fromarray(rgb)
            out_path = os.path.join(output_dir, f'sample_{idx}_rgb.png')
            rgb_img.save(out_path)
            print(f"Saved RGB to: {out_path}")
            
        # Save depth
        if 'depth' in f:
            depth = f['depth'][:]
            # Normalize depth for visualization
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
            depth_norm = depth_norm.astype(np.uint8)
            out_path = os.path.join(output_dir, f'sample_{idx}_depth.png')
            plt.imsave(out_path, depth_norm, cmap='viridis')
            print(f"Saved depth to: {out_path}")
            
        # Save semantic
        if 'semantic_map' in f:
            semantic = f['semantic_map'][:]
            out_path = os.path.join(output_dir, f'sample_{idx}_semantic.png')
            plt.imsave(out_path, semantic, cmap='tab20')
            # Save semantic label information
            unique_labels = np.unique(semantic)
            info_path = os.path.join(output_dir, f'sample_{idx}_semantic_info.txt')
            with open(info_path, 'w') as f_info:
                f_info.write(f"Unique semantic labels: {unique_labels}\n")
                f_info.write(f"Number of unique labels: {len(unique_labels)}\n")
                f_info.write(f"Label counts:\n")
                for label in unique_labels:
                    count = np.sum(semantic == label)
                    percentage = count / semantic.size * 100
                    f_info.write(f"  Label {label}: {count} pixels ({percentage:.2f}%)\n")
            print(f"Saved semantic visualization to: {out_path}")
            print(f"Saved semantic information to: {info_path}")

def main():
    # Directory containing your H5 files
    data_dir = '/mnt/semantic_data/train'
    output_dir = 'h5_examination_output'
    num_samples = 3
    
    # Get all H5 files
    h5_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    
    if not h5_files:
        print(f"No H5 files found in {data_dir}")
        return
        
    print(f"Found {len(h5_files)} H5 files")
    
    # Select random samples
    selected_files = random.sample(h5_files, min(num_samples, len(h5_files)))
    
    # Process each selected file
    for idx, h5_file in enumerate(selected_files):
        print(f"\nProcessing file {idx + 1}/{len(selected_files)}")
        try:
            save_visualizations(h5_file, output_dir, idx)
        except Exception as e:
            print(f"Error processing {h5_file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()