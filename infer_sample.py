import os
import sys
import argparse
from types import SimpleNamespace

import torch
import numpy as np
# Compatibility shim for NumPy>=1.24 where np.long was removed
if not hasattr(np, 'long'):
    np.long = int
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
sys.path.append(SRC_DIR)

from model import get as get_model  # noqa: E402
from data import get as get_data  # noqa: E402


def build_args(
    dir_data: str,
    data_name: str,
    split_json: str,
    checkpoint: str,
    output_dir: str,
    backbone_module: str,
    backbone_name: str,
    head_specify: str,
    patch_height: int,
    patch_width: int,
    top_crop: int,
    inference_steps: int,
    num_train_timesteps: int,
    max_depth: float,
) -> SimpleNamespace:
    # Only include attributes that datasets/model expect to read
    args = SimpleNamespace(
        # dataset
        dir_data=dir_data,
        data_name=data_name,
        split_json=split_json,
        patch_height=patch_height,
        patch_width=patch_width,
        top_crop=top_crop,
        augment=False,
        num_sample=0,
        test_crop=False,
        # hardware
        num_threads=1,
        opt_level='O0',
        # model
        model_name='Diffusion_DCbase_',
        backbone_module=backbone_module,
        backbone_name=backbone_name,
        head_specify=head_specify,
        inference_steps=inference_steps,
        num_train_timesteps=num_train_timesteps,
        # logging/saving
        max_depth=max_depth,
        # unused but referenced in some places
        num_summary=1,
        save_result_only=False,
        save_raw_npdepth=False,
    )
    args.pretrain = checkpoint
    return args


def unnormalize_rgb(rgb_tensor: torch.Tensor) -> np.ndarray:
    # rgb_tensor shape: [1, 3, H, W]
    img_mean = torch.tensor((0.485, 0.456, 0.406), device=rgb_tensor.device).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225), device=rgb_tensor.device).view(1, 3, 1, 1)
    rgb = rgb_tensor.clone()
    rgb.mul_(img_std).add_(img_mean)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb_np = (rgb[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    return rgb_np


def colorize_depth(depth: np.ndarray, max_depth: float, cmap_name: str = 'plasma') -> np.ndarray:
    # depth expected in meters (H, W)
    depth = np.clip(depth, 0.0, max_depth)
    norm = (depth / max_depth)  # 0..1
    cm = plt.get_cmap(cmap_name)
    colored = (cm((norm * 255.0).astype(np.uint8))[:, :, :3] * 255.0).astype(np.uint8)
    return colored


# Remove model semantic usage; keep only dataset-provided semantic_map

def save_semantic_map_from_sample(sample: dict, out_dir: str) -> bool:
    if 'semantic_map' not in sample or sample['semantic_map'] is None:
        return False
    sem = sample['semantic_map']
    # sem can be numpy or torch
    if isinstance(sem, np.ndarray):
        sem_np = sem
    elif torch.is_tensor(sem):
        sem_np = sem.detach().cpu().numpy()
    else:
        return False

    # Handle shapes
    # If [C, H, W] and C>1 -> logits/one-hot
    if sem_np.ndim == 3 and sem_np.shape[0] > 1:
        sem_np = np.argmax(sem_np, axis=0)
    # If [1, H, W] -> squeeze
    if sem_np.ndim == 3 and sem_np.shape[0] == 1:
        sem_np = sem_np[0]

    # If already RGB image [H, W, 3]
    if sem_np.ndim == 3 and sem_np.shape[2] == 3:
        img = Image.fromarray(sem_np.astype(np.uint8), 'RGB')
        img.save(os.path.join(out_dir, '03_semantic.png'))
        return True

    # Otherwise assume label map [H, W]
    sem_np = sem_np.astype(np.int32)
    palette = (plt.get_cmap('tab20').colors * 255.0)
    palette = np.array(palette, dtype=np.uint8)
    colored = np.zeros((*sem_np.shape, 3), dtype=np.uint8)
    for cls_id in np.unique(sem_np):
        color = palette[int(cls_id) % len(palette)]
        colored[sem_np == cls_id] = color

    Image.fromarray(colored, 'RGB').save(os.path.join(out_dir, '03_semantic.png'))
    np.save(os.path.join(out_dir, '03_semantic_raw.npy'), sem_np)
    return True


def main():
    parser = argparse.ArgumentParser(description='Single-sample inference and save outputs')
    parser.add_argument('--dir_data', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--data_name', type=str, default='NYU', choices=['NYU', 'KITTIDC'])
    parser.add_argument('--split_json', type=str, required=True, help='Path to JSON split file')
    parser.add_argument('--index', type=int, default=0, help='Index of sample in split to run')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint (model_xxxxx.pt)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')

    # Model/backbone config
    parser.add_argument('--backbone_module', type=str, default='mmbev_resnet', choices=['mmbev_resnet', 'swin', 'mpvit'])
    parser.add_argument('--backbone_name', type=str, default='mmbev_res50')
    parser.add_argument('--head_specify', type=str, default='DDIMDepthEstimate_Res',
                        choices=['DDIMDepthEstimate_Res', 'DDIMDepthEstimate_Swin_ADD',
                                 'DDIMDepthEstimate_Swin_ADDHAHI', 'DDIMDepthEstimate_ResVis',
                                 'DDIMDepthEstimate_Swin_ADDHAHIVis', 'DDIMDepthEstimate_MPVIT_ADDHAHI'])
    parser.add_argument('--inference_steps', type=int, default=20)
    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--max_depth', type=float, default=88.0)

    # Input sizing (KITTI uses these; NYU is ignored and internally fixed)
    parser.add_argument('--patch_height', type=int, default=352)
    parser.add_argument('--patch_width', type=int, default=1216)
    parser.add_argument('--top_crop', type=int, default=0)

    args_cli = parser.parse_args()

    os.makedirs(args_cli.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build args used by internal modules
    args = build_args(
        dir_data=args_cli.dir_data,
        data_name=args_cli.data_name,
        split_json=args_cli.split_json,
        checkpoint=args_cli.checkpoint,
        output_dir=args_cli.output_dir,
        backbone_module=args_cli.backbone_module,
        backbone_name=args_cli.backbone_name,
        head_specify=args_cli.head_specify,
        patch_height=args_cli.patch_height,
        patch_width=args_cli.patch_width,
        top_crop=args_cli.top_crop,
        inference_steps=args_cli.inference_steps,
        num_train_timesteps=args_cli.num_train_timesteps,
        max_depth=args_cli.max_depth,
    )

    # Build dataset and get a single sample
    data_cls = get_data(args)
    dataset = data_cls(args, 'test')
    idx = max(0, min(args_cli.index, len(dataset) - 1))
    sample = dataset[idx]

    # Convert to batched tensors and move to device; ensure depth_map is tensor
    def to_batched(t):
        if t is None:
            return None
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        if torch.is_tensor(t):
            t = t.unsqueeze(0)
            return t.to(device)
        return t

    batched_sample = {}
    for key in ['rgb', 'dep', 'gt', 'K', 'depth_mask', 'depth_map']:
        batched_sample[key] = to_batched(sample[key])
    # Include semantic_map if present (model may ignore; safe to pass)
    if 'semantic_map' in sample:
        batched_sample['semantic_map'] = to_batched(sample['semantic_map'])

    # Build model
    model_cls = get_model(args)
    net = model_cls(args).to(device)
    net.eval()

    # Load checkpoint
    assert os.path.exists(args.pretrain), f'Checkpoint not found: {args.pretrain}'
    checkpoint = torch.load(args.pretrain, map_location=device)
    missing, unexpected = net.load_state_dict(checkpoint['net'], strict=False)
    if unexpected:
        print('Unexpected keys in checkpoint:', unexpected)
    if missing:
        print('Missing keys in checkpoint:', missing)

    # Inference
    with torch.no_grad():
        output = net(batched_sample)

    # Save RGB
    rgb_np = unnormalize_rgb(batched_sample['rgb'])
    Image.fromarray(rgb_np, 'RGB').save(os.path.join(args_cli.output_dir, '01_rgb.png'))

    # Save predicted depth: color and raw 16-bit
    pred = output['pred']  # [1, 1, H, W]
    if torch.is_tensor(pred):
        pred_np = pred[0, 0].detach().cpu().numpy()
    else:
        pred_np = np.array(pred)[0, 0]

    # Raw 16-bit depth (scaled by 256, like KITTI format)
    raw16 = (np.clip(pred_np, 0.0, args.max_depth) * 256.0).astype(np.uint16)
    Image.fromarray(raw16, mode='I;16').save(os.path.join(args_cli.output_dir, '02_pred_depth_raw16.png'))

    # Colored
    colored = colorize_depth(pred_np, args.max_depth, cmap_name='plasma')
    Image.fromarray(colored, 'RGB').save(os.path.join(args_cli.output_dir, '02_pred_depth_color.png'))

    # Optional: refined depth if exists
    if 'refineddepth' in output and output['refineddepth'] is not None:
        rd = output['refineddepth']
        if torch.is_tensor(rd):
            rd_np = rd[0, 0].detach().cpu().numpy()
        else:
            rd_np = np.array(rd)[0, 0]
        rd_colored = colorize_depth(rd_np, args.max_depth, cmap_name='plasma')
        Image.fromarray(rd_colored, 'RGB').save(os.path.join(args_cli.output_dir, '02b_refined_depth_color.png'))

    # Save semantic map only if provided by dataset sample
    if not save_semantic_map_from_sample(sample, args_cli.output_dir):
        with open(os.path.join(args_cli.output_dir, '03_semantic_info.txt'), 'w') as f:
            f.write('No semantic_map provided in dataset sample. Skipped saving semantic map.')

    # Also save numpy for analysis
    np.save(os.path.join(args_cli.output_dir, '02_pred_depth_raw.npy'), pred_np)

    print(f'Saved results to: {args_cli.output_dir}')


if __name__ == '__main__':
    main() 