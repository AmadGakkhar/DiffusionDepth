"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NYU Depth V2 Dataset Helper
"""


import os
import warnings
import numpy as np
import json
import h5py
from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from model.ops.depth_map_proc import simple_depth_completion 

warnings.filterwarnings("ignore", category=UserWarning)

"""
NYUDepthV2 json file has a following format:

{
    "train": [
        {
            "filename": "train/bedroom_0078/00066.h5"
        }, ...
    ],
    "val": [
        {
            "filename": "train/study_0008/00351.h5"
        }, ...
    ],
    "test": [
        {
            "filename": "val/official/00001.h5"
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""


class NYU(BaseDataset):
    def __init__(self, args, mode):
        super(NYU, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # For NYUDepthV2, crop size is fixed
        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        # Camera intrinsics [fx, fy, cx, cy]
        self.K = torch.Tensor([
            5.1885790117450188e+02 / 2.0,
            5.1946961112127485e+02 / 2.0,
            3.2558244941119034e+02 / 2.0 - 8.0,
            2.5373616633400465e+02 / 2.0 - 6.0
        ])

        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(self.args.dir_data,
                                 self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        # Debug: Check if semantic_map exists
        if 'semantic_map' not in f:
            print(f"[WARNING] No 'semantic_map' in {path_file}")
            print(f"Available keys: {list(f.keys())}")
            # Create dummy semantic map if missing
            semantic_h5 = np.zeros_like(dep_h5, dtype=np.uint8)
        else:
            semantic_h5 = f['semantic_map'][:]

        # Normalize semantic shape/kind
        semantic_kind = 'label'
        if semantic_h5.ndim == 3:
            if semantic_h5.shape[0] > 1 and semantic_h5.shape[0] < 1000:
                # one-hot [C,H,W]
                semantic_h5 = np.argmax(semantic_h5, axis=0).astype('uint8')
            elif semantic_h5.shape[2] == 1:
                semantic_h5 = semantic_h5[..., 0].astype('uint8')
            elif semantic_h5.shape[2] == 3:
                semantic_kind = 'rgb'
                semantic_h5 = semantic_h5.astype('uint8')
        else:
            semantic_h5 = semantic_h5.astype('uint8')

        # Create PIL image accordingly
        if semantic_kind == 'rgb':
            semantic = Image.fromarray(semantic_h5, mode='RGB')
        else:
            semantic = Image.fromarray(semantic_h5, mode='L')

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')

        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            # Use already loaded semantic data
            semantic = Image.fromarray(semantic_h5, mode='L')

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                semantic = TF.hflip(semantic)

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)
            semantic = TF.rotate(semantic, angle=degree, resample=Image.NEAREST)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale

            t_sem = T.Compose([
                T.Resize(scale, interpolation=Image.NEAREST),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.Lambda(lambda x: torch.from_numpy(x.astype(np.int64)).unsqueeze(0))
            ])
            semantic = t_sem(semantic)

        else:
            # Use already loaded semantic data
            semantic = Image.fromarray(semantic_h5, mode='L')
            
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            K = self.K.clone()

            t_sem = T.Compose([
                T.Resize(self.height, interpolation=Image.NEAREST),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.Lambda(lambda x: torch.from_numpy(x.astype(np.int64)).unsqueeze(0))
            ])
            semantic = t_sem(semantic)

        dep_sp = self.get_sparse_depth(dep, self.args.num_sample)

        """
        Add depth mask and simple_map
        """
        # here >=0 denotes actually we do not apply depth mask if >0 we apply 
        depth_mask = (dep_sp > 0)
        depth_maps = []
        # 这里有个bug，需要每个是-1
        for sparse_map in dep_sp: 
            depth_map = np.asarray(sparse_map, dtype=np.float32)
            depth_map, _ = simple_depth_completion(depth_map)
            depth_maps.append(depth_map)
        depth_maps = np.stack(depth_maps)  # bs, h, w

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'depth_mask': depth_mask, 'depth_map': depth_maps, 'semantic': semantic}

        return output

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp
