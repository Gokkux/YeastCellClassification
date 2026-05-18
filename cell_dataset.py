import os
import json
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2 

from skimage import io 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
import pandas as pd 
import cv2 
import random

class MultiModalRobinDataset(Dataset):
    def __init__(
            self,
            cfg,
            mode: str,
            # 【核心修改】接收两个独立的 DataFrame
            df_scatter: pd.DataFrame,
            df_microscope: pd.DataFrame,
    ):

        self.mode = mode
        self.cfg = cfg
        image_size = getattr(cfg.model, 'image_size', 224) 
        # --- 1. 按类别对两个 DataFrame 进行分组 ---
        self.scatter_groups = dict(list(df_scatter.groupby('label_int')))
        self.microscope_groups = dict(list(df_microscope.groupby('label_int')))
        self.all_labels = sorted(list(self.scatter_groups.keys())) # 以 scatter 为主导
        self.paired_data = []
        self.shuffle_partners() # 调用配对函数
        # --- 【核心修改】为两种模态分别定义 Transforms ---
        # Transform for Modality A: Light Scattering
        transform_a_train = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ])
        transform_a_val_test = A.Compose([
            A.Resize(image_size, image_size),
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ])

        # Transform for Modality B: Microscope
        # 假设显微镜图使用标准的ImageNet归一化
        transform_b_train = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        transform_b_val_test = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        if mode == 'train':
            self.transform_scatter = transform_a_train
            self.transform_microscope = transform_b_train
        else: # 'val' or 'test'
            self.transform_scatter = transform_a_val_test
            self.transform_microscope = transform_b_val_test

    def shuffle_partners(self):
        """
        为训练集重新进行随机配对。
        对于验证/测试集，使用一个固定的配对。
        这个方法应该在每个训练epoch开始时被调用。
        """
        print(f"[{self.mode.upper()}] Re-pairing data...")
        self.paired_data = []
        
        for label in self.all_labels:
            if label not in self.microscope_groups:
                continue

            scatter_paths = self.scatter_groups[label]['path_scatter'].tolist()
            microscope_paths = self.microscope_groups[label]['path_microscope'].tolist()
            
            # 【关键】只在训练模式下才随机打乱
            if self.mode == 'train':
                random.shuffle(microscope_paths)
            
            num_scatter = len(scatter_paths)
            num_microscope = len(microscope_paths)
            
            # 确定数据集的最终大小
            # 训练时，长度是多数方的长度；验证/测试时，是少数方的长度
            if self.mode == 'train':
                num_pairs = max(num_scatter, num_microscope)
            else:
                num_pairs = min(num_scatter, num_microscope)

            for i in range(num_pairs):
                # 使用模运算 (%) 来实现循环配对
                scatter_path = scatter_paths[i % num_scatter]
                microscope_path = microscope_paths[i % num_microscope]
                
                self.paired_data.append({
                    'path_scatter': scatter_path,
                    'path_microscope': microscope_path,
                    'label_int': label
                })
        
        # 只有训练集需要最终打乱顺序
        if self.mode == 'train':
            random.shuffle(self.paired_data)

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, index: int):

        data_pair = self.paired_data[index]
        path_scatter = data_pair['path_scatter']
        path_microscope = data_pair['path_microscope']
        label = data_pair['label_int']
        # --- 加载和处理模态 A: 光散射图 ---
        try:
            # 假设光散射图是 .tif 格式
            image_scatter_np = self._load_and_prepare_image(io.imread(path_scatter))
            transformed_scatter = self.transform_scatter(image=image_scatter_np)
            img_scatter_tensor = transformed_scatter['image']
        except Exception as e:
            # 错误处理
            print(f"Error loading scatter image {path_scatter}: {e}. Returning dummy.")
            img_scatter_tensor = torch.zeros((3, self.cfg.model.image_size, self.cfg.model.image_size))

        # --- 加载和处理模态 B: 显微镜图 ---
        try:
            # 假设显微镜图是 .jpg 格式
            image_microscope_np = self._load_and_prepare_image(np.array(Image.open(path_microscope).convert("RGB")))
            transformed_microscope = self.transform_microscope(image=image_microscope_np)
            img_microscope_tensor = transformed_microscope['image']
        except Exception as e:
            # 错误处理
            print(f"Error loading microscope image {path_microscope}: {e}. Returning dummy.")
            img_microscope_tensor = torch.zeros((3, self.cfg.model.image_size, self.cfg.model.image_size))

        label_tensor = torch.tensor(label, dtype=torch.long)

        # 【核心修改】返回一个包含两个图像张量的元组，以及标签
        return (img_scatter_tensor, img_microscope_tensor), label_tensor

    def _load_and_prepare_image(self, image_np):
        """一个辅助函数，用于统一处理图像加载后的通道和类型问题"""
        if image_np.ndim == 2:
            image_np = np.stack([image_np]*3, axis=-1)
        elif image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        if image_np.dtype != np.uint8:
            max_val = image_np.max()
            if max_val > 0:
                image_np = (image_np / max_val * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
        return image_np

# ==============================================================================
# `build_dataloaders` 函数现在调用新的 Dataset
# ==============================================================================
def build_dataloaders(
    train_dfs, val_dfs, test_dfs, # <-- dfs 现在是元组 (df_scatter, df_microscope)
    batch_size, num_workers, cfg, distributed=False
):    # 【核心修改】将 CellLightScatteringDataset 替换为 MultiModalCellDataset
    train_scatter_df, train_microscope_df = train_dfs
    val_scatter_df, val_microscope_df = val_dfs
    test_scatter_df, test_microscope_df = test_dfs

    # 【核心修改】调用新的 Dataset
    train_dataset = MultiModalRobinDataset(cfg, 'train', df_scatter=train_scatter_df, df_microscope=train_microscope_df)
    val_dataset = MultiModalRobinDataset(cfg, 'val', df_scatter=val_scatter_df, df_microscope=val_microscope_df)
    test_dataset = MultiModalRobinDataset(cfg, 'test', df_scatter=test_scatter_df, df_microscope=test_microscope_df)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=(train_sampler is None), sampler=train_sampler, 
                                  num_workers=num_workers, pin_memory=True)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader
