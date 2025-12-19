"""
数据集模块 (PyTorch 版本)
提供 LIDC-IDRI 数据的加载、预处理和数据增强功能
"""
import os
import numpy as np
from glob import glob
from typing import Tuple, List, Optional, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    USE_AUGMENTATION, AUGMENTATION_CONFIG, RANDOM_SEED
)


class LIDCDataset(Dataset):
    """
    LIDC-IDRI 数据集类 (PyTorch Dataset)
    负责加载和预处理肺结节CT图像及其分割掩码
    """
    
    def __init__(self, data_dir: str, 
                 img_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
                 augment: bool = False,
                 transform: Optional[Callable] = None):
        """
        初始化数据集
        
        参数:
            data_dir: 数据目录路径（应包含病例子文件夹）
            img_size: 目标图像尺寸 (height, width)
            augment: 是否进行数据增强
            transform: 自定义变换函数
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.transform = transform
        
        self.image_paths = []
        self.mask_paths = []
        
        self._scan_data_directory()
        
        # 基础变换
        self.resize = T.Resize(img_size)
        self.to_tensor = T.ToTensor()
    
    def _scan_data_directory(self):
        """扫描数据目录，获取所有图像和掩码的路径"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 遍历病例 -> 结节 -> images/masks
        for patient_id in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue
            
            for nodule in os.listdir(patient_path):
                nodule_path = os.path.join(patient_path, nodule)
                if not os.path.isdir(nodule_path):
                    continue
                
                images_dir = os.path.join(nodule_path, "images")
                masks_dir = os.path.join(nodule_path, "masks")
                
                if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                    continue
                
                # 获取所有图像文件
                for img_file in sorted(os.listdir(images_dir)):
                    if not img_file.endswith('.png'):
                        continue
                    
                    img_path = os.path.join(images_dir, img_file)
                    mask_path = os.path.join(masks_dir, img_file)
                    
                    # 只有当对应的mask存在时才添加
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
        
        print(f"  ✓ 扫描完成: 找到 {len(self.image_paths)} 对图像-掩码")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # 灰度图
        
        # 调整大小
        image = self.resize(image)
        mask = self.resize(mask)
        
        # 数据增强（图像和掩码需要同步变换）
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # 自定义变换
        if self.transform:
            image = self.transform(image)
        
        # 转换为张量
        image = self.to_tensor(image)  # [C, H, W], 范围 [0, 1]
        mask = self.to_tensor(mask)    # [1, H, W], 范围 [0, 1]
        
        # 二值化掩码
        mask = (mask > 0.5).float()
        
        return image, mask
    
    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        数据增强 - 对图像和掩码应用相同的几何变换
        """
        config = AUGMENTATION_CONFIG
        
        # 随机水平翻转
        if config.get('horizontal_flip', False) and torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 随机垂直翻转
        if config.get('vertical_flip', False) and torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # 随机旋转
        rotation_range = config.get('rotation_range', 0)
        if rotation_range > 0:
            angle = float(torch.empty(1).uniform_(-rotation_range, rotation_range))
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # 随机亮度调整（仅对图像）
        brightness_range = config.get('brightness_range', 0)
        if brightness_range > 0:
            factor = float(torch.empty(1).uniform_(1 - brightness_range, 1 + brightness_range))
            image = TF.adjust_brightness(image, factor)
        
        # 随机对比度调整（仅对图像）
        contrast_range = config.get('contrast_range', 0)
        if contrast_range > 0:
            factor = float(torch.empty(1).uniform_(1 - contrast_range, 1 + contrast_range))
            image = TF.adjust_contrast(image, factor)
        
        return image, mask


def create_dataloaders(train_dir: str, val_dir: str, test_dir: str,
                       batch_size: int = BATCH_SIZE,
                       num_workers: int = NUM_WORKERS,
                       augment_train: bool = USE_AUGMENTATION) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试 DataLoader
    
    参数:
        train_dir, val_dir, test_dir: 各数据集的目录路径
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        augment_train: 是否对训练数据进行增强
    
    返回:
        (train_loader, val_loader, test_loader)
    """
    print("\n创建 DataLoader...")
    
    # 创建数据集
    print("  [1/3] 加载训练集...")
    train_dataset = LIDCDataset(train_dir, augment=augment_train)
    
    print("  [2/3] 加载验证集...")
    val_dataset = LIDCDataset(val_dir, augment=False)
    
    print("  [3/3] 加载测试集...")
    test_dataset = LIDCDataset(test_dir, augment=False)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    print(f"\n  ✓ 训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次/epoch")
    print(f"  ✓ 验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次/epoch")
    print(f"  ✓ 测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次/epoch")
    
    return train_loader, val_loader, test_loader


def get_sample_batch(dataloader: DataLoader, n_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取一个样本批次用于可视化"""
    images, masks = next(iter(dataloader))
    return images[:n_samples], masks[:n_samples]


if __name__ == "__main__":
    # 测试代码
    from configs.config import TRAIN_DIR, VAL_DIR, TEST_DIR
    
    print("测试 PyTorch 数据加载模块...")
    
    if os.path.exists(TRAIN_DIR):
        # 测试数据集
        dataset = LIDCDataset(TRAIN_DIR, augment=True)
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            img, mask = dataset[0]
            print(f"图像形状: {img.shape}, 范围: [{img.min():.3f}, {img.max():.3f}]")
            print(f"掩码形状: {mask.shape}, 唯一值: {torch.unique(mask).tolist()}")
        
        # 测试 DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_img, batch_mask = next(iter(loader))
        print(f"批次图像形状: {batch_img.shape}")
        print(f"批次掩码形状: {batch_mask.shape}")
    else:
        print(f"训练目录不存在: {TRAIN_DIR}")
        print("请先运行 split_dataset.py 划分数据")
