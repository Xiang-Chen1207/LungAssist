"""
数据集模块 (PyTorch 版本)
提供 LIDC-IDRI 数据的加载、预处理和数据增强功能
支持多种图像预处理方法的消融实验
"""
import os
import numpy as np
import cv2
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
    USE_AUGMENTATION, AUGMENTATION_CONFIG, RANDOM_SEED,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    MEDIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA,
    CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD,
    GHPF_KERNEL_SIZE, GHPF_SIGMA
)


class ImagePreprocessor:
    """
    图像预处理器
    支持多种预处理方法:
    - none: 不进行预处理
    - clahe: 仅 CLAHE
    - clahe_median: CLAHE + 中值滤波
    - clahe_smooth: CLAHE + 高斯平滑
    - clahe_median_edge_ghpf: CLAHE + 中值 + 边缘检测 + GHPF
    - clahe_smooth_edge_ghpf: CLAHE + 平滑 + 边缘检测 + GHPF
    """

    VALID_METHODS = [
        'none', 'clahe', 'clahe_median', 'clahe_smooth',
        'clahe_median_edge_ghpf', 'clahe_smooth_edge_ghpf'
    ]

    def __init__(self, method: str = 'none'):
        """
        初始化预处理器

        参数:
            method: 预处理方法名称
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"无效的预处理方法: {method}. 可选: {self.VALID_METHODS}")

        self.method = method

        # 创建 CLAHE 对象
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """应用 CLAHE (对比度受限自适应直方图均衡)"""
        if len(image.shape) == 3:
            # 如果是RGB图像，转换为灰度
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        enhanced = self.clahe.apply(gray)
        return enhanced

    def apply_median(self, image: np.ndarray) -> np.ndarray:
        """应用中值滤波"""
        return cv2.medianBlur(image, MEDIAN_KERNEL_SIZE)

    def apply_gaussian(self, image: np.ndarray) -> np.ndarray:
        """应用高斯平滑"""
        return cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

    def apply_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        应用边缘检测流水线提取ROI
        返回ROI掩码
        """
        # Canny边缘检测
        edges = cv2.Canny(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

        # 形态学闭运算连接边缘
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 填充轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = np.zeros_like(image, dtype=np.uint8)

        # 只保留较大的区域
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                cv2.drawContours(roi_mask, [contour], -1, 1, -1)

        return roi_mask

    def apply_ghpf(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        应用高斯高通滤波器 (GHPF)

        参数:
            image: 输入图像
            roi_mask: ROI掩码

        返回:
            处理后的图像
        """
        # 高通滤波器 (Gaussian High-Pass Filter)
        lowpass = cv2.GaussianBlur(image, GHPF_KERNEL_SIZE, GHPF_SIGMA)
        highpass = cv2.subtract(image, lowpass)
        highpass = cv2.add(highpass, 127)  # 添加偏移避免负值

        # 应用ROI掩码 - 保留ROI区域，其他区域设为原值
        result = image.copy()
        result[roi_mask > 0] = highpass[roi_mask > 0]

        return result

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        根据设定的方法预处理图像

        参数:
            image: 输入图像 (RGB 或 灰度)

        返回:
            预处理后的图像 (灰度)
        """
        if self.method == 'none':
            # 不进行预处理，只转换为灰度
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image

        elif self.method == 'clahe':
            # 仅 CLAHE
            return self.apply_clahe(image)

        elif self.method == 'clahe_median':
            # CLAHE + 中值滤波
            enhanced = self.apply_clahe(image)
            return self.apply_median(enhanced)

        elif self.method == 'clahe_smooth':
            # CLAHE + 高斯平滑
            enhanced = self.apply_clahe(image)
            return self.apply_gaussian(enhanced)

        elif self.method == 'clahe_median_edge_ghpf':
            # CLAHE + 中值 + 边缘检测 + GHPF
            enhanced = self.apply_clahe(image)
            denoised = self.apply_median(enhanced)
            roi_mask = self.apply_edge_detection(denoised)
            return self.apply_ghpf(denoised, roi_mask)

        elif self.method == 'clahe_smooth_edge_ghpf':
            # CLAHE + 平滑 + 边缘检测 + GHPF
            enhanced = self.apply_clahe(image)
            denoised = self.apply_gaussian(enhanced)
            roi_mask = self.apply_edge_detection(denoised)
            return self.apply_ghpf(denoised, roi_mask)

        else:
            raise ValueError(f"未知的预处理方法: {self.method}")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """使预处理器可调用"""
        return self.preprocess(image)


class LIDCDataset(Dataset):
    """
    LIDC-IDRI 数据集类 (PyTorch Dataset)
    负责加载和预处理肺结节CT图像及其分割掩码
    支持多种图像预处理方法
    """

    def __init__(self, data_dir: str,
                 img_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
                 augment: bool = False,
                 transform: Optional[Callable] = None,
                 preprocess_method: str = 'none'):
        """
        初始化数据集

        参数:
            data_dir: 数据目录路径（应包含病例子文件夹）
            img_size: 目标图像尺寸 (height, width)
            augment: 是否进行数据增强
            transform: 自定义变换函数
            preprocess_method: 预处理方法 ('none', 'clahe', 'clahe_median',
                              'clahe_smooth', 'clahe_median_edge_ghpf',
                              'clahe_smooth_edge_ghpf')
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.transform = transform
        self.preprocess_method = preprocess_method

        self.image_paths = []
        self.mask_paths = []

        self._scan_data_directory()

        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(preprocess_method)

        # 基础变换
        self.resize = T.Resize(img_size)
        self.to_tensor = T.ToTensor()

        print(f"  ✓ 预处理方法: {preprocess_method}")

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
        # 加载图像为numpy数组
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 调整大小
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        # 应用预处理 (得到灰度图)
        processed_image = self.preprocessor(image)

        # 转换为PIL Image用于数据增强
        processed_image_pil = Image.fromarray(processed_image)
        mask_pil = Image.fromarray(mask)

        # 数据增强（图像和掩码需要同步变换）
        if self.augment:
            processed_image_pil, mask_pil = self._augment(processed_image_pil, mask_pil)

        # 自定义变换
        if self.transform:
            processed_image_pil = self.transform(processed_image_pil)

        # 转换为张量
        # 灰度图需要转换为3通道以适配模型
        processed_image_np = np.array(processed_image_pil)
        if len(processed_image_np.shape) == 2:
            # 复制灰度通道到3个通道
            processed_image_np = np.stack([processed_image_np] * 3, axis=-1)

        processed_image_pil = Image.fromarray(processed_image_np)

        image_tensor = self.to_tensor(processed_image_pil)  # [C, H, W], 范围 [0, 1]
        mask_tensor = self.to_tensor(mask_pil)  # [1, H, W], 范围 [0, 1]

        # 二值化掩码
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

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
                       augment_train: bool = USE_AUGMENTATION,
                       preprocess_method: str = 'none') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试 DataLoader

    参数:
        train_dir, val_dir, test_dir: 各数据集的目录路径
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        augment_train: 是否对训练数据进行增强
        preprocess_method: 预处理方法

    返回:
        (train_loader, val_loader, test_loader)
    """
    print(f"\n创建 DataLoader (预处理方法: {preprocess_method})...")

    # 创建数据集
    print("  [1/3] 加载训练集...")
    train_dataset = LIDCDataset(train_dir, augment=augment_train,
                                 preprocess_method=preprocess_method)

    print("  [2/3] 加载验证集...")
    val_dataset = LIDCDataset(val_dir, augment=False,
                               preprocess_method=preprocess_method)

    print("  [3/3] 加载测试集...")
    test_dataset = LIDCDataset(test_dir, augment=False,
                                preprocess_method=preprocess_method)

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
    print("=" * 60)

    # 测试所有预处理方法
    methods = ['none', 'clahe', 'clahe_median', 'clahe_smooth',
               'clahe_median_edge_ghpf', 'clahe_smooth_edge_ghpf']

    if os.path.exists(TRAIN_DIR):
        for method in methods:
            print(f"\n测试预处理方法: {method}")
            print("-" * 40)

            # 测试数据集
            dataset = LIDCDataset(TRAIN_DIR, augment=False, preprocess_method=method)
            print(f"数据集大小: {len(dataset)}")

            if len(dataset) > 0:
                img, mask = dataset[0]
                print(f"图像形状: {img.shape}, 范围: [{img.min():.3f}, {img.max():.3f}]")
                print(f"掩码形状: {mask.shape}, 唯一值: {torch.unique(mask).tolist()}")
    else:
        print(f"训练目录不存在: {TRAIN_DIR}")
        print("请先运行 split_dataset.py 划分数据")
