"""
配置文件 - 包含所有可调参数 (PyTorch 版本)
支持图像预处理方法的消融实验
"""
import os
import torch

# ==================== 路径配置 ====================
# 数据根目录 - 请根据你的实际路径修改
DATA_ROOT = "/home/chenx/code/medical_project/data/LIDC-IDRI-slices"

# 项目根目录 - 使用 medical_project 作为根目录
PROJECT_ROOT = "/home/chenx/code/medical_project"

# 划分后的数据目录
SPLIT_DATA_DIR = os.path.join(PROJECT_ROOT, "data_split")
TRAIN_DIR = os.path.join(SPLIT_DATA_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DATA_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DATA_DIR, "test")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ==================== 图像预处理配置 ====================
# 预处理方法选项:
# 'none'          - 不进行预处理
# 'clahe'         - 仅 CLAHE
# 'clahe_median'  - CLAHE + 中值滤波
# 'clahe_smooth'  - CLAHE + 高斯平滑
# 'clahe_median_edge_ghpf' - CLAHE + 中值 + 边缘检测 + GHPF
# 'clahe_smooth_edge_ghpf' - CLAHE + 平滑 + 边缘检测 + GHPF
PREPROCESS_METHOD = 'none'

# CLAHE 参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# 中值滤波核大小
MEDIAN_KERNEL_SIZE = 3

# 高斯平滑参数
GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA = 0

# 边缘检测 (Canny) 参数
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# GHPF (高斯高通滤波) 参数
GHPF_KERNEL_SIZE = (15, 15)
GHPF_SIGMA = 3

# ==================== 数据配置 ====================
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3        # RGB图像
MASK_CHANNELS = 1       # 二值mask

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 随机种子（保证可复现性）
RANDOM_SEED = 42

# ==================== 模型配置 ====================
# U-Net编码器的滤波器数量
ENCODER_FILTERS = [64, 128, 256, 512]
BRIDGE_FILTERS = 1024

# Dropout率
DROPOUT_RATE = 0.3

# ==================== 训练配置 ====================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# 早停参数
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# 损失函数平滑因子
SMOOTH = 1.0

# DataLoader 参数
NUM_WORKERS = 4
PIN_MEMORY = True

# ==================== 数据增强配置 ====================
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': 0.1,
    'contrast_range': 0.1,
}

# ==================== 设备配置 ====================
def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("✗ 未检测到 GPU，使用 CPU")
    return device

DEVICE = None  # 在运行时设置

# ==================== 辅助函数 ====================
def create_directories():
    """创建所有必要的目录"""
    dirs = [
        SPLIT_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
        OUTPUT_DIR, CHECKPOINT_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ 所有目录已创建/确认存在")

def print_config(preprocess_method=None):
    """打印当前配置"""
    print("=" * 50)
    print("当前配置:")
    print("=" * 50)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"图像尺寸: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
    print(f"数据划分: 训练{TRAIN_RATIO*100}% / 验证{VAL_RATIO*100}% / 测试{TEST_RATIO*100}%")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"预处理方法: {preprocess_method if preprocess_method else PREPROCESS_METHOD}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print("=" * 50)

def set_seed(seed=RANDOM_SEED):
    """设置随机种子以确保可复现性"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    print_config()
    create_directories()
    device = get_device()
