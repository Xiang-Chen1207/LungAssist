# LIDC-IDRI U-Net 肺结节分割项目 (PyTorch 版本)

基于 U-Net 架构的肺结节 CT 图像分割系统，使用 LIDC-IDRI 数据集进行训练和评估。

## 📁 项目结构

```
lidc_unet_pytorch/
├── configs/
│   └── config.py          # 配置文件（路径、超参数等）
├── src/
│   ├── __init__.py
│   ├── dataset.py         # PyTorch Dataset 和 DataLoader
│   ├── model.py           # U-Net 模型定义
│   ├── losses.py          # 损失函数（Dice、IoU、Focal等）
│   ├── metrics.py         # 评估指标
│   └── utils.py           # 工具函数（可视化、日志等）
├── split_dataset.py       # 数据集划分脚本
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── predict.py             # 预测脚本
├── requirements.txt       # Python 依赖
└── README.md              # 说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建 conda 环境（推荐）
conda create -n lidc python=3.10
conda activate lidc

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 版本
pip install torch torchvision

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 验证 GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 3. 配置数据路径

编辑 `configs/config.py`，修改 `DATA_ROOT` 为你的数据路径：

```python
DATA_ROOT = "/home/chenx/code/medical_project/data/LIDC-IDRI-slices"
```

### 4. 划分数据集

将数据按 70% / 20% / 10% 划分为训练集、验证集、测试集：

```bash
python split_dataset.py
```

### 5. 训练模型

基本训练：

```bash
python train.py
```

指定 GPU 和自定义参数：

```bash
python train.py --gpu 0 --epochs 100 --batch_size 16 --lr 0.0001 --loss bce_dice --name my_experiment
```

可用参数：
- `--gpu`: 指定 GPU ID（默认自动选择）
- `--epochs`: 训练轮数（默认 100）
- `--batch_size`: 批次大小（默认 8）
- `--lr`: 学习率（默认 1e-4）
- `--loss`: 损失函数，可选 `dice`, `bce`, `bce_dice`, `focal`, `tversky`
- `--model`: 模型大小，可选 `small`, `standard`, `large`
- `--name`: 实验名称
- `--resume`: 从检查点继续训练
- `--no_augment`: 禁用数据增强

### 6. 评估模型

```bash
python evaluate.py outputs/xxx/checkpoints/best_model.pth --test_dir data_split/test
```

多阈值评估（找最佳阈值）：

```bash
python evaluate.py outputs/xxx/checkpoints/best_model.pth --multi_threshold
```

### 7. 预测新图像

单张图像：

```bash
python predict.py checkpoints/best_model.pth input_image.png --overlay
```

批量预测：

```bash
python predict.py checkpoints/best_model.pth input_folder/ --output predictions/
```

## 🏗️ 模型架构

```
输入 (B, 3, 128, 128)
    ↓
[编码器] Conv(64) → Conv(128) → Conv(256) → Conv(512)
    ↓                    ↓           ↓          ↓
    ↓              跳跃连接 ←----←----←----←----←
    ↓
[桥接层] Conv(1024)
    ↓
[解码器] Conv(512) → Conv(256) → Conv(128) → Conv(64)
    ↓
输出 (B, 1, 128, 128) sigmoid
```

### 模型变体

| 类型 | 参数量 | 说明 |
|------|--------|------|
| `small` | ~7.8M | 适用于显存 < 4GB |
| `standard` | ~31M | **默认**，推荐 |
| `large` | ~124M | 适用于显存 > 12GB |

## 📊 损失函数

| 损失函数 | 说明 | 适用场景 |
|---------|------|---------|
| `dice` | Dice Loss | 基础分割任务 |
| `bce` | Binary Cross-Entropy | 像素级分类 |
| `bce_dice` | BCE + Dice | **推荐**，综合效果好 |
| `focal` | Focal Loss | 类别不平衡 |
| `tversky` | Tversky Loss | 控制 FP/FN 权重 |

## 📈 评估指标

- **Dice Coefficient**: 分割重叠度，主要指标
- **IoU (Jaccard)**: 交并比
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: 精确率和召回率的调和平均
- **Pixel Accuracy**: 像素准确率

## 📝 配置说明

主要配置项（`configs/config.py`）：

```python
# 数据路径
DATA_ROOT = "/path/to/LIDC-IDRI-slices"

# 图像尺寸
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 训练参数
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4

# DataLoader
NUM_WORKERS = 4  # 数据加载进程数
```

## 🔧 常见问题

### Q: 如何指定 GPU？

```bash
# 方法1：命令行参数
python train.py --gpu 1

# 方法2：环境变量
export CUDA_VISIBLE_DEVICES=1
python train.py
```

### Q: 显存不足怎么办？

1. 减小 `batch_size`（如 4 或 2）
2. 使用 `--model small` 小型模型
3. 减小图像尺寸（修改 config.py 中的 IMG_HEIGHT/IMG_WIDTH）

### Q: 如何从检查点继续训练？

```bash
python train.py --resume outputs/xxx/checkpoints/checkpoint_epoch_50.pth
```

### Q: PyTorch 与 CUDA 版本对应

| PyTorch | CUDA |
|---------|------|
| 2.0.x | 11.7, 11.8 |
| 2.1.x | 11.8, 12.1 |
| 2.2.x | 11.8, 12.1 |

## 🆚 PyTorch vs TensorFlow

| 方面 | PyTorch | TensorFlow |
|------|---------|------------|
| GPU 配置 | 简单，通常开箱即用 | 需要精确版本匹配 |
| 调试 | 动态图，易于调试 | 静态图（2.x 改进） |
| 社区 | 研究领域主流 | 工业部署更成熟 |

## 📚 参考

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [LIDC-IDRI Dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

MIT License
