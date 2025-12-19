# LungAssist - 肺结节医学图像分割系统

<div align="center">

**基于深度学习的肺结节 CT 图像分割与识别系统**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 项目简介

LungAssist 是一个专业的医学图像分割系统，专注于肺结节的自动识别与分割。项目基于 LIDC-IDRI 数据集，采用深度学习技术（U-Net 架构）和传统图像处理方法相结合的方式，实现高精度的肺结节检测。

### 核心功能

- 🔬 **多种预处理方法**：CLAHE、中值滤波、高斯滤波、维纳滤波等
- 🎯 **ROI 提取**：分水岭算法、边缘检测等多种 ROI 定位策略
- 🧠 **深度学习分割**：基于 U-Net 的端到端语义分割
- 📊 **传统方法对比**：12 种传统图像处理方法的性能对比
- 🖥️ **可视化工具**：训练监控、结果可视化、交互式标注工具
- 🌐 **Web 应用**：基于 Flask 的 Web 端标注和预测界面

---

## 🗂️ 项目结构

```
LungAssist/
├── data/                    # 数据目录（需自行准备）
│   └── LIDC-IDRI-slices/   # LIDC-IDRI 数据集切片
│
├── notebooks/               # Jupyter Notebook 实验和分析
│   ├── tradition.ipynb                  # 传统方法实验
│   ├── tradition_abalation.ipynb        # 传统方法消融实验
│   └── lidc_unet_detection.ipynb        # U-Net 检测实验
│
├── src/                     # 核心源代码
│   ├── preprocessing/       # 数据预处理模块
│   │   └── contrast_methods.py          # 12种对比方法实现
│   │
│   ├── models/              # 模型定义
│   │   └── unet.py          # U-Net 模型（标准、小型、大型）
│   │
│   ├── training/            # 训练相关
│   │   ├── dataset.py       # PyTorch Dataset 和 DataLoader
│   │   ├── train.py         # 训练脚本
│   │   ├── losses.py        # 损失函数（Dice、BCE、Focal 等）
│   │   └── metrics.py       # 评估指标
│   │
│   ├── evaluation/          # 模型评估
│   │   └── evaluate.py      # 评估脚本
│   │
│   ├── prediction/          # 预测推理
│   │   └── predict.py       # 预测脚本
│   │
│   └── utils/               # 工具函数
│       └── utils.py         # 可视化、日志、检查点管理等
│
├── configs/                 # 配置文件
│   └── config.py            # 全局配置（路径、超参数等）
│
├── scripts/                 # 运行脚本
│   ├── split_dataset.py     # 数据集划分脚本
│   ├── train_clahe.sh       # CLAHE 训练脚本
│   ├── train_clahe_median.sh
│   └── ...                  # 其他训练脚本
│
├── outputs/                 # 训练输出（模型、日志、可视化）
│   └── [自动生成]
│
├── tools/                   # 额外工具
│   └── medical_image_gui.py # 医学图像标注 GUI 工具
│
├── web_app/                 # Web 应用
│   ├── app.py               # Flask 应用主文件
│   ├── templates/           # HTML 模板
│   └── requirements.txt     # Web 应用依赖
│
├── requirements.txt         # Python 依赖
├── .gitignore              # Git 忽略文件
└── README.md               # 项目说明文档（本文件）
```

---

## 🚀 快速开始

### 1. 环境准备

#### 系统要求

- Python 3.8+
- CUDA 11.7+ (推荐使用 GPU)
- 8GB+ RAM (16GB+ 推荐)
- 10GB+ 磁盘空间

#### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/Xiang-Chen1207/LungAssist.git
cd LungAssist

# 创建虚拟环境（推荐使用 conda）
conda create -n lungassist python=3.10
conda activate lungassist

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 版本（不推荐，速度较慢）
pip install torch torchvision

# 安装其他依赖
pip install -r requirements.txt
```

#### 验证 GPU

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"
```

---

### 2. 数据集准备

#### LIDC-IDRI 数据集

本项目使用 [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) 数据集。

**数据集结构示例：**

```
data/LIDC-IDRI-slices/
├── LIDC-IDRI-0001/
│   ├── images/
│   │   ├── slice_001.png
│   │   ├── slice_002.png
│   │   └── ...
│   ├── mask-0/
│   ├── mask-1/
│   ├── mask-2/
│   └── mask-3/
├── LIDC-IDRI-0002/
└── ...
```

#### 配置数据路径

编辑 `configs/config.py`：

```python
# 方法1：直接修改配置文件
DATA_ROOT = "/path/to/your/LIDC-IDRI-slices"

# 方法2：设置环境变量（推荐）
export LIDC_DATA_ROOT="/path/to/your/LIDC-IDRI-slices"
```

#### 划分数据集

将数据按 70% 训练集、20% 验证集、10% 测试集划分：

```bash
python scripts/split_dataset.py
```

---

### 3. 模型训练

#### 基础训练

```bash
python src/training/train.py
```

#### 自定义参数训练

```bash
python src/training/train.py \
    --gpu 0 \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --loss bce_dice \
    --model standard \
    --name my_experiment
```

**可用参数：**

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--gpu` | 指定 GPU ID | 自动选择 | 0, 1, 2... |
| `--epochs` | 训练轮数 | 100 | 任意正整数 |
| `--batch_size` | 批次大小 | 8 | 2, 4, 8, 16, 32 |
| `--lr` | 学习率 | 1e-4 | 1e-3 ~ 1e-6 |
| `--loss` | 损失函数 | bce_dice | dice, bce, bce_dice, focal, tversky |
| `--model` | 模型大小 | standard | small, standard, large |
| `--name` | 实验名称 | 自动生成 | 任意字符串 |
| `--resume` | 继续训练 | None | 检查点路径 |
| `--no_augment` | 禁用数据增强 | False | - |

#### 使用预设脚本

```bash
# CLAHE + 中值滤波训练
bash scripts/train_clahe_median.sh

# CLAHE + 平滑 + 边缘检测 + GHPF 训练
bash scripts/train_clahe_smooth_edge_ghpf.sh
```

---

### 4. 模型评估

#### 评估测试集

```bash
python src/evaluation/evaluate.py \
    outputs/experiment_name/checkpoints/best_model.pth \
    --test_dir data_split/test
```

#### 多阈值评估（寻找最佳阈值）

```bash
python src/evaluation/evaluate.py \
    outputs/experiment_name/checkpoints/best_model.pth \
    --multi_threshold
```

#### 可视化评估结果

```bash
python src/evaluation/evaluate.py \
    outputs/experiment_name/checkpoints/best_model.pth \
    --visualize \
    --num_samples 10
```

---

### 5. 模型预测

#### 单张图像预测

```bash
python src/prediction/predict.py \
    checkpoints/best_model.pth \
    input_image.png \
    --overlay \
    --output predictions/
```

#### 批量预测

```bash
python src/prediction/predict.py \
    checkpoints/best_model.pth \
    input_folder/ \
    --output predictions/ \
    --save_mask
```

---

### 6. Web 应用

启动 Web 端标注和预测界面：

```bash
cd web_app
pip install -r requirements.txt
python app.py
```

访问 `http://localhost:5000` 即可使用。

---

### 7. 传统方法对比

运行传统图像处理方法对比实验：

```bash
python src/preprocessing/contrast_methods.py
```

生成 12 种方法的性能对比报告（CSV 和可视化图表）。

---

## 🏗️ 模型架构

### U-Net 架构

```
输入 (B, 3, 128, 128)
    ↓
[编码器] Conv(64) → Conv(128) → Conv(256) → Conv(512)
    ↓         ↓           ↓            ↓
    ↓    跳跃连接 ←----←----←----←----←
    ↓
[桥接层] Conv(1024)
    ↓
[解码器] Conv(512) → Conv(256) → Conv(128) → Conv(64)
    ↓
输出 (B, 1, 128, 128) Sigmoid
```

### 模型变体

| 模型类型 | 参数量 | 显存需求 | 适用场景 |
|---------|--------|---------|---------|
| **Small** | ~7.8M | < 4GB | 资源受限环境 |
| **Standard** | ~31M | 4-8GB | 推荐使用 |
| **Large** | ~124M | > 12GB | 高精度需求 |

---

## 📊 性能指标

### 评估指标

- **Dice Coefficient**: 分割重叠度（主要指标）
- **IoU (Jaccard)**: 交并比
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: 精确率和召回率的调和平均
- **Pixel Accuracy**: 像素准确率

### 损失函数

| 损失函数 | 说明 | 适用场景 |
|---------|------|---------|
| `dice` | Dice Loss | 基础分割任务 |
| `bce` | Binary Cross-Entropy | 像素级分类 |
| `bce_dice` | BCE + Dice | **推荐**，综合效果好 |
| `focal` | Focal Loss | 类别不平衡 |
| `tversky` | Tversky Loss | 控制 FP/FN 权重 |

---

## 🛠️ 传统方法对比

项目实现了 12 种传统图像处理方法的组合：

### 预处理方法（3种）
1. CLAHE + 中值滤波
2. CLAHE + 高斯平滑
3. CLAHE + 维纳滤波

### ROI 提取（2种）
1. 分水岭算法
2. 边缘检测

### 分割策略（2种）
1. GHPF + Otsu
2. GHPF + 灰度重建 + Otsu

**组合结果：** 3 × 2 × 2 = 12 种方法

运行对比实验：

```bash
python src/preprocessing/contrast_methods.py
```

---

## 🖥️ 工具使用

### 医学图像标注工具

启动 GUI 标注工具：

```bash
python tools/medical_image_gui.py
```

**功能：**
- 图像浏览和标注
- 多种标注工具（画笔、橡皮擦、区域选择）
- 标注保存和导出

---

## 📁 数据集说明

### LIDC-IDRI 数据集

- **来源**: [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- **内容**: 1,018 例胸部 CT 扫描
- **标注**: 4 位放射科医生独立标注
- **许可**: 公开数据集

### 数据处理流程

1. **数据清洗**: 去除无效样本
2. **共识机制**: 多专家标注投票
3. **数据增强**: 旋转、翻转、亮度/对比度调整
4. **归一化**: 标准化到 [0, 1] 范围

---

## 🔧 常见问题

### Q: 显存不足怎么办？

**解决方案：**

1. 减小 batch_size：
   ```bash
   python src/training/train.py --batch_size 4
   ```

2. 使用小型模型：
   ```bash
   python src/training/train.py --model small
   ```

3. 减小图像尺寸（修改 `configs/config.py`）：
   ```python
   IMG_HEIGHT = 64
   IMG_WIDTH = 64
   ```

### Q: 如何从检查点继续训练？

```bash
python src/training/train.py --resume outputs/experiment/checkpoints/checkpoint_epoch_50.pth
```

### Q: 如何指定 GPU？

```bash
# 方法1：命令行参数
python src/training/train.py --gpu 1

# 方法2：环境变量
export CUDA_VISIBLE_DEVICES=1
python src/training/train.py
```

### Q: PyTorch 与 CUDA 版本对应关系？

| PyTorch 版本 | CUDA 版本 |
|-------------|----------|
| 2.0.x | 11.7, 11.8 |
| 2.1.x | 11.8, 12.1 |
| 2.2.x | 11.8, 12.1 |
| 2.3.x | 11.8, 12.1 |

---

## 📚 参考文献

1. **U-Net 原论文**:
   Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
   [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

2. **LIDC-IDRI 数据集**:
   Armato III, S. G., et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans. *Medical Physics*.

3. **Dice Loss**:
   Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV*.

---

## 🤝 贡献指南

欢迎贡献代码、提出问题或建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 📧 联系方式

**项目维护者**: Xiang Chen
**GitHub**: [@Xiang-Chen1207](https://github.com/Xiang-Chen1207)
**项目主页**: [https://github.com/Xiang-Chen1207/LungAssist](https://github.com/Xiang-Chen1207/LungAssist)

---

## 🙏 致谢

- 感谢 LIDC-IDRI 数据集的提供者
- 感谢 PyTorch 团队的优秀框架
- 感谢所有贡献者和用户的支持

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！⭐**

Made with ❤️ by Xiang Chen

</div>
