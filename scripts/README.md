# LIDC-IDRI U-Net 预处理消融实验脚本

本目录包含用于评估不同图像预处理方法对 U-Net 分割性能影响的训练和评估脚本。

## 预处理方法说明

| 方法名称 | 描述 | 对应原方法 |
|---------|------|-----------|
| `clahe` | 仅 CLAHE (对比度受限自适应直方图均衡) | - |
| `clahe_median` | CLAHE + 中值滤波 | - |
| `clahe_smooth` | CLAHE + 高斯平滑 | - |
| `clahe_median_edge_ghpf` | CLAHE + 中值 + 边缘检测 + GHPF | C3 |
| `clahe_smooth_edge_ghpf` | CLAHE + 平滑 + 边缘检测 + GHPF | C7 |

## 快速开始

### 运行所有实验

```bash
cd /home/chenx/code/medical_project/scripts
./run_all_experiments.sh
```

可以通过环境变量自定义参数:

```bash
GPU_ID=0 EPOCHS=50 BATCH_SIZE=16 ./run_all_experiments.sh
```

### 运行单个实验

```bash
# 训练 CLAHE 方法
./train_clahe.sh

# 训练 CLAHE + 中值滤波 方法
./train_clahe_median.sh

# 训练 CLAHE + 平滑 方法
./train_clahe_smooth.sh

# 训练 CLAHE + 中值 + 边缘检测 + GHPF 方法 (C3)
./train_clahe_median_edge_ghpf.sh

# 训练 CLAHE + 平滑 + 边缘检测 + GHPF 方法 (C7)
./train_clahe_smooth_edge_ghpf.sh
```

### 命令行参数

所有训练脚本支持以下参数:

```bash
./train_clahe.sh --gpu 0 --epochs 100 --batch_size 32 --lr 1e-4
```

### 评估所有模型

训练完成后，运行评估脚本:

```bash
./evaluate_all_models.sh
```

## 直接使用 Python 脚本

也可以直接使用 Python 脚本:

```bash
cd /home/chenx/code/medical_project/code/lidc_unet

# 训练
python train.py --preprocess clahe_median_edge_ghpf --name my_experiment --epochs 100

# 评估
python evaluate.py /path/to/best_model.pth --preprocess clahe_median_edge_ghpf
```

## 输出目录结构

所有结果将保存在 `/home/chenx/code/medical_project/outputs/` 目录下:

```
outputs/
├── unet_clahe_<timestamp>/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   ├── logs/
│   └── training_log.csv
├── unet_clahe_median_<timestamp>/
│   └── ...
└── ...
```

## 数据路径

- 数据划分目录: `/home/chenx/code/medical_project/data_split/`
  - 训练集: `data_split/train/`
  - 验证集: `data_split/val/`
  - 测试集: `data_split/test/`

## 预处理参数配置

预处理参数可以在 `configs/config.py` 中修改:

- CLAHE: `CLAHE_CLIP_LIMIT=2.0`, `CLAHE_TILE_GRID_SIZE=(8,8)`
- 中值滤波: `MEDIAN_KERNEL_SIZE=3`
- 高斯平滑: `GAUSSIAN_KERNEL_SIZE=(3,3)`, `GAUSSIAN_SIGMA=0`
- 边缘检测: `CANNY_LOW_THRESHOLD=50`, `CANNY_HIGH_THRESHOLD=150`
- GHPF: `GHPF_KERNEL_SIZE=(15,15)`, `GHPF_SIGMA=3`
