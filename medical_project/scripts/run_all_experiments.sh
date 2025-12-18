#!/bin/bash
# 运行所有5种预处理方法的训练实验
#
# 实验内容:
# 1. CLAHE - 仅对比度增强
# 2. CLAHE + 中值滤波 - 对比度增强 + 去噪
# 3. CLAHE + 高斯平滑 - 对比度增强 + 平滑
# 4. CLAHE + 中值 + 边缘检测 + GHPF (C3) - 完整流水线(中值)
# 5. CLAHE + 平滑 + 边缘检测 + GHPF (C7) - 完整流水线(平滑)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "LIDC-IDRI U-Net 预处理方法消融实验"
echo "=============================================="
echo "项目目录: $PROJECT_DIR"
echo "开始时间: $(date)"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
fi

# 定义预处理方法
METHODS=(
    "clahe"
    "clahe_median"
    "clahe_smooth"
    "clahe_median_edge_ghpf"
    "clahe_smooth_edge_ghpf"
)

# 训练参数 (可以通过命令行覆盖)
GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "训练参数:"
echo "  GPU ID: $GPU_ID"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# 运行每个实验
for i in "${!METHODS[@]}"; do
    METHOD="${METHODS[$i]}"
    echo "=============================================="
    echo "[$((i+1))/${#METHODS[@]}] 训练: $METHOD"
    echo "=============================================="

    bash "$SCRIPT_DIR/train_${METHOD}.sh" \
        --gpu $GPU_ID \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE

    echo ""
    echo "完成: $METHOD"
    echo "时间: $(date)"
    echo ""
done

echo "=============================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "=============================================="
echo ""
echo "结果保存在:"
for METHOD in "${METHODS[@]}"; do
    echo "  - $PROJECT_DIR/outputs/unet_${METHOD}*"
done
