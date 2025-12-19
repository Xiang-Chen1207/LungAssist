#!/bin/bash
# 训练脚本 - CLAHE + 中值 + 边缘检测 + GHPF 预处理方法 (对应 C3)
# 方法: CLAHE + 中值滤波 + 边缘检测 + 高斯高通滤波

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code/lidc_unet"

echo "=============================================="
echo "LIDC-IDRI U-Net 训练 - CLAHE + 中值 + 边缘检测 + GHPF (C3)"
echo "=============================================="
echo "项目目录: $PROJECT_DIR"
echo "代码目录: $CODE_DIR"
echo ""

cd "$CODE_DIR"

python train.py \
    --preprocess clahe_median_edge_ghpf \
    --name unet_clahe_median_edge_ghpf \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --model standard \
    --loss bce_dice \
    "$@"

echo ""
echo "训练完成!"
echo "结果保存在: $PROJECT_DIR/outputs/unet_clahe_median_edge_ghpf*"
