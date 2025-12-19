#!/bin/bash
# 训练脚本 - CLAHE + 高斯平滑 预处理方法
# 方法: CLAHE + 高斯平滑

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code/lidc_unet"

echo "=============================================="
echo "LIDC-IDRI U-Net 训练 - CLAHE + 高斯平滑"
echo "=============================================="
echo "项目目录: $PROJECT_DIR"
echo "代码目录: $CODE_DIR"
echo ""

cd "$CODE_DIR"

python train.py \
    --preprocess clahe_smooth \
    --name unet_clahe_smooth \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --model standard \
    --loss bce_dice \
    "$@"

echo ""
echo "训练完成!"
echo "结果保存在: $PROJECT_DIR/outputs/unet_clahe_smooth*"
