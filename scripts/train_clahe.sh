#!/bin/bash
# 训练脚本 - CLAHE 预处理方法
# 方法: 仅 CLAHE (对比度受限自适应直方图均衡)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code/lidc_unet"

echo "=============================================="
echo "LIDC-IDRI U-Net 训练 - CLAHE 预处理"
echo "=============================================="
echo "项目目录: $PROJECT_DIR"
echo "代码目录: $CODE_DIR"
echo ""

cd "$CODE_DIR"

python train.py \
    --preprocess clahe \
    --name unet_clahe \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --model standard \
    --loss bce_dice \
    "$@"

echo ""
echo "训练完成!"
echo "结果保存在: $PROJECT_DIR/outputs/unet_clahe*"
