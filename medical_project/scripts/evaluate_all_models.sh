#!/bin/bash
# 评估所有训练好的模型在测试集上的表现
#
# 使用方法:
#   ./evaluate_all_models.sh
#   或指定GPU: GPU_ID=0 ./evaluate_all_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code/lidc_unet"
OUTPUT_DIR="$PROJECT_DIR/outputs"

echo "=============================================="
echo "LIDC-IDRI U-Net 模型评估"
echo "=============================================="
echo "项目目录: $PROJECT_DIR"
echo ""

cd "$CODE_DIR"

# 定义预处理方法
METHODS=(
    "clahe"
    "clahe_median"
    "clahe_smooth"
    "clahe_median_edge_ghpf"
    "clahe_smooth_edge_ghpf"
)

# GPU 设置
GPU_ID=${GPU_ID:-0}

# 创建评估结果汇总文件
SUMMARY_FILE="$OUTPUT_DIR/evaluation_summary.csv"
echo "Method,Dice,Dice_std,IoU,IoU_std,Precision,Recall,F1" > "$SUMMARY_FILE"

echo "开始评估..."
echo ""

for METHOD in "${METHODS[@]}"; do
    echo "=============================================="
    echo "评估: $METHOD"
    echo "=============================================="

    # 查找最佳模型
    MODEL_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "unet_${METHOD}*" | sort | tail -1)

    if [ -z "$MODEL_DIR" ]; then
        echo "警告: 未找到 $METHOD 的训练结果目录"
        continue
    fi

    MODEL_PATH="$MODEL_DIR/checkpoints/best_model.pth"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "警告: 未找到模型文件 $MODEL_PATH"
        continue
    fi

    EVAL_OUTPUT_DIR="$MODEL_DIR/evaluation"

    echo "模型路径: $MODEL_PATH"
    echo "输出目录: $EVAL_OUTPUT_DIR"

    python evaluate.py "$MODEL_PATH" \
        --preprocess "$METHOD" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        --gpu $GPU_ID \
        --n_vis 10

    echo ""
    echo "完成: $METHOD"
    echo ""
done

echo "=============================================="
echo "所有评估完成!"
echo "=============================================="
echo ""
echo "评估结果保存在各自的实验目录中"
echo "汇总文件: $SUMMARY_FILE"
