"""
评估脚本 (PyTorch 版本)
在测试集上评估训练好的模型
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    TEST_DIR, OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, NUM_WORKERS
)
from src.dataset import LIDCDataset, create_dataloaders
from src.model import UNet, build_unet
from src.losses import dice_coef, iou_score
from src.metrics import compute_all_metrics, evaluate_model, print_metrics, MetricTracker
from src.utils import (
    visualize_predictions, save_metrics_to_json, print_gpu_info, load_checkpoint
)


def load_model(model_path: str, device: torch.device, model_type: str = 'standard'):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型文件路径
        device: 计算设备
        model_type: 模型类型
    
    返回:
        加载的模型
    """
    # 创建模型
    model = build_unet(model_type)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型已加载: {model_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: Dice={checkpoint['metrics'].get('dice', 'N/A'):.4f}")
    else:
        # 直接是 state_dict
        model.load_state_dict(checkpoint)
        print(f"✓ 模型权重已加载: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate(model_path: str,
             test_dir: str = TEST_DIR,
             batch_size: int = BATCH_SIZE,
             output_dir: str = None,
             n_visualize: int = 10,
             save_predictions: bool = True,
             model_type: str = 'standard',
             threshold: float = 0.5):
    """
    评估模型
    
    参数:
        model_path: 模型文件路径
        test_dir: 测试数据目录
        batch_size: 批次大小
        output_dir: 输出目录
        n_visualize: 可视化样本数
        save_predictions: 是否保存预测结果
        model_type: 模型类型
        threshold: 二值化阈值
    """
    print("\n" + "=" * 60)
    print("LIDC-IDRI U-Net 肺结节分割 - 评估 (PyTorch)")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_gpu_info()
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 加载模型 ====================
    print("\n[1/4] 加载模型...")
    model = load_model(model_path, device, model_type)
    
    # ==================== 加载测试数据 ====================
    print("\n[2/4] 加载测试数据...")
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试目录不存在: {test_dir}")
    
    test_dataset = LIDCDataset(test_dir, augment=False)
    
    if len(test_dataset) == 0:
        raise ValueError("测试集为空，请检查数据目录")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"测试样本数: {len(test_dataset)}")
    
    # ==================== 模型预测 ====================
    print("\n[3/4] 模型预测...")
    
    all_images = []
    all_masks = []
    all_preds = []
    tracker = MetricTracker()
    
    for images, masks in tqdm(test_loader, desc="评估中"):
        images = images.to(device)
        masks = masks.to(device)
        
        # 预测
        outputs = model(images)
        
        # 计算指标
        batch_metrics = compute_all_metrics(outputs, masks, threshold)
        tracker.update(**batch_metrics)
        
        # 保存用于可视化
        all_images.append(images.cpu())
        all_masks.append(masks.cpu())
        all_preds.append(outputs.cpu())
    
    # 合并所有结果
    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    print(f"预测形状: {all_preds.shape}")
    print(f"预测范围: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    
    # ==================== 计算指标 ====================
    print("\n[4/4] 计算评估指标...")
    
    metrics = tracker.get_average()
    metrics_std = tracker.get_std()
    
    # 合并均值和标准差
    full_metrics = {}
    for key in metrics:
        full_metrics[key] = metrics[key]
        full_metrics[f'{key}_std'] = metrics_std[key]
    
    print_metrics(full_metrics, "测试集评估结果")
    
    # ==================== 保存结果 ====================
    print("\n保存评估结果...")
    
    # 保存指标
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    save_metrics_to_json(full_metrics, metrics_path)
    
    # 可视化预测结果
    if n_visualize > 0:
        print(f"\n生成可视化 ({n_visualize} 个样本)...")
        
        # 随机选择样本
        indices = torch.randperm(len(all_images))[:n_visualize]
        
        vis_path = os.path.join(output_dir, 'predictions_visualization.png')
        visualize_predictions(
            all_images[indices],
            all_masks[indices],
            all_preds[indices],
            n_samples=len(indices),
            save_path=vis_path,
            show=False
        )
    
    # 保存预测结果
    if save_predictions:
        predictions_path = os.path.join(output_dir, 'predictions.npz')
        np.savez_compressed(
            predictions_path,
            images=all_images.numpy(),
            true_masks=all_masks.numpy(),
            pred_masks=all_preds.numpy(),
            pred_binary=(all_preds > threshold).numpy()
        )
        print(f"预测结果已保存到: {predictions_path}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("评估总结:")
    print("=" * 60)
    print(f"  测试样本数: {len(test_dataset)}")
    print(f"  Dice 系数:  {metrics['dice']:.4f} ± {metrics_std['dice']:.4f}")
    print(f"  IoU 分数:   {metrics['iou']:.4f} ± {metrics_std['iou']:.4f}")
    print(f"  精确率:     {metrics['precision']:.4f}")
    print(f"  召回率:     {metrics['recall']:.4f}")
    print(f"  F1 分数:    {metrics['f1']:.4f}")
    print(f"\n  结果保存到: {output_dir}")
    print("=" * 60)
    
    return full_metrics


def evaluate_multiple_thresholds(model_path: str, test_dir: str,
                                  thresholds: list = None,
                                  model_type: str = 'standard'):
    """
    在多个阈值下评估模型，找到最佳阈值
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\n多阈值评估...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = load_model(model_path, device, model_type)
    
    # 加载数据
    test_dataset = LIDCDataset(test_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # 收集所有预测
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_masks.append(masks)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # 在不同阈值下评估
    results = []
    
    print(f"\n{'阈值':<10} {'Dice':<10} {'IoU':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 50)
    
    for thresh in thresholds:
        metrics = compute_all_metrics(all_preds, all_masks, thresh)
        results.append({'threshold': thresh, **metrics})
        
        print(f"{thresh:<10.2f} {metrics['dice']:<10.4f} {metrics['iou']:<10.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
    
    # 找到最佳阈值
    best_result = max(results, key=lambda x: x['dice'])
    print(f"\n最佳阈值: {best_result['threshold']} (Dice = {best_result['dice']:.4f})")
    
    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="LIDC-IDRI U-Net 肺结节分割评估 (PyTorch)"
    )
    
    parser.add_argument('model_path', type=str,
                        help='模型文件路径 (.pth)')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR,
                        help='测试数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['small', 'standard', 'large'],
                        help='模型类型')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--n_vis', type=int, default=10,
                        help='可视化样本数')
    parser.add_argument('--no_save_pred', action='store_true',
                        help='不保存预测结果')
    parser.add_argument('--multi_threshold', action='store_true',
                        help='进行多阈值评估')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定使用的 GPU ID')
    
    args = parser.parse_args()
    
    # 设置 GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    if args.multi_threshold:
        evaluate_multiple_thresholds(
            args.model_path, args.test_dir, model_type=args.model_type
        )
    else:
        evaluate(
            model_path=args.model_path,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            n_visualize=args.n_vis,
            save_predictions=not args.no_save_pred,
            model_type=args.model_type,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
