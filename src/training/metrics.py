"""
评估指标模块 (PyTorch 版本)
实现用于分割任务的各种评估指标
"""
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import defaultdict

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SMOOTH


class MetricTracker:
    """
    指标追踪器
    用于在训练/验证过程中累计和计算平均指标
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def get_average(self) -> Dict[str, float]:
        """获取所有指标的平均值"""
        return {key: np.mean(values) for key, values in self.metrics.items()}
    
    def get_std(self) -> Dict[str, float]:
        """获取所有指标的标准差"""
        return {key: np.std(values) for key, values in self.metrics.items()}


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                     threshold: float = 0.5, smooth: float = SMOOTH) -> torch.Tensor:
    """
    计算 Dice 系数
    
    参数:
        pred: 预测值 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
        threshold: 二值化阈值
        smooth: 平滑因子
    """
    pred = (pred > threshold).float()
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    return (2.0 * intersection + smooth) / (union + smooth)


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = SMOOTH) -> torch.Tensor:
    """
    计算 IoU / Jaccard Index
    """
    pred = (pred > threshold).float()
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def precision(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = SMOOTH) -> torch.Tensor:
    """
    计算精确率 Precision = TP / (TP + FP)
    """
    pred = (pred > threshold).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    
    return (tp + smooth) / (tp + fp + smooth)


def recall(pred: torch.Tensor, target: torch.Tensor,
           threshold: float = 0.5, smooth: float = SMOOTH) -> torch.Tensor:
    """
    计算召回率 Recall = TP / (TP + FN)
    """
    pred = (pred > threshold).float()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5, smooth: float = SMOOTH) -> torch.Tensor:
    """
    计算特异性 Specificity = TN / (TN + FP)
    """
    pred = (pred > threshold).float()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    return (tn + smooth) / (tn + fp + smooth)


def f1_score(pred: torch.Tensor, target: torch.Tensor,
             threshold: float = 0.5) -> torch.Tensor:
    """
    计算 F1 分数 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision(pred, target, threshold)
    rec = recall(pred, target, threshold)
    
    return 2 * (prec * rec) / (prec + rec + 1e-7)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor,
                   threshold: float = 0.5) -> torch.Tensor:
    """
    计算像素准确率
    """
    pred = (pred > threshold).float()
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, float]:
    """
    计算所有评估指标
    
    参数:
        pred: 预测值 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
        threshold: 二值化阈值
    
    返回:
        包含所有指标的字典
    """
    with torch.no_grad():
        metrics = {
            'dice': dice_coefficient(pred, target, threshold).item(),
            'iou': iou_score(pred, target, threshold).item(),
            'precision': precision(pred, target, threshold).item(),
            'recall': recall(pred, target, threshold).item(),
            'specificity': specificity(pred, target, threshold).item(),
            'f1': f1_score(pred, target, threshold).item(),
            'pixel_acc': pixel_accuracy(pred, target, threshold).item(),
        }
    
    return metrics


def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor,
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    逐样本计算指标，然后取平均
    """
    batch_size = pred.shape[0]
    metrics_list = []
    
    for i in range(batch_size):
        sample_metrics = compute_all_metrics(
            pred[i:i+1], target[i:i+1], threshold
        )
        metrics_list.append(sample_metrics)
    
    # 计算平均
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    return avg_metrics


def evaluate_model(model, dataloader, device, threshold: float = 0.5) -> Dict[str, float]:
    """
    在数据集上评估模型
    
    参数:
        model: PyTorch 模型
        dataloader: 数据加载器
        device: 计算设备
        threshold: 二值化阈值
    
    返回:
        评估指标字典
    """
    model.eval()
    tracker = MetricTracker()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # 预测
            outputs = model(images)
            
            # 计算指标
            batch_metrics = compute_all_metrics(outputs, masks, threshold)
            tracker.update(**batch_metrics)
    
    return tracker.get_average()


def print_metrics(metrics: Dict[str, float], title: str = "评估结果"):
    """格式化打印指标"""
    print(f"\n{title}")
    print("=" * 50)
    print(f"{'指标':<15} {'值':>10}")
    print("-" * 50)
    
    for key in ['dice', 'iou', 'precision', 'recall', 'specificity', 'f1', 'pixel_acc']:
        if key in metrics:
            value = metrics[key]
            std = metrics.get(f'{key}_std', None)
            if std:
                print(f"{key:<15} {value:>10.4f} ± {std:.4f}")
            else:
                print(f"{key:<15} {value:>10.4f}")
    
    print("=" * 50)


def confusion_matrix_stats(pred: torch.Tensor, target: torch.Tensor,
                           threshold: float = 0.5) -> Dict[str, int]:
    """
    计算混淆矩阵统计
    """
    pred = (pred > threshold).float()
    
    tp = (pred * target).sum().int().item()
    tn = ((1 - pred) * (1 - target)).sum().int().item()
    fp = (pred * (1 - target)).sum().int().item()
    fn = ((1 - pred) * target).sum().int().item()
    
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


if __name__ == "__main__":
    print("=" * 60)
    print("评估指标测试 (PyTorch)")
    print("=" * 60)
    
    # 创建测试数据
    torch.manual_seed(42)
    
    # 模拟一个批次
    pred = torch.rand(4, 1, 128, 128)
    target = (torch.rand(4, 1, 128, 128) > 0.7).float()  # 约30%正类
    
    print(f"\n预测形状: {pred.shape}")
    print(f"目标形状: {target.shape}")
    print(f"目标正类比例: {target.mean():.2%}")
    
    # 计算所有指标
    metrics = compute_all_metrics(pred, target)
    print_metrics(metrics)
    
    # 混淆矩阵
    cm = confusion_matrix_stats(pred, target)
    print(f"\n混淆矩阵: {cm}")
