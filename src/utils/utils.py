"""
工具函数模块 (PyTorch 版本)
提供可视化、日志记录等辅助功能
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import OUTPUT_DIR


def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None,
                          show: bool = True):
    """
    绘制训练历史曲线
    
    参数:
        history: 包含 'train_loss', 'val_loss', 'train_dice', 'val_dice' 等的字典
        save_path: 保存路径（可选）
        show: 是否显示图像
    """
    # 确定要绘制的指标
    train_metrics = [k for k in history.keys() if k.startswith('train_')]
    
    n_metrics = len(train_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, train_key in zip(axes, train_metrics):
        metric_name = train_key.replace('train_', '')
        val_key = f'val_{metric_name}'
        
        # 训练曲线
        ax.plot(history[train_key], label=f'Train', linewidth=2)
        
        # 验证曲线（如果存在）
        if val_key in history:
            ax.plot(history[val_key], label=f'Val', linewidth=2, linestyle='--')
        
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_predictions(images: torch.Tensor, true_masks: torch.Tensor,
                          pred_masks: torch.Tensor, 
                          n_samples: int = 5,
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    可视化分割预测结果
    
    参数:
        images: 原始图像 (B, C, H, W)
        true_masks: 真实掩码 (B, 1, H, W)
        pred_masks: 预测掩码 (B, 1, H, W)
        n_samples: 显示样本数
        save_path: 保存路径
        show: 是否显示
    """
    # 转换为 numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(true_masks, torch.Tensor):
        true_masks = true_masks.cpu().numpy()
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    
    n_samples = min(n_samples, len(images))
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # 原始图像 (C, H, W) -> (H, W, C)
        img = images[i].transpose(1, 2, 0)
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        
        ax = axes[i, 0]
        ax.imshow(img)
        ax.set_title('Original Image')
        ax.axis('off')
        
        # 真实掩码
        ax = axes[i, 1]
        ax.imshow(true_masks[i, 0], cmap='gray')
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        # 预测掩码
        ax = axes[i, 2]
        ax.imshow(pred_masks[i, 0], cmap='gray')
        ax.set_title('Prediction')
        ax.axis('off')
        
        # 叠加显示
        ax = axes[i, 3]
        overlay = create_overlay(img, true_masks[i, 0], pred_masks[i, 0])
        ax.imshow(overlay)
        ax.set_title('Overlay (Green=GT, Red=Pred)')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"预测可视化已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_overlay(image: np.ndarray, true_mask: np.ndarray, 
                   pred_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    创建图像与掩码的叠加显示
    """
    # 确保图像在 [0, 1] 范围
    if image.max() > 1:
        image = image / 255.0
    
    # 确保是 3 通道
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # 二值化掩码
    true_mask = (true_mask > 0.5).astype(np.float32)
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    
    # 创建彩色叠加
    overlay = image.copy()
    
    # 真实掩码 -> 绿色
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + true_mask * alpha, 0, 1)
    
    # 预测掩码 -> 红色
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_mask * alpha, 0, 1)
    
    return overlay


def plot_sample_batch(images: torch.Tensor, masks: torch.Tensor,
                      n_samples: int = 8, title: str = "Sample Batch"):
    """
    绘制数据批次的样本
    """
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols * 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    
    for i in range(n_samples):
        row = (i // n_cols) * 2
        col = i % n_cols
        
        # 图像 (C, H, W) -> (H, W, C)
        img = images[i].transpose(1, 2, 0)
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Image {i+1}')
        axes[row, col].axis('off')
        
        # 掩码
        axes[row + 1, col].imshow(masks[i, 0], cmap='gray')
        axes[row + 1, col].set_title(f'Mask {i+1}')
        axes[row + 1, col].axis('off')
    
    # 隐藏空白子图
    for ax in axes.flat:
        if not ax.images:
            ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_metrics_to_json(metrics: Dict, filepath: str):
    """保存指标到 JSON 文件"""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return obj
    
    metrics_converted = {k: convert(v) for k, v in metrics.items()}
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics_converted, f, indent=2, ensure_ascii=False)
    
    print(f"指标已保存到: {filepath}")


def load_metrics_from_json(filepath: str) -> Dict:
    """从 JSON 文件加载指标"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_experiment_dir(base_dir: str = OUTPUT_DIR, 
                          experiment_name: str = None) -> str:
    """
    创建实验输出目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{timestamp}_{experiment_name}"
    else:
        dir_name = timestamp
    
    experiment_dir = os.path.join(base_dir, dir_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建子目录
    for subdir in ['checkpoints', 'predictions', 'logs']:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"实验目录已创建: {experiment_dir}")
    return experiment_dir


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, 'training_log.txt')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'lr': []
        }
        
        self._write_header()
    
    def _write_header(self):
        with open(self.log_file, 'w') as f:
            f.write(f"训练日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """记录 epoch 指标"""
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['train_dice'].append(train_metrics.get('dice', 0))
        self.history['val_dice'].append(val_metrics.get('dice', 0))
        self.history['train_iou'].append(train_metrics.get('iou', 0))
        self.history['val_iou'].append(val_metrics.get('iou', 0))
        self.history['lr'].append(lr)
        
        message = f"Epoch {epoch:3d} | "
        message += f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
        message += f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
        message += f"Val Dice: {val_metrics.get('dice', 0):.4f} | "
        message += f"Val IoU: {val_metrics.get('iou', 0):.4f} | "
        message += f"LR: {lr:.2e}"
        
        self.log(message)
    
    def save_history(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(self.log_dir, 'training_history.json')
        save_metrics_to_json(self.history, filepath)
        return filepath
    
    def plot_history(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'training_curves.png')
        plot_training_history(self.history, save_path=save_path, show=False)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'max', verbose: bool = True):
        """
        参数:
            patience: 等待改善的 epoch 数
            min_delta: 最小改善量
            mode: 'min' 或 'max'
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def print_gpu_info():
    """打印 GPU 信息"""
    if torch.cuda.is_available():
        print(f"\n✓ 检测到 {torch.cuda.device_count()} 个 GPU:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}")
            print(f"      显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"      CUDA 核心: {props.multi_processor_count} SM")
        
        print(f"\n  PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        return True
    else:
        print("\n✗ 未检测到 GPU，将使用 CPU 训练")
        return False


def save_checkpoint(model, optimizer, epoch: int, metrics: Dict, 
                    filepath: str, scheduler=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"检查点已加载: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


if __name__ == "__main__":
    print("=" * 60)
    print("工具函数测试 (PyTorch)")
    print("=" * 60)
    
    # 测试 GPU 检测
    print_gpu_info()
    
    # 创建测试数据
    print("\n创建测试数据...")
    
    # 模拟训练历史
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.28],
        'train_dice': [0.5, 0.6, 0.7, 0.75, 0.8],
        'val_dice': [0.48, 0.58, 0.68, 0.72, 0.76],
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("绘制训练曲线...")
    plot_training_history(history, show=False, 
                         save_path=os.path.join(OUTPUT_DIR, 'test_history.png'))
    
    # 测试早停
    print("\n测试早停机制...")
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.7, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        stop = early_stop(score)
        print(f"  Epoch {i}: score={score}, stop={stop}")
        if stop:
            break
    
    print("\n✓ 工具函数测试完成")
