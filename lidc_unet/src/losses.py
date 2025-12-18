"""
损失函数模块 (PyTorch 版本)
实现用于图像分割的各种损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SMOOTH


# ==================== Dice 损失 ====================

class DiceLoss(nn.Module):
    """
    Dice 损失函数
    
    Loss = 1 - Dice
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    def __init__(self, smooth: float = SMOOTH):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class DiceCoefficient(nn.Module):
    """Dice 系数（用于指标计算）"""
    def __init__(self, smooth: float = SMOOTH):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        return (2.0 * intersection + self.smooth) / (union + self.smooth)


# ==================== IoU 损失 ====================

class IoULoss(nn.Module):
    """
    IoU / Jaccard 损失函数
    
    Loss = 1 - IoU
    IoU = |A ∩ B| / |A ∪ B|
    """
    def __init__(self, smooth: float = SMOOTH):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


class IoUScore(nn.Module):
    """IoU 分数（用于指标计算）"""
    def __init__(self, smooth: float = SMOOTH):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        return (intersection + self.smooth) / (union + self.smooth)


# ==================== 组合损失函数 ====================

class BCEDiceLoss(nn.Module):
    """
    Binary Cross-Entropy + Dice Loss
    
    医学图像分割中最常用的组合损失函数
    BCE 关注像素级分类，Dice 关注整体分割质量
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCEIoULoss(nn.Module):
    """Binary Cross-Entropy + IoU Loss"""
    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.iou = IoULoss()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.iou_weight * self.iou(pred, target)


# ==================== Focal Loss ====================

class FocalLoss(nn.Module):
    """
    Focal Loss - 专门用于处理类别不平衡问题
    
    FL(p) = -α * (1-p)^γ * log(p)
    
    参数:
        alpha: 平衡因子
        gamma: 聚焦参数，增大 γ 会更关注难分类样本
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 防止 log(0)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # 计算交叉熵
        ce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # 计算 focal weight
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = self.alpha * torch.pow(1 - p_t, self.gamma)
        
        focal_loss = focal_weight * ce
        return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice Loss 组合"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal(pred, target) + self.dice(pred, target)


# ==================== Tversky Loss ====================

class TverskyLoss(nn.Module):
    """
    Tversky Loss
    
    Tversky = TP / (TP + α*FP + β*FN)
    
    参数:
        alpha: FP的权重
        beta: FN的权重
        - α > β: 更关注减少假阳性
        - α < β: 更关注减少假阴性（常用于医学图像，宁可多检出）
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = SMOOTH):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    
    FTL = (1 - Tversky)^γ
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = SMOOTH
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)


# ==================== 损失函数工厂 ====================

def get_loss_function(loss_name: str) -> nn.Module:
    """
    根据名称获取损失函数
    
    参数:
        loss_name: 损失函数名称
    
    返回:
        对应的损失函数模块
    """
    loss_functions = {
        'dice': DiceLoss(),
        'iou': IoULoss(),
        'bce': nn.BCELoss(),
        'bce_dice': BCEDiceLoss(),
        'bce_iou': BCEIoULoss(),
        'focal': FocalLoss(),
        'focal_dice': FocalDiceLoss(),
        'tversky': TverskyLoss(),
        'focal_tversky': FocalTverskyLoss(),
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"未知损失函数: {loss_name}\n可选: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name]


# ==================== 便捷函数 ====================

def dice_coef(pred: torch.Tensor, target: torch.Tensor, smooth: float = SMOOTH) -> torch.Tensor:
    """计算 Dice 系数"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = SMOOTH) -> torch.Tensor:
    """计算 IoU 分数"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


if __name__ == "__main__":
    print("=" * 60)
    print("损失函数测试 (PyTorch)")
    print("=" * 60)
    
    # 创建测试数据
    torch.manual_seed(42)
    
    pred = torch.tensor([[[[0.9, 0.8, 0.1, 0.0],
                           [0.8, 0.7, 0.2, 0.1],
                           [0.1, 0.2, 0.1, 0.0],
                           [0.0, 0.1, 0.0, 0.1]]]], dtype=torch.float32)
    
    target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)
    
    print(f"\n预测形状: {pred.shape}")
    print(f"目标形状: {target.shape}")
    
    # 测试各个损失函数
    print("\n损失函数值:")
    print("-" * 40)
    
    losses = [
        ("Dice Coef", lambda p, t: dice_coef(p, t)),
        ("Dice Loss", DiceLoss()),
        ("IoU Score", lambda p, t: iou_score(p, t)),
        ("IoU Loss", IoULoss()),
        ("BCE Loss", nn.BCELoss()),
        ("BCE+Dice Loss", BCEDiceLoss()),
        ("Focal Loss", FocalLoss()),
        ("Tversky Loss", TverskyLoss()),
    ]
    
    for name, loss_fn in losses:
        if callable(loss_fn):
            value = loss_fn(pred, target)
        else:
            value = loss_fn(pred, target)
        print(f"{name:20s}: {value.item():.6f}")
