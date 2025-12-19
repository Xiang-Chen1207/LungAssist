"""
U-Net 模型定义模块 (PyTorch 版本)
实现用于医学图像分割的 U-Net 架构
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    ENCODER_FILTERS, BRIDGE_FILTERS, DROPOUT_RATE
)


class ConvBlock(nn.Module):
    """
    卷积块：两次 Conv2D + BatchNorm + ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dropout_rate: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    编码器块：卷积块 + 最大池化
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        skip = self.conv_block(x)  # 用于跳跃连接
        pooled = self.pool(skip)   # 继续下采样
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    解码器块：转置卷积（上采样） + 跳跃连接 + 卷积块
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        # 转置卷积上采样
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                          kernel_size=2, stride=2)
        # 卷积块（输入通道数 = out_channels * 2，因为要拼接跳跃连接）
        self.conv_block = ConvBlock(out_channels * 2, out_channels, 
                                    dropout_rate=dropout_rate)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # 处理尺寸不匹配的情况
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)  # 跳跃连接
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    U-Net 模型
    
    架构示意:
        输入 (3, 128, 128)
            ↓
        [编码器] 64 → 128 → 256 → 512  (每层后池化)
            ↓
        [桥接层] 1024
            ↓
        [解码器] 512 → 256 → 128 → 64  ← 跳跃连接
            ↓
        输出 (1, 128, 128) sigmoid
    """
    
    def __init__(self, 
                 in_channels: int = IMG_CHANNELS,
                 out_channels: int = 1,
                 encoder_filters: list = None,
                 bridge_filters: int = BRIDGE_FILTERS,
                 dropout_rate: float = DROPOUT_RATE):
        """
        初始化 U-Net
        
        参数:
            in_channels: 输入通道数（RGB=3）
            out_channels: 输出通道数（二分类=1）
            encoder_filters: 编码器各层滤波器数量
            bridge_filters: 桥接层滤波器数量
            dropout_rate: Dropout 比率
        """
        super().__init__()
        
        if encoder_filters is None:
            encoder_filters = ENCODER_FILTERS
        
        self.encoder_filters = encoder_filters
        
        # ==================== 编码器 ====================
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        
        for i, filters in enumerate(encoder_filters):
            # 逐层增加 dropout
            current_dropout = dropout_rate * (i / len(encoder_filters))
            self.encoders.append(EncoderBlock(prev_channels, filters, current_dropout))
            prev_channels = filters
        
        # ==================== 桥接层 ====================
        self.bridge = ConvBlock(encoder_filters[-1], bridge_filters, dropout_rate=dropout_rate)
        
        # ==================== 解码器 ====================
        self.decoders = nn.ModuleList()
        decoder_filters = encoder_filters[::-1]  # 反转
        prev_channels = bridge_filters
        
        for i, filters in enumerate(decoder_filters):
            current_dropout = dropout_rate * (1 - i / len(decoder_filters))
            self.decoders.append(DecoderBlock(prev_channels, filters, current_dropout))
            prev_channels = filters
        
        # ==================== 输出层 ====================
        self.output_conv = nn.Conv2d(encoder_filters[0], out_channels, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """He 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 编码器：保存跳跃连接
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)
        
        # 桥接层
        x = self.bridge(x)
        
        # 解码器：使用跳跃连接（反向顺序）
        skips = skips[::-1]
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)
        
        # 输出
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        
        return x


class UNetSmall(UNet):
    """小型 U-Net，适用于显存较小的 GPU"""
    def __init__(self, in_channels=IMG_CHANNELS, out_channels=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_filters=[32, 64, 128, 256],
            bridge_filters=512,
            dropout_rate=0.2
        )


class UNetLarge(UNet):
    """大型 U-Net，适用于显存较大的 GPU"""
    def __init__(self, in_channels=IMG_CHANNELS, out_channels=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_filters=[64, 128, 256, 512, 1024],
            bridge_filters=2048,
            dropout_rate=0.3
        )


def build_unet(model_type: str = 'standard', **kwargs) -> UNet:
    """
    工厂函数：根据类型创建 U-Net
    
    参数:
        model_type: 'small', 'standard', 'large'
        **kwargs: 传递给模型的其他参数
    """
    models = {
        'small': UNetSmall,
        'standard': UNet,
        'large': UNetLarge
    }
    
    if model_type not in models:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(models.keys())}")
    
    return models[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size=(3, 128, 128)):
    """打印模型摘要"""
    try:
        from torchinfo import summary
        summary(model, input_size=(1, *input_size), col_names=["input_size", "output_size", "num_params"])
    except ImportError:
        print(f"模型: {model.__class__.__name__}")
        print(f"参数量: {count_parameters(model):,}")


if __name__ == "__main__":
    print("=" * 60)
    print("U-Net 模型测试 (PyTorch)")
    print("=" * 60)
    
    # 测试标准模型
    print("\n[1] 标准 U-Net:")
    model = UNet()
    print(f"参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    print(f"输出范围: [{y.min():.4f}, {y.max():.4f}]")
    
    # 测试小型模型
    print("\n[2] 小型 U-Net:")
    model_small = UNetSmall()
    print(f"参数量: {count_parameters(model_small):,}")
    
    # 测试大型模型
    print("\n[3] 大型 U-Net:")
    model_large = UNetLarge()
    print(f"参数量: {count_parameters(model_large):,}")
    
    # GPU 测试
    if torch.cuda.is_available():
        print("\n[4] GPU 测试:")
        device = torch.device("cuda")
        model = model.to(device)
        x = x.to(device)
        y = model(x)
        print(f"GPU 输出: {y.shape}, 设备: {y.device}")
