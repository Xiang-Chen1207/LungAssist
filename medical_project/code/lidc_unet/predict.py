"""
预测脚本 (PyTorch 版本)
对新图像进行分割预测
"""
import os
import sys
import argparse
import numpy as np
from glob import glob
from PIL import Image

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIR
from src.model import build_unet


def load_model(model_path: str, device: torch.device, model_type: str = 'standard'):
    """加载模型"""
    model = build_unet(model_type)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型已加载: {model_path}")
    return model


def preprocess_image(image_path: str, target_size: tuple = (IMG_HEIGHT, IMG_WIDTH)):
    """
    预处理单张图像
    
    返回:
        预处理后的张量 (1, C, H, W)
    """
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    return tensor


def postprocess_mask(mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """后处理预测掩码"""
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8)
    return mask


@torch.no_grad()
def predict_single(model, image_path: str, device: torch.device,
                   threshold: float = 0.5,
                   save_path: str = None, show: bool = True):
    """
    对单张图像进行预测
    """
    # 预处理
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # 预测
    pred = model(img_tensor)
    
    # 后处理
    mask = postprocess_mask(pred, threshold)
    prob_map = pred.squeeze().cpu().numpy()
    
    # 可视化
    if show or save_path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 原图
        original = Image.open(image_path).convert('RGB')
        original = original.resize((IMG_WIDTH, IMG_HEIGHT))
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 概率图
        axes[1].imshow(prob_map, cmap='jet')
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        
        # 二值掩码
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f'Binary Mask (threshold={threshold})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    return mask, prob_map


@torch.no_grad()
def predict_batch(model, image_paths: list, device: torch.device,
                  threshold: float = 0.5, output_dir: str = None):
    """批量预测"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    masks = []
    
    print(f"批量预测 {len(image_paths)} 张图像...")
    
    for i, img_path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(img_path)}")
        
        save_path = None
        if output_dir:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{filename}_pred.png")
        
        mask, _ = predict_single(model, img_path, device, threshold,
                                  save_path=save_path, show=False)
        masks.append(mask)
        
        # 保存掩码
        if output_dir:
            mask_path = os.path.join(output_dir, f"{filename}_mask.npy")
            np.save(mask_path, mask)
    
    print(f"\n✓ 完成！结果保存到: {output_dir}")
    
    return masks


@torch.no_grad()
def predict_with_overlay(model, image_path: str, device: torch.device,
                         threshold: float = 0.5,
                         save_path: str = None, show: bool = True):
    """预测并创建叠加可视化"""
    # 预测
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    pred = model(img_tensor)
    
    mask = postprocess_mask(pred, threshold)
    prob_map = pred.squeeze().cpu().numpy()
    
    # 加载原图
    original = Image.open(image_path).convert('RGB')
    original = original.resize((IMG_WIDTH, IMG_HEIGHT))
    original_np = np.array(original) / 255.0
    
    # 创建叠加图
    overlay = original_np.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([1, 0, 0]) * 0.5
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(prob_map, cmap='jet')
    axes[1].set_title('Probability')
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return mask


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="LIDC-IDRI U-Net 肺结节分割预测 (PyTorch)"
    )
    
    parser.add_argument('model_path', type=str,
                        help='模型文件路径')
    parser.add_argument('input', type=str,
                        help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['small', 'standard', 'large'],
                        help='模型类型')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示结果')
    parser.add_argument('--overlay', action='store_true',
                        help='显示叠加可视化')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定使用的 GPU ID')
    
    args = parser.parse_args()
    
    # 设置 GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model_path, device, args.model_type)
    
    # 确定输入
    if os.path.isfile(args.input):
        # 单张图像
        if args.overlay:
            predict_with_overlay(
                model, args.input, device,
                threshold=args.threshold,
                save_path=args.output,
                show=not args.no_show
            )
        else:
            predict_single(
                model, args.input, device,
                threshold=args.threshold,
                save_path=args.output,
                show=not args.no_show
            )
    
    elif os.path.isdir(args.input):
        # 目录批量处理
        image_paths = glob(os.path.join(args.input, '*.png'))
        image_paths.extend(glob(os.path.join(args.input, '*.jpg')))
        image_paths.extend(glob(os.path.join(args.input, '*.jpeg')))
        
        if not image_paths:
            print(f"目录中未找到图像文件: {args.input}")
            return
        
        output_dir = args.output or os.path.join(OUTPUT_DIR, 'predictions')
        predict_batch(model, image_paths, device, args.threshold, output_dir)
    
    else:
        print(f"无效的输入路径: {args.input}")


if __name__ == "__main__":
    main()
