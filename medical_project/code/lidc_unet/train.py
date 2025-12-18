"""
训练脚本 (PyTorch 版本)
执行 U-Net 模型的训练流程
"""
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR,
    USE_AUGMENTATION, PREPROCESS_METHOD,
    create_directories, print_config, set_seed, get_device
)
from src.dataset import create_dataloaders
from src.model import UNet, build_unet, count_parameters
from src.losses import get_loss_function, dice_coef, iou_score
from src.metrics import compute_all_metrics, MetricTracker
from src.utils import (
    TrainingLogger, EarlyStopping, print_gpu_info,
    save_checkpoint, create_experiment_dir, plot_training_history
)


def train_one_epoch(model, dataloader, criterion, optimizer, device, 
                    epoch: int, total_epochs: int) -> dict:
    """
    训练一个 epoch
    
    返回:
        包含 loss, dice, iou 的字典
    """
    model.train()
    tracker = MetricTracker()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]', 
                leave=False, ncols=100)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            dice = dice_coef(outputs, masks)
            iou = iou_score(outputs, masks)
        
        tracker.update(loss=loss.item(), dice=dice.item(), iou=iou.item())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })
    
    return tracker.get_average()


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch: int, 
             total_epochs: int) -> dict:
    """
    验证模型
    
    返回:
        包含 loss, dice, iou 等指标的字典
    """
    model.eval()
    tracker = MetricTracker()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Val]',
                leave=False, ncols=100)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 计算所有指标
        metrics = compute_all_metrics(outputs, masks)
        metrics['loss'] = loss.item()
        tracker.update(**metrics)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metrics["dice"]:.4f}'
        })
    
    return tracker.get_average()


def train(train_dir: str = TRAIN_DIR,
          val_dir: str = VAL_DIR,
          batch_size: int = BATCH_SIZE,
          epochs: int = EPOCHS,
          learning_rate: float = LEARNING_RATE,
          loss_function: str = 'bce_dice',
          model_type: str = 'standard',
          use_augmentation: bool = USE_AUGMENTATION,
          experiment_name: str = None,
          resume_from: str = None,
          num_workers: int = NUM_WORKERS,
          preprocess_method: str = PREPROCESS_METHOD):
    """
    主训练函数

    参数:
        train_dir: 训练数据目录
        val_dir: 验证数据目录
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        loss_function: 损失函数名称
        model_type: 模型类型 ('small', 'standard', 'large')
        use_augmentation: 是否使用数据增强
        experiment_name: 实验名称
        resume_from: 继续训练的检查点路径
        num_workers: DataLoader 工作进程数
        preprocess_method: 图像预处理方法
    """
    print("\n" + "=" * 60)
    print("LIDC-IDRI U-Net 肺结节分割 - 训练 (PyTorch)")
    print("=" * 60)
    
    # 设置随机种子
    set_seed()
    
    # 创建必要目录
    create_directories()

    # 打印配置
    print_config(preprocess_method)

    # 获取设备
    device = get_device()

    # 创建实验目录
    if experiment_name is None:
        experiment_name = f"unet_{preprocess_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_dir = create_experiment_dir(OUTPUT_DIR, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')

    # 初始化日志
    logger = TrainingLogger(log_dir)
    logger.log(f"实验名称: {experiment_name}")
    logger.log(f"设备: {device}")
    logger.log(f"训练目录: {train_dir}")
    logger.log(f"验证目录: {val_dir}")
    logger.log(f"预处理方法: {preprocess_method}")

    # 初始化 CSV 日志文件
    csv_log_path = os.path.join(experiment_dir, 'training_log.csv')
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as f:
            f.write('epoch,train_loss,train_dice,train_iou,val_loss,val_dice,val_iou,lr\n')
    
    # ==================== 加载数据 ====================
    logger.log("\n[1/4] 加载数据...")

    train_loader, val_loader, _ = create_dataloaders(
        train_dir, val_dir, val_dir,  # 暂时用 val_dir 代替 test_dir
        batch_size=batch_size,
        num_workers=num_workers,
        augment_train=use_augmentation,
        preprocess_method=preprocess_method
    )
    
    # ==================== 构建模型 ====================
    logger.log("\n[2/4] 构建模型...")
    
    model = build_unet(model_type)
    model = model.to(device)
    
    logger.log(f"模型类型: {model_type}")
    logger.log(f"模型参数量: {count_parameters(model):,}")
    
    # 损失函数
    criterion = get_loss_function(loss_function)
    logger.log(f"损失函数: {loss_function}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                            weight_decay=WEIGHT_DECAY)
    logger.log(f"优化器: AdamW (lr={learning_rate}, weight_decay={WEIGHT_DECAY})")
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR
    )
    
    # 早停
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='max')
    
    # 起始 epoch
    start_epoch = 1
    best_dice = 0.0
    
    # 从检查点恢复
    if resume_from and os.path.exists(resume_from):
        logger.log(f"\n从检查点恢复: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('metrics', {}).get('dice', 0)
        logger.log(f"从 Epoch {start_epoch} 继续，最佳 Dice: {best_dice:.4f}")
    
    # ==================== 开始训练 ====================
    logger.log("\n[3/4] 开始训练...")
    logger.log(f"Epochs: {epochs}, Batch Size: {batch_size}")
    logger.log("-" * 60)
    
    try:
        for epoch in range(start_epoch, epochs + 1):
            # 训练
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, epochs
            )
            
            # 验证
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, epochs
            )
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录日志
            logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
            
            # 写入 CSV 日志
            with open(csv_log_path, 'a') as f:
                f.write(f"{epoch},{train_metrics['loss']:.6f},{train_metrics['dice']:.6f},{train_metrics['iou']:.6f},"
                        f"{val_metrics['loss']:.6f},{val_metrics['dice']:.6f},{val_metrics['iou']:.6f},{current_lr:.8f}\n")
            
            # 更新学习率
            scheduler.step(val_metrics['dice'])
            
            # 保存最佳模型
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_metrics, 
                               best_model_path, scheduler)
                logger.log(f"  ★ 新的最佳模型! Dice: {best_dice:.4f}")
            
            # 定期保存检查点
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                save_checkpoint(model, optimizer, epoch, val_metrics,
                               checkpoint_path, scheduler)
            
            # 早停检查
            if early_stopping(val_metrics['dice']):
                logger.log(f"\n早停触发，停止训练")
                break
        
        logger.log("\n训练完成！")
        
    except KeyboardInterrupt:
        logger.log("\n训练被用户中断")
    
    # ==================== 保存结果 ====================
    logger.log("\n[4/4] 保存训练结果...")
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, epoch, val_metrics, final_model_path, scheduler)
    
    # 保存训练历史
    history_path = logger.save_history()
    
    # 绘制训练曲线
    logger.plot_history()
    
    # 打印总结
    logger.log("\n" + "=" * 60)
    logger.log("训练总结:")
    logger.log(f"  最佳验证 Dice: {best_dice:.4f}")
    logger.log(f"  最佳模型: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    logger.log(f"  实验目录: {experiment_dir}")
    logger.log("=" * 60)
    
    return model, logger.history, experiment_dir


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="LIDC-IDRI U-Net 肺结节分割训练 (PyTorch)"
    )
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR,
                        help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default=VAL_DIR,
                        help='验证数据目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='学习率')
    parser.add_argument('--loss', type=str, default='bce_dice',
                        choices=['dice', 'bce', 'bce_dice', 'focal', 'tversky'],
                        help='损失函数')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='standard',
                        choices=['small', 'standard', 'large'],
                        help='模型类型')
    
    # 其他参数
    parser.add_argument('--no_augment', action='store_true',
                        help='禁用数据增强')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--resume', type=str, default=None,
                        help='继续训练的检查点路径')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help='DataLoader 工作进程数')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定使用的 GPU ID')
    parser.add_argument('--preprocess', type=str, default=PREPROCESS_METHOD,
                        choices=['none', 'clahe', 'clahe_median', 'clahe_smooth',
                                 'clahe_median_edge_ghpf', 'clahe_smooth_edge_ghpf'],
                        help='图像预处理方法')

    args = parser.parse_args()
    
    # 设置 GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"使用 GPU {args.gpu}")
    
    # 运行训练
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_function=args.loss,
        model_type=args.model,
        use_augmentation=not args.no_augment,
        experiment_name=args.name,
        resume_from=args.resume,
        num_workers=args.workers,
        preprocess_method=args.preprocess
    )


if __name__ == "__main__":
    main()
