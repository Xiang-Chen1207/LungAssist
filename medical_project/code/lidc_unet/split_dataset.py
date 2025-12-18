"""
数据集划分脚本 (带医生共识逻辑)
将 LIDC-IDRI 数据按病例级别划分为训练集、验证集、测试集
只保留至少3位医生同意有结节的样本
"""
import os
import shutil
import random
import argparse
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DATA_ROOT, SPLIT_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, 
    IMG_HEIGHT, IMG_WIDTH, create_directories
)

# 图像尺寸
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# 共识阈值
MIN_VOTES = 3  # 至少需要几位医生同意
NOISE_THRESHOLD = 10  # mask像素和小于此值视为无结节


def get_patient_nodule_info(data_root):
    """
    扫描数据目录，获取所有病例和结节信息
    
    返回:
        patient_nodules: dict, {patient_id: [nodule_paths]}
    """
    patient_nodules = defaultdict(list)
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    
    # 遍历所有病例文件夹
    for patient_id in os.listdir(data_root):
        patient_path = os.path.join(data_root, patient_id)
        
        if not os.path.isdir(patient_path):
            continue
        
        # 检查是否符合 LIDC-IDRI-XXXX 格式
        if not patient_id.startswith("LIDC-IDRI-"):
            continue
        
        # 遍历该病例下的所有结节
        for nodule in os.listdir(patient_path):
            nodule_path = os.path.join(patient_path, nodule)
            if os.path.isdir(nodule_path):
                # 检查是否包含images文件夹
                images_path = os.path.join(nodule_path, "images")
                if os.path.exists(images_path):
                    patient_nodules[patient_id].append(nodule_path)
    
    return patient_nodules


def split_patients(patient_ids, train_ratio, val_ratio, test_ratio, seed):
    """
    按比例划分病例ID
    """
    random.seed(seed)
    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)
    
    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def process_nodule_with_consensus(nodule_path, dest_dir, patient_id, nodule_name):
    """
    处理单个结节目录，应用医生共识逻辑
    
    共识规则:
    1. 读取4位医生的mask
    2. 统计有多少位医生认为存在结节 (mask sum > NOISE_THRESHOLD)
    3. 如果 >= MIN_VOTES 位医生同意 → 有结节，取面积最大的mask作为GT
    4. 如果 < MIN_VOTES 位医生同意 → 无结节，取面积最小的mask作为GT（通常是空白）
    
    返回:
        positive_count: 有结节样本数量
        negative_count: 无结节样本数量
        total_count: 总切片数量
    """
    images_dir = os.path.join(nodule_path, "images")
    
    if not os.path.exists(images_dir):
        return 0, 0, 0
    
    # 创建目标目录
    dest_nodule_dir = os.path.join(dest_dir, patient_id, nodule_name)
    dest_images_dir = os.path.join(dest_nodule_dir, "images")
    dest_masks_dir = os.path.join(dest_nodule_dir, "masks")
    
    positive_count = 0  # 有结节
    negative_count = 0  # 无结节
    total_count = 0
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    for filename in image_files:
        total_count += 1
        img_path = os.path.join(images_dir, filename)
        
        # 读取图像 (灰度)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        
        # 读取4位医生的masks
        masks = []
        for i in range(4):
            mask_path = os.path.join(nodule_path, f"mask-{i}", filename)
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize(m, IMAGE_SIZE)
                    masks.append(m)
                else:
                    masks.append(np.zeros(IMAGE_SIZE, dtype=np.uint8))
            else:
                masks.append(np.zeros(IMAGE_SIZE, dtype=np.uint8))
        
        # === 共识逻辑 ===
        mask_sums = [np.sum(m) for m in masks]
        
        # 统计有多少个医生认为有结节 (阈值排除噪点)
        votes = sum(1 for s in mask_sums if s > NOISE_THRESHOLD)
        
        # 判定
        if votes >= MIN_VOTES:
            # 有结节: 取面积最大的mask作为GT
            gt_mask = masks[np.argmax(mask_sums)]
            gt_mask = (gt_mask > 127).astype(np.uint8) * 255  # 二值化
            positive_count += 1
        else:
            # 无结节: 取面积最小的mask作为GT (通常是空白或接近空白)
            gt_mask = masks[np.argmin(mask_sums)]
            gt_mask = (gt_mask > 127).astype(np.uint8) * 255  # 二值化
            negative_count += 1
        
        # 创建目录并保存
        os.makedirs(dest_images_dir, exist_ok=True)
        os.makedirs(dest_masks_dir, exist_ok=True)
        
        # 保存图像 (转为3通道RGB以保持与原代码兼容)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(dest_images_dir, filename), img_rgb)
        
        # 保存mask
        cv2.imwrite(os.path.join(dest_masks_dir, filename), gt_mask)
    
    return positive_count, negative_count, total_count


def split_dataset(data_root=DATA_ROOT, output_dir=SPLIT_DATA_DIR,
                  train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, 
                  test_ratio=TEST_RATIO, seed=RANDOM_SEED,
                  min_votes=MIN_VOTES):
    """
    主函数：执行数据集划分（带共识逻辑）
    """
    print("=" * 60)
    print("LIDC-IDRI 数据集划分 (带医生共识逻辑)")
    print("=" * 60)
    print(f"源数据目录: {data_root}")
    print(f"输出目录: {output_dir}")
    print(f"划分比例: 训练 {train_ratio*100}% / 验证 {val_ratio*100}% / 测试 {test_ratio*100}%")
    print(f"共识阈值: 至少 {min_votes} 位医生同意")
    print(f"随机种子: {seed}")
    print("=" * 60)
    
    # 获取病例信息
    print("\n[1/4] 扫描数据目录...")
    patient_nodules = get_patient_nodule_info(data_root)
    n_patients = len(patient_nodules)
    n_nodules = sum(len(v) for v in patient_nodules.values())
    print(f"  ✓ 找到 {n_patients} 个病例, 共 {n_nodules} 个结节文件夹")
    
    if n_patients == 0:
        raise ValueError("未找到任何病例数据，请检查数据路径")
    
    # 划分病例
    print("\n[2/4] 划分病例...")
    train_ids, val_ids, test_ids = split_patients(
        patient_nodules.keys(), train_ratio, val_ratio, test_ratio, seed
    )
    print(f"  ✓ 训练集: {len(train_ids)} 病例")
    print(f"  ✓ 验证集: {len(val_ids)} 病例")
    print(f"  ✓ 测试集: {len(test_ids)} 病例")
    
    # 创建输出目录
    print("\n[3/4] 创建输出目录...")
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    # 清空旧目录
    for d in [train_dir, val_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("  ✓ 目录创建完成")
    
    # 处理数据（应用共识逻辑）
    print("\n[4/4] 处理数据 (应用共识逻辑)...")
    
    split_mapping = [
        (train_ids, train_dir, "训练集"),
        (val_ids, val_dir, "验证集"),
        (test_ids, test_dir, "测试集")
    ]
    
    stats = {
        'train': {'positive': 0, 'negative': 0, 'total': 0},
        'val': {'positive': 0, 'negative': 0, 'total': 0},
        'test': {'positive': 0, 'negative': 0, 'total': 0}
    }
    
    for patient_ids, dest_dir, name in split_mapping:
        print(f"\n  处理{name}...")
        
        stat_key = name.replace('集', '').replace('训练', 'train').replace('验证', 'val').replace('测试', 'test')
        
        for patient_id in tqdm(patient_ids, desc=f"  {name}"):
            for nodule_path in patient_nodules[patient_id]:
                nodule_name = os.path.basename(nodule_path)
                pos, neg, total = process_nodule_with_consensus(
                    nodule_path, dest_dir, patient_id, nodule_name
                )
                stats[stat_key]['positive'] += pos
                stats[stat_key]['negative'] += neg
                stats[stat_key]['total'] += total
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("划分完成！统计信息:")
    print("=" * 70)
    print(f"共识规则: >= {min_votes} 位医生同意 → 有结节 (取最大mask)")
    print(f"          < {min_votes} 位医生同意 → 无结节 (取最小mask)")
    print("-" * 70)
    print(f"{'数据集':<8} {'有结节':>10} {'无结节':>10} {'总计':>10} {'正样本比例':>12}")
    print("-" * 70)
    
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        pos = stats[split_name]['positive']
        neg = stats[split_name]['negative']
        total = pos + neg
        ratio = pos / total * 100 if total > 0 else 0
        print(f"  {split_name.upper():<6} {pos:>10} {neg:>10} {total:>10} {ratio:>10.1f}%")
    
    total_pos = sum(s['positive'] for s in stats.values())
    total_neg = sum(s['negative'] for s in stats.values())
    total_all = total_pos + total_neg
    print("-" * 70)
    print(f"  {'总计':<6} {total_pos:>10} {total_neg:>10} {total_all:>10} {total_pos/total_all*100:>10.1f}%")
    print("=" * 70)
    
    # 保存划分信息
    split_info_path = os.path.join(output_dir, "split_info.txt")
    with open(split_info_path, 'w') as f:
        f.write(f"数据集划分信息\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"源目录: {data_root}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"划分比例: {train_ratio}/{val_ratio}/{test_ratio}\n")
        f.write(f"共识阈值: >= {min_votes} 位医生同意 → 有结节\n")
        f.write(f"噪点阈值: mask sum > {NOISE_THRESHOLD}\n\n")
        
        f.write(f"统计信息:\n")
        f.write(f"  训练集: {stats['train']['positive']} 有结节 + {stats['train']['negative']} 无结节 = {stats['train']['positive']+stats['train']['negative']} 总计\n")
        f.write(f"  验证集: {stats['val']['positive']} 有结节 + {stats['val']['negative']} 无结节 = {stats['val']['positive']+stats['val']['negative']} 总计\n")
        f.write(f"  测试集: {stats['test']['positive']} 有结节 + {stats['test']['negative']} 无结节 = {stats['test']['positive']+stats['test']['negative']} 总计\n\n")
        
        f.write(f"训练集病例 ({len(train_ids)}):\n")
        for pid in sorted(train_ids):
            f.write(f"  {pid}\n")
        
        f.write(f"\n验证集病例 ({len(val_ids)}):\n")
        for pid in sorted(val_ids):
            f.write(f"  {pid}\n")
        
        f.write(f"\n测试集病例 ({len(test_ids)}):\n")
        for pid in sorted(test_ids):
            f.write(f"  {pid}\n")
    
    print(f"\n划分信息已保存到: {split_info_path}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIDC-IDRI 数据集划分工具 (带共识逻辑)")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help="源数据目录路径")
    parser.add_argument("--output_dir", type=str, default=SPLIT_DATA_DIR,
                        help="输出目录路径")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO,
                        help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO,
                        help="测试集比例")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="随机种子")
    parser.add_argument("--min_votes", type=int, default=MIN_VOTES,
                        help="最少需要几位医生同意 (默认3)")
    
    args = parser.parse_args()
    
    # 验证比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("训练/验证/测试比例之和必须等于1")
    
    split_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_votes=args.min_votes
    )
