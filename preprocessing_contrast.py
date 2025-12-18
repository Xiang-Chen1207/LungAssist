import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
import pandas as pd
from skimage import exposure, filters, segmentation, morphology, color, util, measure, restoration
from scipy.signal import wiener
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR = "/home/chenx/code/medical_project/data/LIDC-IDRI-slices"
IMAGE_SIZE = (128, 128) 
SAVE_RESULT_DIR = "./output_comparison_results"
CSV_OUTPUT = "./method_comparison.csv"
TEST_SAMPLE_COUNT = 100  # 测试前100张图
# ===========================================

class LIDC_ComparisonProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(SAVE_RESULT_DIR):
            os.makedirs(SAVE_RESULT_DIR)

    def load_data_and_gt(self, max_samples=None):
        """加载数据并生成Ground Truth"""
        patient_dirs = glob.glob(os.path.join(self.data_dir, '**', 'images'), recursive=True)
        valid_samples = []
        
        print(f"扫描数据集中: {self.data_dir} ...")
        
        for img_dir in tqdm(patient_dirs[:300]): 
            if max_samples and len(valid_samples) >= max_samples:
                break
                
            base_dir = os.path.dirname(img_dir)
            image_files = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
            
            for img_path in image_files:
                if max_samples and len(valid_samples) >= max_samples:
                    break
                    
                filename = os.path.basename(img_path)
                
                # 读取图像
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, IMAGE_SIZE)

                # 读取 Masks
                masks = []
                for i in range(4):
                    mask_path = os.path.join(base_dir, f"mask-{i}", filename)
                    if os.path.exists(mask_path):
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        m = cv2.resize(m, IMAGE_SIZE)
                        masks.append(m)
                    else:
                        masks.append(np.zeros(IMAGE_SIZE, dtype=np.uint8))

                # 共识逻辑
                mask_sums = [np.sum(m) for m in masks]
                votes = sum(1 for s in mask_sums if s > 10)

                if votes > 2:
                    gt_mask = masks[np.argmax(mask_sums)]
                    gt_mask = (gt_mask > 127).astype(np.uint8)
                    
                    valid_samples.append({
                        'image': img,
                        'mask': gt_mask,
                        'id': f"{os.path.basename(base_dir)}_{filename}"
                    })
        
        print(f"筛选出 {len(valid_samples)} 个正样本用于测试。")
        return valid_samples

    # ============ Phase 1: 预处理方法 ============
    def preprocess_clahe_median(self, image):
        """CLAHE + 中值滤波"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        denoised = cv2.medianBlur(enhanced, 3)
        return enhanced, denoised
    
    def preprocess_clahe_smooth(self, image):
        """CLAHE + 平滑（高斯）"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return enhanced, denoised
    
    def preprocess_clahe_wiener(self, image):
        """CLAHE + 维纳滤波"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        # 维纳滤波
        img_float = enhanced.astype(float) / 255.0
        denoised_float = wiener(img_float, (5, 5))
        denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
        return enhanced, denoised

    # ============ Phase 2: ROI 提取方法 ============
    def roi_watershed(self, denoised):
        """分水岭流水线提取ROI"""
        # 基础二值化
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 寻找确定前景
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 寻找确定背景
        kernel = np.ones((3,3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        
        # 分水岭标记
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 分水岭算法
        img_color = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # 生成ROI mask
        roi_mask = np.zeros_like(denoised, dtype=np.uint8)
        roi_mask[markers > 1] = 1
        
        return roi_mask
    
    def roi_edge_detection(self, denoised):
        """边缘检测流水线提取ROI"""
        # Canny边缘检测
        edges = cv2.Canny(denoised, 50, 150)
        
        # 形态学闭运算连接边缘
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 填充轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = np.zeros_like(denoised, dtype=np.uint8)
        
        # 只保留较大的区域
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                cv2.drawContours(roi_mask, [contour], -1, 1, -1)
        
        return roi_mask

    # ============ Phase 3: 分割策略 ============
    def segment_ghpf_otsu(self, denoised, roi_mask):
        """GHPF + Otsu分割"""
        # 高通滤波器 (Gaussian High-Pass Filter)
        lowpass = cv2.GaussianBlur(denoised, (15, 15), 3)
        highpass = cv2.subtract(denoised, lowpass)
        highpass = cv2.add(highpass, 127)  # 添加偏移避免负值
        
        # 应用ROI
        masked = highpass.copy()
        masked[roi_mask == 0] = 0
        
        if np.sum(roi_mask) == 0:
            return highpass, np.zeros_like(denoised)
        
        # Otsu阈值
        try:
            thresh_val = filters.threshold_otsu(masked[roi_mask > 0])
            binary = (masked > thresh_val).astype(np.uint8)
        except:
            binary = np.zeros_like(denoised)
        
        # 形状筛选
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        
        final_mask = np.zeros_like(binary)
        for prop in props:
            if 10 < prop.area < 600:
                if prop.eccentricity < 0.85 and prop.solidity > 0.75:
                    final_mask[labels == prop.label] = 1
        
        return highpass, final_mask
    
    def segment_ghpf_reconstruction_otsu(self, denoised, roi_mask):
        """GHPF + 灰度重建 + Otsu分割"""
        # 高通滤波
        lowpass = cv2.GaussianBlur(denoised, (15, 15), 3)
        highpass = cv2.subtract(denoised, lowpass)
        highpass = cv2.add(highpass, 127)
        
        # 灰度重建
        seed = highpass.copy()
        seed[1:-1, 1:-1] = highpass.min()
        reconstructed = reconstruction(seed, highpass, method='dilation')
        
        # 应用ROI
        masked = reconstructed.copy()
        masked[roi_mask == 0] = 0
        
        if np.sum(roi_mask) == 0:
            return reconstructed, np.zeros_like(denoised)
        
        # Otsu阈值
        try:
            thresh_val = filters.threshold_otsu(masked[roi_mask > 0])
            binary = (masked > thresh_val).astype(np.uint8)
        except:
            binary = np.zeros_like(denoised)
        
        # 形状筛选
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        
        final_mask = np.zeros_like(binary)
        for prop in props:
            if 10 < prop.area < 600:
                if prop.eccentricity < 0.85 and prop.solidity > 0.75:
                    final_mask[labels == prop.label] = 1
        
        return reconstructed, final_mask

    # ============ 12种组合方法 ============
    def method_C1(self, image):
        """C1: CLAHE+中值 + 分水岭 + GHPF+Otsu"""
        enhanced, denoised = self.preprocess_clahe_median(image)
        roi_mask = self.roi_watershed(denoised)
        filtered, prediction = self.segment_ghpf_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C2(self, image):
        """C2: CLAHE+中值 + 分水岭 + GHPF+重建+Otsu"""
        enhanced, denoised = self.preprocess_clahe_median(image)
        roi_mask = self.roi_watershed(denoised)
        filtered, prediction = self.segment_ghpf_reconstruction_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C3(self, image):
        """C3: CLAHE+中值 + 边缘检测 + GHPF+Otsu"""
        enhanced, denoised = self.preprocess_clahe_median(image)
        roi_mask = self.roi_edge_detection(denoised)
        filtered, prediction = self.segment_ghpf_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C4(self, image):
        """C4: CLAHE+中值 + 边缘检测 + GHPF+重建+Otsu"""
        enhanced, denoised = self.preprocess_clahe_median(image)
        roi_mask = self.roi_edge_detection(denoised)
        filtered, prediction = self.segment_ghpf_reconstruction_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C5(self, image):
        """C5: CLAHE+平滑 + 分水岭 + GHPF+Otsu"""
        enhanced, denoised = self.preprocess_clahe_smooth(image)
        roi_mask = self.roi_watershed(denoised)
        filtered, prediction = self.segment_ghpf_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C6(self, image):
        """C6: CLAHE+平滑 + 分水岭 + GHPF+重建+Otsu"""
        enhanced, denoised = self.preprocess_clahe_smooth(image)
        roi_mask = self.roi_watershed(denoised)
        filtered, prediction = self.segment_ghpf_reconstruction_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C7(self, image):
        """C7: CLAHE+平滑 + 边缘检测 + GHPF+Otsu"""
        enhanced, denoised = self.preprocess_clahe_smooth(image)
        roi_mask = self.roi_edge_detection(denoised)
        filtered, prediction = self.segment_ghpf_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C8(self, image):
        """C8: CLAHE+平滑 + 边缘检测 + GHPF+重建+Otsu"""
        enhanced, denoised = self.preprocess_clahe_smooth(image)
        roi_mask = self.roi_edge_detection(denoised)
        filtered, prediction = self.segment_ghpf_reconstruction_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C9(self, image):
        """C9: CLAHE+维纳 + 分水岭 + GHPF+Otsu"""
        enhanced, denoised = self.preprocess_clahe_wiener(image)
        roi_mask = self.roi_watershed(denoised)
        filtered, prediction = self.segment_ghpf_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C10(self, image):
        """C10: CLAHE+维纳 + 分水岭 + GHPF+重建+Otsu"""
        enhanced, denoised = self.preprocess_clahe_wiener(image)
        roi_mask = self.roi_watershed(denoised)
        filtered, prediction = self.segment_ghpf_reconstruction_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C11(self, image):
        """C11: CLAHE+维纳 + 边缘检测 + GHPF+Otsu"""
        enhanced, denoised = self.preprocess_clahe_wiener(image)
        roi_mask = self.roi_edge_detection(denoised)
        filtered, prediction = self.segment_ghpf_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }
    
    def method_C12(self, image):
        """C12: CLAHE+维纳 + 边缘检测 + GHPF+重建+Otsu"""
        enhanced, denoised = self.preprocess_clahe_wiener(image)
        roi_mask = self.roi_edge_detection(denoised)
        filtered, prediction = self.segment_ghpf_reconstruction_otsu(denoised, roi_mask)
        return {
            'step1_enhanced': enhanced,
            'step2_denoised': denoised,
            'step3_roi': roi_mask * 255,
            'step4_filtered': filtered,
            'prediction': prediction
        }

    def calculate_dice(self, y_true, y_pred):
        """计算Dice系数"""
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

    def run_comparison(self):
        """运行对比测试"""
        # 加载数据
        samples = self.load_data_and_gt(max_samples=TEST_SAMPLE_COUNT)
        if not samples:
            print("未找到有效数据，请检查路径。")
            return
        
        methods = {
            'C1': self.method_C1,
            'C2': self.method_C2,
            'C3': self.method_C3,
            'C4': self.method_C4,
            'C5': self.method_C5,
            'C6': self.method_C6,
            'C7': self.method_C7,
            'C8': self.method_C8,
            'C9': self.method_C9,
            'C10': self.method_C10,
            'C11': self.method_C11,
            'C12': self.method_C12,
        }
        
        results_data = []
        
        print(f"\n开始对比测试 (前{len(samples)}个样本)...")
        
        for idx, sample in enumerate(tqdm(samples)):
            img = sample['image']
            gt = sample['mask']
            sample_id = sample['id']
            
            row = {'sample_id': sample_id}
            
            # 测试每种方法
            for method_name, method_func in methods.items():
                try:
                    result = method_func(img)
                    dice = self.calculate_dice(gt, result['prediction'])
                    row[method_name] = dice
                except Exception as e:
                    print(f"警告: {method_name} 处理样本 {sample_id} 时出错: {str(e)}")
                    row[method_name] = 0.0
            
            results_data.append(row)
        
        # 保存CSV
        df = pd.DataFrame(results_data)
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"\n结果已保存至: {CSV_OUTPUT}")
        
        # 打印统计信息
        print("\n=== 各方法平均Dice分数 ===")
        method_stats = []
        for method_name in methods.keys():
            avg_dice = df[method_name].mean()
            max_dice = df[method_name].max()
            std_dice = df[method_name].std()
            method_stats.append({
                'Method': method_name,
                'Mean': avg_dice,
                'Max': max_dice,
                'Std': std_dice
            })
            print(f"{method_name:5s}: 平均={avg_dice:.4f}, 最高={max_dice:.4f}, 标准差={std_dice:.4f}")
        
        # 保存统计信息
        stats_df = pd.DataFrame(method_stats)
        stats_df = stats_df.sort_values('Mean', ascending=False)
        stats_df.to_csv(os.path.join(SAVE_RESULT_DIR, 'method_statistics.csv'), index=False)
        
        return df, samples

    def visualize_detailed_comparison(self, sample, df):
        """可视化单个样本的详细处理流程 - 展示前6种方法"""
        img = sample['image']
        gt = sample['mask']
        sample_id = sample['id']
        
        methods = {
            'C1': self.method_C1,
            'C2': self.method_C2,
            'C3': self.method_C3,
            'C4': self.method_C4,
            'C5': self.method_C5,
            'C6': self.method_C6,
        }
        
        fig = plt.figure(figsize=(24, 18))
        
        # 原图和GT
        ax1 = plt.subplot(7, 6, 1)
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(7, 6, 2)
        ax2.imshow(gt, cmap='Greens', alpha=0.8)
        ax2.set_title('Ground Truth', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # 空白占位
        for i in range(3, 7):
            ax = plt.subplot(7, 6, i)
            ax.axis('off')
        
        # 每种方法占一行
        row = 1
        for method_name, method_func in methods.items():
            result = method_func(img)
            dice = self.calculate_dice(gt, result['prediction'])
            
            # Step 1: Enhanced
            ax = plt.subplot(7, 6, row * 6 + 1)
            ax.imshow(result['step1_enhanced'], cmap='gray')
            ax.set_title(f'{method_name}: Enhanced', fontsize=9)
            ax.axis('off')
            
            # Step 2: Denoised
            ax = plt.subplot(7, 6, row * 6 + 2)
            ax.imshow(result['step2_denoised'], cmap='gray')
            ax.set_title('Denoised', fontsize=9)
            ax.axis('off')
            
            # Step 3: ROI
            ax = plt.subplot(7, 6, row * 6 + 3)
            ax.imshow(result['step3_roi'], cmap='gray')
            ax.set_title('ROI Mask', fontsize=9)
            ax.axis('off')
            
            # Step 4: Filtered
            ax = plt.subplot(7, 6, row * 6 + 4)
            ax.imshow(result['step4_filtered'], cmap='gray')
            ax.set_title('GHPF', fontsize=9)
            ax.axis('off')
            
            # Prediction
            ax = plt.subplot(7, 6, row * 6 + 5)
            overlay_pred = color.label2rgb(
                result['prediction'], 
                image=img, 
                bg_label=0, 
                colors=['red'], 
                alpha=0.4
            )
            ax.imshow(overlay_pred)
            ax.set_title(f'Prediction', fontsize=9)
            ax.axis('off')
            
            # 对比图
            ax = plt.subplot(7, 6, row * 6 + 6)
            comparison = np.stack([img]*3, axis=-1).astype(float) / 255.0
            comparison[result['prediction'] > 0] = [1, 0, 0]
            comparison[gt > 0, 1] = 1
            ax.imshow(comparison)
            ax.set_title(f'Dice={dice:.3f}', fontsize=9, fontweight='bold')
            ax.axis('off')
            
            row += 1
        
        plt.suptitle(f'Methods C1-C6 Comparison - Sample: {sample_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = os.path.join(SAVE_RESULT_DIR, 'detailed_comparison_C1-C6.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n详细对比图(C1-C6)已保存至: {save_path}")
        plt.close()
        
        # 第二张图：C7-C12
        methods2 = {
            'C7': self.method_C7,
            'C8': self.method_C8,
            'C9': self.method_C9,
            'C10': self.method_C10,
            'C11': self.method_C11,
            'C12': self.method_C12,
        }
        
        fig = plt.figure(figsize=(24, 18))
        
        # 原图和GT
        ax1 = plt.subplot(7, 6, 1)
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(7, 6, 2)
        ax2.imshow(gt, cmap='Greens', alpha=0.8)
        ax2.set_title('Ground Truth', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        for i in range(3, 7):
            ax = plt.subplot(7, 6, i)
            ax.axis('off')
        
        row = 1
        for method_name, method_func in methods2.items():
            result = method_func(img)
            dice = self.calculate_dice(gt, result['prediction'])
            
            ax = plt.subplot(7, 6, row * 6 + 1)
            ax.imshow(result['step1_enhanced'], cmap='gray')
            ax.set_title(f'{method_name}: Enhanced', fontsize=9)
            ax.axis('off')
            
            ax = plt.subplot(7, 6, row * 6 + 2)
            ax.imshow(result['step2_denoised'], cmap='gray')
            ax.set_title('Denoised', fontsize=9)
            ax.axis('off')
            
            ax = plt.subplot(7, 6, row * 6 + 3)
            ax.imshow(result['step3_roi'], cmap='gray')
            ax.set_title('ROI Mask', fontsize=9)
            ax.axis('off')
            
            ax = plt.subplot(7, 6, row * 6 + 4)
            ax.imshow(result['step4_filtered'], cmap='gray')
            ax.set_title('GHPF', fontsize=9)
            ax.axis('off')
            
            ax = plt.subplot(7, 6, row * 6 + 5)
            overlay_pred = color.label2rgb(
                result['prediction'], 
                image=img, 
                bg_label=0, 
                colors=['red'], 
                alpha=0.4
            )
            ax.imshow(overlay_pred)
            ax.set_title(f'Prediction', fontsize=9)
            ax.axis('off')
            
            ax = plt.subplot(7, 6, row * 6 + 6)
            comparison = np.stack([img]*3, axis=-1).astype(float) / 255.0
            comparison[result['prediction'] > 0] = [1, 0, 0]
            comparison[gt > 0, 1] = 1
            ax.imshow(comparison)
            ax.set_title(f'Dice={dice:.3f}', fontsize=9, fontweight='bold')
            ax.axis('off')
            
            row += 1
        
        plt.suptitle(f'Methods C7-C12 Comparison - Sample: {sample_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path2 = os.path.join(SAVE_RESULT_DIR, 'detailed_comparison_C7-C12.png')
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"详细对比图(C7-C12)已保存至: {save_path2}")
        plt.close()

def reconstruction(seed, mask, method='dilation'):
    """灰度重建辅助函数"""
    from skimage.morphology import reconstruction as skimage_reconstruction
    return skimage_reconstruction(seed, mask, method=method)

if __name__ == "__main__":
    processor = LIDC_ComparisonProcessor(DATA_DIR)
    
    # 1. 运行对比测试
    df, samples = processor.run_comparison()
    
    # 2. 选择中等难度的样本进行详细展示
    method_cols = [col for col in df.columns if col.startswith('C')]
    df['avg_dice'] = df[method_cols].mean(axis=1)
    
    medium_samples = df[(df['avg_dice'] > 0.3) & (df['avg_dice'] < 0.7)]
    
    if len(medium_samples) > 0:
        best_idx = (medium_samples['avg_dice'] - 0.5).abs().idxmin()
        selected_sample = samples[best_idx]
        print(f"\n选择样本 {selected_sample['id']} 进行详细展示 (平均Dice={df.loc[best_idx, 'avg_dice']:.3f})")
    else:
        selected_sample = samples[0]
        print(f"\n选择第一个样本进行详细展示")
    
    # 3. 可视化详细对比
    processor.visualize_detailed_comparison(selected_sample, df)
    
    print("\n处理完成！")
    print(f"CSV结果: {CSV_OUTPUT}")
    print(f"统计信息: {os.path.join(SAVE_RESULT_DIR, 'method_statistics.csv')}")
    print(f"可视化结果: {SAVE_RESULT_DIR}")