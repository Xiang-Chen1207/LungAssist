#!/usr/bin/env python3
"""
医学图像肺结节标注辅助工具 - Web版本
基于 Flask 的 Web 应用
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import base64
import io
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================= 配置 =================
IMAGE_SIZE = (128, 128)
MODEL_PATH = "/home/chenx/code/medical_project/20251218_082827_unet_20251218_082827/checkpoints/best_model.pth"
BACKUP_MODEL_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lidc_unet", "checkpoints", "best_model.pth"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "best_model.pth"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.pth"),
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}


# ================= U-Net 模型定义 =================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.0):
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
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv_block(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels, dropout_rate=dropout_rate)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,
                 encoder_filters=None, bridge_filters=1024, dropout_rate=0.3):
        super().__init__()
        if encoder_filters is None:
            encoder_filters = [64, 128, 256, 512]

        self.encoder_filters = encoder_filters
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i, filters in enumerate(encoder_filters):
            current_dropout = dropout_rate * (i / len(encoder_filters))
            self.encoders.append(EncoderBlock(prev_channels, filters, current_dropout))
            prev_channels = filters

        self.bridge = ConvBlock(encoder_filters[-1], bridge_filters, dropout_rate=dropout_rate)

        self.decoders = nn.ModuleList()
        decoder_filters = encoder_filters[::-1]
        prev_channels = bridge_filters
        for i, filters in enumerate(decoder_filters):
            current_dropout = dropout_rate * (1 - i / len(decoder_filters))
            self.decoders.append(DecoderBlock(prev_channels, filters, current_dropout))
            prev_channels = filters

        self.output_conv = nn.Conv2d(encoder_filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)
        x = self.bridge(x)
        skips = skips[::-1]
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        return x


# ================= 图像处理模块 =================
class ImageProcessor:
    """图像预处理模块 - C3方法实现"""

    @staticmethod
    def preprocess_clahe_median(image):
        """CLAHE + 中值滤波"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        denoised = cv2.medianBlur(enhanced, 3)
        return enhanced, denoised

    @staticmethod
    def roi_edge_detection(denoised):
        """边缘检测流水线提取ROI"""
        edges = cv2.Canny(denoised, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = np.zeros_like(denoised, dtype=np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                cv2.drawContours(roi_mask, [contour], -1, 1, -1)
        return edges, roi_mask

    @staticmethod
    def segment_ghpf_otsu(denoised, roi_mask):
        """GHPF + Otsu分割"""
        from skimage import filters, measure

        lowpass = cv2.GaussianBlur(denoised, (15, 15), 3)
        highpass = cv2.subtract(denoised, lowpass)
        highpass = cv2.add(highpass, 127)

        masked = highpass.copy()
        masked[roi_mask == 0] = 0

        if np.sum(roi_mask) == 0:
            return highpass, np.zeros_like(denoised)

        try:
            thresh_val = filters.threshold_otsu(masked[roi_mask > 0])
            binary = (masked > thresh_val).astype(np.uint8)
        except:
            binary = np.zeros_like(denoised)

        labels = measure.label(binary)
        props = measure.regionprops(labels)

        final_mask = np.zeros_like(binary)
        for prop in props:
            if 10 < prop.area < 600:
                if prop.eccentricity < 0.85 and prop.solidity > 0.75:
                    final_mask[labels == prop.label] = 1

        return highpass, final_mask

    @classmethod
    def process_c3(cls, image):
        """C3方法: CLAHE+中值 + 边缘检测 + GHPF+Otsu"""
        enhanced, denoised = cls.preprocess_clahe_median(image)
        edges, roi_mask = cls.roi_edge_detection(denoised)
        filtered, prediction = cls.segment_ghpf_otsu(denoised, roi_mask)
        return {
            'original': image,
            'enhanced': enhanced,
            'denoised': denoised,
            'edges': edges,
            'roi_mask': roi_mask * 255,
            'filtered': filtered,
            'prediction': prediction
        }


# ================= 全局变量 =================
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """加载模型"""
    global model
    model_path = MODEL_PATH

    if not os.path.exists(model_path):
        for backup_path in BACKUP_MODEL_PATHS:
            if os.path.exists(backup_path):
                model_path = backup_path
                break
        else:
            print("警告: 未找到模型权重文件")
            return False

    try:
        model = UNet(in_channels=3, out_channels=1)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"模型加载成功: {model_path}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def numpy_to_base64(img):
    """将numpy数组转换为base64字符串"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_overlay(base_img, mask, color=(255, 0, 0), alpha=0.5):
    """创建叠加图像"""
    if len(base_img.shape) == 2:
        base_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_img.copy()

    mask_bool = mask > 0.5 if mask.max() <= 1 else mask > 127
    overlay = base_rgb.copy()
    overlay[mask_bool] = color

    result = cv2.addWeighted(base_rgb, 1 - alpha, overlay, alpha, 0)
    return result


# ================= 路由 =================
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    """处理图像"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图像'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400

    mode = request.form.get('mode', 'assisted')

    try:
        # 读取图像
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': '无法读取图像'}), 400

        # 调整大小
        img = cv2.resize(img, IMAGE_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = {
            'original': numpy_to_base64(img),
        }

        if mode == 'assisted':
            # C3处理流程
            c3_results = ImageProcessor.process_c3(gray)

            results['enhanced'] = numpy_to_base64(c3_results['enhanced'])
            results['denoised'] = numpy_to_base64(c3_results['denoised'])
            results['edges'] = numpy_to_base64(c3_results['edges'])
            results['roi_mask'] = numpy_to_base64(c3_results['roi_mask'])
            results['filtered'] = numpy_to_base64(c3_results['filtered'])

            # 传统方法预测叠加
            trad_overlay = create_overlay(gray, c3_results['prediction'] * 255, color=(255, 0, 0))
            results['traditional_prediction'] = numpy_to_base64(trad_overlay)

        # U-Net预测
        if model is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred_mask = output.squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

            unet_overlay = create_overlay(gray, pred_mask * 255, color=(0, 255, 0))
            results['unet_prediction'] = numpy_to_base64(unet_overlay)
            results['unet_mask'] = numpy_to_base64(pred_mask * 255)

            # 计算检测统计
            nodule_pixels = np.sum(pred_mask > 0)
            total_pixels = pred_mask.size
            results['stats'] = {
                'nodule_pixels': int(nodule_pixels),
                'total_pixels': int(total_pixels),
                'percentage': round(100 * nodule_pixels / total_pixels, 2)
            }
        else:
            results['unet_prediction'] = None
            results['warning'] = '模型未加载，U-Net预测不可用'

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """获取系统状态"""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    })


# ================= 启动 =================
if __name__ == '__main__':
    print("正在加载模型...")
    load_model()
    print(f"设备: {device}")
    print("启动Web服务器...")
    app.run(host='0.0.0.0', port=5000, debug=True)
