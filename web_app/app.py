#!/usr/bin/env python3
"""
医学图像肺结节标注辅助工具 - v3.0 (含处理流可视化)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import base64
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ================= 配置 =================
DISPLAY_SIZE = (512, 512) 
MODEL_INPUT_SIZE = (128, 128)
MODEL_PATH = "/home/chenx/code/medical_project/outputs/20251218_082827_unet_20251218_082827/checkpoints/best_model.pth"
# 备用路径逻辑保持不变...
BACKUP_MODEL_PATHS = ["best_model.pth"] 

# ================= 模型定义 (UNet) =================
# ... (保持之前的 UNet 类定义不变，为了节省篇幅，此处省略，请保留原本的 Class 代码) ...
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout: x = self.dropout(x)
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
    def __init__(self, in_channels=3, out_channels=1, encoder_filters=None, bridge_filters=1024, dropout_rate=0.3):
        super().__init__()
        if encoder_filters is None: encoder_filters = [64, 128, 256, 512]
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i, filters in enumerate(encoder_filters):
            self.encoders.append(EncoderBlock(prev_channels, filters, dropout_rate * (i / len(encoder_filters))))
            prev_channels = filters
        self.bridge = ConvBlock(encoder_filters[-1], bridge_filters, dropout_rate=dropout_rate)
        self.decoders = nn.ModuleList()
        decoder_filters = encoder_filters[::-1]
        prev_channels = bridge_filters
        for i, filters in enumerate(decoder_filters):
            self.decoders.append(DecoderBlock(prev_channels, filters, dropout_rate * (1 - i / len(decoder_filters))))
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

# ================= 辅助函数 =================
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model
    # (加载模型逻辑同上，省略以节省空间，请确保此处有完整的 load_model 函数)
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        for backup_path in BACKUP_MODEL_PATHS:
            if os.path.exists(backup_path):
                model_path = backup_path
                break
        else:
            return False
    try:
        net = UNet(in_channels=3, out_channels=1)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        model = net
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def numpy_to_base64_png(img, ensure_rgb=True):
    """转Base64，如果图是单通道灰度，可视情况转RGB"""
    if ensure_rgb and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def generate_process_steps(img_bgr):
    """
    生成传统的预处理步骤图像，用于前端展示
    """
    steps = {}
    
    # 1. 灰度化
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    steps['1_Gray'] = numpy_to_base64_png(cv2.resize(gray, DISPLAY_SIZE))
    
    # 2. CLAHE 限制对比度自适应直方图均衡化 (增强肺纹理)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    steps['2_CLAHE'] = numpy_to_base64_png(cv2.resize(enhanced, DISPLAY_SIZE))
    
    # 3. 中值滤波 (去噪)
    denoised = cv2.medianBlur(enhanced, 3)
    steps['3_Denoise'] = numpy_to_base64_png(cv2.resize(denoised, DISPLAY_SIZE))
    
    # 4. 边缘检测 (Canny) - 帮助医生看清结节边界
    edges = cv2.Canny(denoised, 50, 150)
    # 反色显示边缘（白底黑线）在网页上更好看，或者保持黑底白线
    steps['4_Edges'] = numpy_to_base64_png(cv2.resize(edges, DISPLAY_SIZE))
    
    # 5. 阈值分割 (粗略ROI)
    _, binary = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY)
    steps['5_Threshold'] = numpy_to_base64_png(cv2.resize(binary, DISPLAY_SIZE))
    
    return steps

# ================= 路由 =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files: return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. 准备显示用的原图
        img_display = cv2.resize(img_original, DISPLAY_SIZE)
        
        results = {
            'display_image': numpy_to_base64_png(img_display),
            'steps': generate_process_steps(img_original) # 生成处理流
        }

        # 2. U-Net 推理
        if model is not None:
            img_input = cv2.resize(img_original, MODEL_INPUT_SIZE)
            img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred_mask = output.squeeze().cpu().numpy()
                pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)

            # 3. 生成透明叠加层 (绿色)
            mask_resized = cv2.resize(pred_mask_binary, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
            overlay_rgba = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 4), dtype=np.uint8)
            # 绿色, Alpha=100
            overlay_rgba[mask_resized == 1] = [0, 255, 0, 100] 
            
            results['ai_mask_png'] = numpy_to_base64_png(overlay_rgba)
            
            nodule_pixels = np.sum(pred_mask_binary > 0)
            results['stats'] = {
                'detected': bool(nodule_pixels > 0),
                'confidence': 'High' if nodule_pixels > 50 else 'Low'
            }
        else:
            results['error'] = 'Model not loaded'

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5010, debug=True)