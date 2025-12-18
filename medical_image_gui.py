#!/usr/bin/env python3
"""
医学图像肺结节标注辅助工具
提供拖拽图片功能，支持辅助标记和直接标记两种模式
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QScrollArea, QFrame, QStatusBar, QProgressBar,
    QButtonGroup, QRadioButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, QPoint, QRect, QSize
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont,
    QDragEnterEvent, QDropEvent, QPalette, QBrush
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lidc_unet'))

# ================= 配置 =================
IMAGE_SIZE = (128, 128)
# 模型权重文件路径 - 可以根据实际情况修改
MODEL_PATH = "/home/chenx/code/medical_project/20251218_082827_unet_20251218_082827/checkpoints/best_model.pth"
# 备用模型路径
BACKUP_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "lidc_unet", "checkpoints", "best_model.pth"),
    os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth"),
    os.path.join(os.path.dirname(__file__), "best_model.pth"),
]


# ================= U-Net 模型定义 =================
class ConvBlock(nn.Module):
    """卷积块"""
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
    """编码器块"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv_block(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """解码器块"""
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
    """U-Net模型"""
    def __init__(self, in_channels=3, out_channels=1,
                 encoder_filters=None, bridge_filters=1024, dropout_rate=0.3):
        super().__init__()
        if encoder_filters is None:
            encoder_filters = [64, 128, 256, 512]

        self.encoder_filters = encoder_filters

        # 编码器
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i, filters in enumerate(encoder_filters):
            current_dropout = dropout_rate * (i / len(encoder_filters))
            self.encoders.append(EncoderBlock(prev_channels, filters, current_dropout))
            prev_channels = filters

        # 桥接层
        self.bridge = ConvBlock(encoder_filters[-1], bridge_filters, dropout_rate=dropout_rate)

        # 解码器
        self.decoders = nn.ModuleList()
        decoder_filters = encoder_filters[::-1]
        prev_channels = bridge_filters
        for i, filters in enumerate(decoder_filters):
            current_dropout = dropout_rate * (1 - i / len(decoder_filters))
            self.decoders.append(DecoderBlock(prev_channels, filters, current_dropout))
            prev_channels = filters

        # 输出层
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


# ================= 可绘制的图像标签 =================
class DrawableImageLabel(QLabel):
    """支持绘制的图像标签"""
    roi_drawn = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #2a2a2a;")

        self.drawing = False
        self.drawing_enabled = False
        self.start_point = None
        self.end_point = None
        self.roi_rect = None
        self.original_pixmap = None

    def enable_drawing(self, enable=True):
        """启用/禁用绘制功能"""
        self.drawing_enabled = enable
        if enable:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_image(self, pixmap):
        """设置图像"""
        self.original_pixmap = pixmap
        self.roi_rect = None
        self.update_display()

    def update_display(self):
        """更新显示"""
        if self.original_pixmap is None:
            return

        display_pixmap = self.original_pixmap.copy()

        if self.roi_rect is not None:
            painter = QPainter(display_pixmap)
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self.roi_rect)
            painter.end()

        scaled = display_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update_roi_rect()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            self.update_roi_rect()
            if self.roi_rect is not None:
                self.roi_drawn.emit(self.roi_rect)

    def update_roi_rect(self):
        if self.start_point and self.end_point:
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())

            # 转换到图像坐标
            if self.original_pixmap:
                scale_x = self.original_pixmap.width() / self.width()
                scale_y = self.original_pixmap.height() / self.height()
                self.roi_rect = QRect(
                    int(x1 * scale_x), int(y1 * scale_y),
                    int((x2 - x1) * scale_x), int((y2 - y1) * scale_y)
                )
                self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()


# ================= 拖放区域 =================
class DropArea(QLabel):
    """支持拖放的区域"""
    image_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setText("拖拽图片到此处\n或点击选择文件")
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #666;
                border-radius: 10px;
                background-color: #1e1e1e;
                color: #888;
                font-size: 16px;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #2a2a2a;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 3px dashed #4CAF50;
                    border-radius: 10px;
                    background-color: #2a3a2a;
                    color: #4CAF50;
                    font-size: 16px;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #666;
                border-radius: 10px;
                background-color: #1e1e1e;
                color: #888;
                font-size: 16px;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                self.image_dropped.emit(file_path)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #666;
                border-radius: 10px;
                background-color: #1e1e1e;
                color: #888;
                font-size: 16px;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择医学图像",
                "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if file_path:
                self.image_dropped.emit(file_path)


# ================= 主窗口 =================
class MedicalImageGUI(QMainWindow):
    """医学图像标注辅助工具主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("肺结节标注辅助工具 - LIDC-IDRI")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(self._get_dark_style())

        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processing_results = {}

        # 初始化UI
        self._init_ui()
        self._load_model()

    def _get_dark_style(self):
        """获取暗色主题样式"""
        return """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #4CAF50;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
            QRadioButton {
                spacing: 8px;
                font-size: 14px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QStatusBar {
                background-color: #2a2a2a;
                border-top: 1px solid #444;
            }
            QScrollArea {
                border: none;
            }
        """

    def _init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # 顶部控制区
        control_group = QGroupBox("操作控制")
        control_layout = QHBoxLayout(control_group)

        # 拖放区域
        self.drop_area = DropArea()
        self.drop_area.image_dropped.connect(self.load_image)
        self.drop_area.setFixedSize(200, 120)
        control_layout.addWidget(self.drop_area)

        # 模式选择
        mode_group = QGroupBox("标记模式")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_group = QButtonGroup()

        self.assisted_radio = QRadioButton("辅助标记 (C3处理流程)")
        self.assisted_radio.setChecked(True)
        self.direct_radio = QRadioButton("直接标记 (U-Net模型)")

        self.mode_group.addButton(self.assisted_radio, 1)
        self.mode_group.addButton(self.direct_radio, 2)

        mode_layout.addWidget(self.assisted_radio)
        mode_layout.addWidget(self.direct_radio)
        control_layout.addWidget(mode_group)

        # 操作按钮
        btn_layout = QVBoxLayout()

        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
        """)

        self.clear_btn = QPushButton("清除")
        self.clear_btn.clicked.connect(self.clear_all)

        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)

        btn_layout.addWidget(self.process_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.save_btn)

        control_layout.addLayout(btn_layout)
        control_layout.addStretch()

        # 模型状态
        self.model_status = QLabel("模型状态: 加载中...")
        self.model_status.setStyleSheet("color: #FFC107; padding: 10px;")
        control_layout.addWidget(self.model_status)

        main_layout.addWidget(control_group)

        # 图像显示区域
        display_splitter = QSplitter(Qt.Horizontal)

        # 左侧：输入和ROI绘制
        left_group = QGroupBox("输入图像 (可在辅助模式下圈选ROI)")
        left_layout = QVBoxLayout(left_group)
        self.input_image_label = DrawableImageLabel()
        self.input_image_label.roi_drawn.connect(self.on_roi_drawn)
        left_layout.addWidget(self.input_image_label)
        display_splitter.addWidget(left_group)

        # 中间：处理步骤
        middle_group = QGroupBox("C3处理流程")
        middle_layout = QVBoxLayout(middle_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.steps_layout = QHBoxLayout(scroll_content)
        self.steps_layout.setSpacing(10)

        # 创建步骤显示标签
        self.step_labels = {}
        step_names = ['CLAHE增强', '中值滤波', '边缘检测', 'ROI区域', 'GHPF滤波', '传统预测']
        for name in step_names:
            step_widget = QWidget()
            step_vlayout = QVBoxLayout(step_widget)
            step_vlayout.setSpacing(5)

            label = QLabel()
            label.setFixedSize(150, 150)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid #444; background-color: #2a2a2a;")

            title = QLabel(name)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 11px; color: #aaa;")

            step_vlayout.addWidget(label)
            step_vlayout.addWidget(title)

            self.step_labels[name] = label
            self.steps_layout.addWidget(step_widget)

        scroll_area.setWidget(scroll_content)
        middle_layout.addWidget(scroll_area)
        display_splitter.addWidget(middle_group)

        # 右侧：模型预测结果
        right_group = QGroupBox("U-Net模型预测结果")
        right_layout = QVBoxLayout(right_group)
        self.result_label = QLabel()
        self.result_label.setMinimumSize(256, 256)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #2a2a2a;")
        right_layout.addWidget(self.result_label)

        # 结果信息
        self.result_info = QLabel("等待处理...")
        self.result_info.setAlignment(Qt.AlignCenter)
        self.result_info.setStyleSheet("color: #aaa; padding: 10px;")
        right_layout.addWidget(self.result_info)

        display_splitter.addWidget(right_group)

        # 设置分割比例
        display_splitter.setSizes([400, 600, 400])

        main_layout.addWidget(display_splitter)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪 - 请拖入图像或点击选择文件")

    def _load_model(self):
        """加载U-Net模型"""
        model_path = MODEL_PATH

        # 检查主路径
        if not os.path.exists(model_path):
            # 尝试备用路径
            for backup_path in BACKUP_MODEL_PATHS:
                if os.path.exists(backup_path):
                    model_path = backup_path
                    break
            else:
                self.model_status.setText("模型状态: 未找到模型文件")
                self.model_status.setStyleSheet("color: #f44336; padding: 10px;")
                QMessageBox.warning(
                    self, "警告",
                    f"未找到模型权重文件:\n{MODEL_PATH}\n\n"
                    "直接标记功能将不可用。\n"
                    "请将模型文件放置到正确位置。"
                )
                return

        try:
            self.model = UNet(in_channels=3, out_channels=1)
            checkpoint = torch.load(model_path, map_location=self.device)

            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            self.model_status.setText(f"模型状态: 已加载 ({self.device})")
            self.model_status.setStyleSheet("color: #4CAF50; padding: 10px;")
            self.statusBar.showMessage(f"模型加载成功 - {model_path}")

        except Exception as e:
            self.model_status.setText(f"模型状态: 加载失败")
            self.model_status.setStyleSheet("color: #f44336; padding: 10px;")
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")

    def load_image(self, file_path):
        """加载图像"""
        try:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)

            if self.current_image is None:
                raise ValueError("无法读取图像文件")

            # 调整大小
            self.current_image = cv2.resize(self.current_image, IMAGE_SIZE)

            # 转换为QPixmap显示
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # 显示在输入标签
            self.input_image_label.set_image(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

            # 更新状态
            self.process_btn.setEnabled(True)
            self.statusBar.showMessage(f"已加载图像: {os.path.basename(file_path)}")

            # 清除之前的处理结果
            self.clear_results()

            # 根据模式启用绘制
            if self.assisted_radio.isChecked():
                self.input_image_label.enable_drawing(True)
            else:
                self.input_image_label.enable_drawing(False)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败:\n{str(e)}")

    def process_image(self):
        """处理图像"""
        if self.current_image is None:
            return

        self.statusBar.showMessage("处理中...")
        QApplication.processEvents()

        try:
            if self.assisted_radio.isChecked():
                self._process_assisted()
            else:
                self._process_direct()

            self.save_btn.setEnabled(True)
            self.statusBar.showMessage("处理完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败:\n{str(e)}")
            self.statusBar.showMessage("处理失败")

    def _process_assisted(self):
        """辅助标记模式处理"""
        # 获取灰度图像
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        # C3处理流程
        results = ImageProcessor.process_c3(gray)
        self.processing_results = results

        # 显示各步骤结果
        step_mapping = {
            'CLAHE增强': 'enhanced',
            '中值滤波': 'denoised',
            '边缘检测': 'edges',
            'ROI区域': 'roi_mask',
            'GHPF滤波': 'filtered',
            '传统预测': 'prediction'
        }

        for step_name, result_key in step_mapping.items():
            if result_key in results:
                img = results[result_key]
                if result_key == 'prediction':
                    # 预测结果叠加显示
                    overlay = self._create_overlay(gray, img * 255, color=(255, 0, 0))
                    self._display_on_label(self.step_labels[step_name], overlay)
                else:
                    self._display_on_label(self.step_labels[step_name], img)

        # 同时进行U-Net预测
        if self.model is not None:
            self._run_unet_prediction()

        self.result_info.setText("辅助标记完成\n可在左侧图像上圈选ROI区域")

    def _process_direct(self):
        """直接标记模式处理"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，无法使用直接标记功能")
            return

        self._run_unet_prediction()
        self.result_info.setText("U-Net模型预测完成\n红色区域为检测到的结节")

    def _run_unet_prediction(self):
        """运行U-Net预测"""
        # 准备输入
        img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_mask = output.squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # 存储结果
        self.processing_results['unet_prediction'] = pred_mask

        # 显示结果
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        overlay = self._create_overlay(gray, pred_mask * 255, color=(0, 255, 0))
        self._display_on_label(self.result_label, overlay, size=256)

    def _create_overlay(self, base_img, mask, color=(255, 0, 0), alpha=0.5):
        """创建叠加图像"""
        if len(base_img.shape) == 2:
            base_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = base_img.copy()

        mask_bool = mask > 127 if mask.max() > 1 else mask > 0.5
        overlay = base_rgb.copy()
        overlay[mask_bool] = color

        result = cv2.addWeighted(base_rgb, 1 - alpha, overlay, alpha, 0)
        return result

    def _display_on_label(self, label, img, size=150):
        """在标签上显示图像"""
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img

        h, w = img_rgb.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(size, size, Qt.KeepAspectRatio))

    def on_roi_drawn(self, rect):
        """ROI绘制完成回调"""
        if not self.assisted_radio.isChecked():
            return

        self.statusBar.showMessage(f"ROI区域: ({rect.x()}, {rect.y()}) - {rect.width()}x{rect.height()}")

        # 可以在这里添加基于ROI的进一步处理
        if self.model is not None and 'unet_prediction' in self.processing_results:
            # 高亮ROI区域内的预测结果
            pred = self.processing_results['unet_prediction']
            roi_pred = pred[rect.y():rect.y()+rect.height(),
                           rect.x():rect.x()+rect.width()]
            nodule_pixels = np.sum(roi_pred > 0)
            total_pixels = roi_pred.size

            self.result_info.setText(
                f"ROI区域分析:\n"
                f"结节像素: {nodule_pixels}\n"
                f"总像素: {total_pixels}\n"
                f"占比: {100*nodule_pixels/total_pixels:.1f}%"
            )

    def clear_results(self):
        """清除处理结果"""
        for label in self.step_labels.values():
            label.clear()
            label.setStyleSheet("border: 1px solid #444; background-color: #2a2a2a;")

        self.result_label.clear()
        self.result_info.setText("等待处理...")
        self.processing_results = {}
        self.save_btn.setEnabled(False)

    def clear_all(self):
        """清除所有"""
        self.current_image = None
        self.current_image_path = None
        self.input_image_label.set_image(None)
        self.input_image_label.clear()
        self.clear_results()
        self.process_btn.setEnabled(False)
        self.statusBar.showMessage("已清除所有内容")

    def save_result(self):
        """保存结果"""
        if not self.processing_results:
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return

        try:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

            # 保存各步骤结果
            for key, img in self.processing_results.items():
                if isinstance(img, np.ndarray):
                    save_path = os.path.join(save_dir, f"{base_name}_{key}.png")
                    cv2.imwrite(save_path, img if img.max() <= 1 else img)

            # 保存叠加结果
            if 'unet_prediction' in self.processing_results:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                overlay = self._create_overlay(gray, self.processing_results['unet_prediction'] * 255)
                save_path = os.path.join(save_dir, f"{base_name}_result_overlay.png")
                cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            QMessageBox.information(self, "成功", f"结果已保存到:\n{save_dir}")
            self.statusBar.showMessage(f"结果已保存到: {save_dir}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = MedicalImageGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
