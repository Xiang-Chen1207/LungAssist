# 肺结节标注辅助工具 - Web版

基于 Flask 的医学图像肺结节标注辅助 Web 应用。

## 功能特点

- **拖拽上传**: 支持拖拽图片或点击选择文件
- **两种标记模式**:
  - **辅助标记**: 显示完整 C3 处理流程 (CLAHE + 中值滤波 + 边缘检测 + GHPF + Otsu)
  - **直接标记**: 仅显示 U-Net 模型预测结果
- **实时处理**: 快速图像处理和预测
- **统计信息**: 显示结节检测统计数据

## 安装依赖

```bash
cd web_app
pip install -r requirements.txt
```

## 运行方式

```bash
python app.py
```

服务器将在 `http://0.0.0.0:5000` 启动。

## 访问地址

- 本地访问: http://localhost:5000
- 局域网访问: http://<服务器IP>:5000

## 模型配置

默认模型路径配置在 `app.py` 中的 `MODEL_PATH` 变量。如果模型文件位置不同，请修改此路径。

## 技术栈

- **后端**: Flask
- **前端**: HTML5 + CSS3 + JavaScript
- **深度学习**: PyTorch (U-Net)
- **图像处理**: OpenCV, scikit-image

## API 接口

### POST /process
处理上传的图像

**参数**:
- `image`: 图像文件
- `mode`: 处理模式 (`assisted` 或 `direct`)

**返回**: JSON 格式的处理结果，包含 base64 编码的图像

### GET /status
获取系统状态

**返回**: 模型加载状态、设备信息
