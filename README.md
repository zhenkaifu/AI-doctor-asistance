# 实时语音识别与说话人分离项目

本项目由三部分组成：前端 Vue 3、后端 NestJS 中继、以及 Python ASR 服务。

## 1. Python ASR 服务 (Port: 8001)
负责使用 `faster-whisper` 进行语音识别，并集成了 `pyannote.audio` 结构进行说话人分离。

### 安装依赖
```bash
cd python-asr
pip install -r requirements.txt
```

### 运行
```bash
python3 main.py
```

## 2. NestJS 后端 (Port: 3000)
作为业务中继，管理前端 WebSocket 连接并流式转发音频。

### 安装依赖
```bash
cd backend
npm install
```

### 运行
```bash
npm run start:dev
```

## 3. Vue 3 前端 (Port: 5173)
实时采集麦克风，显示带说话人标签的字幕。

### 安装依赖
```bash
cd frontend
npm install
```

### 运行
```bash
npm run dev
```

## 注意事项
- **Mac M2 优化**：Python 服务默认使用 `cpu` + `int8` 运行 `faster-whisper` 的 `small` 模型，这在 M2 上非常流畅。
- **说话人分离**：`pyannote.audio` 的实时分离逻辑在 `main.py` 中留有接口。如需开启完整功能，请在 `main.py` 中填入你的 HuggingFace Token 并启用对应代码。
- **音频格式**：前端自动将音频转换为 16kHz Int16 格式发送。
