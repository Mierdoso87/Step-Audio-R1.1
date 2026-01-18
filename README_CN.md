<p align="center">
  <img src="assets/logo.png" height="100">
</p>

<h1 align="center">Step-Audio-R1.1</h1>

<p align="center">
  <strong>首个支持测试时计算扩展的音频语言模型</strong>
</p>

<p align="center">
  <a href="README.md">English</a> | 
  <a href="README_CN.md">简体中文</a> | 
  <a href="README_TW.md">繁體中文</a> | 
  <a href="README_JP.md">日本語</a>
</p>

<p align="center">
  <a href="https://hub.docker.com/r/neosun/step-audio-r1.1"><img src="https://img.shields.io/docker/pulls/neosun/step-audio-r1.1?style=flat-square&logo=docker" alt="Docker Pulls"></a>
  <a href="https://github.com/neosu/Step-Audio-R1.1/stargazers"><img src="https://img.shields.io/github/stars/neosu/Step-Audio-R1.1?style=flat-square&logo=github" alt="Stars"></a>
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-R1.1"><img src="https://img.shields.io/badge/🤗-Model-yellow?style=flat-square" alt="HuggingFace"></a>
  <a href="https://arxiv.org/pdf/2511.15848"><img src="https://img.shields.io/badge/📄-Paper-red?style=flat-square" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License"></a>
</p>

---

## 🎯 项目简介

Step-Audio-R1.1 是一款先进的音频语言模型，结合了**实时响应能力**与**强大的推理能力**。它是首个成功实现测试时计算扩展的音频模型，在综合音频基准测试中超越了 Gemini 2.5 Pro。

<p align="center">
  <img src="assets/ui-screenshot.png" width="80%" alt="Web UI 截图">
</p>

### ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🎙️ **5种处理模式** | ASR语音识别、语音转文字、翻译、摘要、理解问答 |
| ⚡ **高并发处理** | 同时处理4个85分钟音频文件 |
| 🧠 **深度推理** | 基于声学特征的思维链推理 |
| 🐳 **一体化Docker** | 单容器包含 vLLM + Web UI + API |
| 🌐 **Web界面** | 美观、响应式的音频处理界面 |
| 📡 **REST API** | OpenAI兼容的API接口 |
| 🔧 **长音频支持** | 自动分段处理超过85分钟的音频 |

---

## 🚀 快速开始

### 方式一：Docker（推荐）

```bash
# 拉取一体化镜像
docker pull neosun/step-audio-r1.1:latest

# 运行（需要挂载模型文件）
docker run --gpus all \
  -v /path/to/Step-Audio-R1.1:/model:ro \
  -p 9100:9100 \
  -p 9101:9999 \
  neosun/step-audio-r1.1:latest
```

### 方式二：Docker Compose

```bash
# 克隆仓库
git clone https://github.com/neosu/Step-Audio-R1.1.git
cd Step-Audio-R1.1

# 下载模型（约65GB）
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1

# 启动服务
docker compose up -d
```

访问 Web UI：**http://localhost:9100**

---

## 📦 安装部署

### 环境要求

- **GPU**：4× NVIDIA GPU，每个≥40GB显存（已测试 L40S/H100/H800）
- **Docker**：20.10+ 并安装 NVIDIA Container Toolkit
- **存储**：约65GB用于模型文件

### 下载模型

```bash
# 方法1：Git LFS
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1

# 方法2：Hugging Face CLI
pip install huggingface_hub
huggingface-cli download stepfun-ai/Step-Audio-R1.1 --local-dir ./Step-Audio-R1.1
```

### 配置说明

从模板创建 `.env` 文件：

```bash
cp .env.example .env
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WEB_PORT` | 9100 | Web UI 端口 |
| `VLLM_PORT` | 9101 | vLLM API 端口 |
| `MODEL_PATH` | ./Step-Audio-R1.1 | 模型文件路径 |
| `TENSOR_PARALLEL_SIZE` | 4 | GPU数量 |
| `MAX_NUM_SEQS` | 4 | 最大并发请求数 |
| `GPU_MEMORY_UTILIZATION` | 0.85 | GPU显存使用率 |

---

## 🎮 使用方法

### Web 界面

访问 `http://localhost:9100`：
1. 上传音频文件（WAV、MP3、FLAC、M4A）
2. 选择处理模式
3. 点击"处理"并等待结果

### API 调用

```bash
# 处理音频文件
curl -X POST http://localhost:9100/api/process \
  -F "audio=@your_audio.wav" \
  -F "mode=summarize"

# 可用模式：asr, s2t, translate, summarize, understand
```

### Python 示例

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:9100/api/process",
        files={"audio": f},
        data={"mode": "understand", "question": "讨论了什么内容？"}
    )
print(response.json()["answer"])
```

### 长音频处理

对于超过85分钟的音频，使用智能处理器：

```bash
python smart_audio_processor.py input.wav -m summarize -o output.json -p 4
```

---

## 📊 性能基准

### 处理模式对比（5分钟音频）

| 模式 | 耗时 | 输出 | 适用场景 |
|------|------|------|----------|
| ASR | 52.8秒 | 4,496字符 | 精确转录 |
| S2T | 46.8秒 | 3,713字符 | 结构化笔记 |
| Translate | 51.7秒 | 1,690字符 | 跨语言理解 |
| Summarize | 26.0秒 | 1,637字符 | 快速概览 |
| Understand | 29.5秒 | 2,025字符 | 深度分析 |

### 并发测试（4×85分钟音频）

| 模式 | 成功率 | 总耗时 |
|------|--------|--------|
| 所有模式 | 20/20 ✅ | 约250秒/组 |

### 处理效率

| 音频长度 | 效率（秒/分钟） |
|----------|-----------------|
| 5分钟 | 5-13 |
| 30分钟 | 1-3 |
| 60-85分钟 | 0.5-0.8 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│              Step-Audio-R1.1 容器                    │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────────┐ │
│  │   Web UI    │    │         vLLM 服务器          │ │
│  │  (Flask)    │───▶│  (Qwen2.5 32B + 音频编码器)  │ │
│  │  端口 9100  │    │        端口 9999             │ │
│  └─────────────┘    └─────────────────────────────┘ │
│         │                       │                   │
│         ▼                       ▼                   │
│  ┌─────────────┐    ┌─────────────────────────────┐ │
│  │  REST API   │    │    4× GPU 张量并行           │ │
│  └─────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
Step-Audio-R1.1/
├── app.py                    # Web UI + API 服务器
├── smart_audio_processor.py  # 长音频处理器
├── docker-compose.yml        # 一体化部署配置
├── Dockerfile.allinone       # 一体化镜像
├── static/                   # 前端资源
├── templates/                # HTML 模板
├── test_audio/               # 测试报告
│   ├── FULL_BENCHMARK_REPORT.md
│   └── CONCURRENCY_TEST_REPORT.md
└── assets/                   # 文档图片
```

---

## 🔧 高级配置

### 自定义 Docker Compose

```yaml
services:
  step-audio:
    image: neosun/step-audio-r1.1:latest
    runtime: nvidia
    environment:
      - TENSOR_PARALLEL_SIZE=4
      - MAX_NUM_SEQS=4
      - GPU_MEMORY_UTILIZATION=0.85
    volumes:
      - ./Step-Audio-R1.1:/model:ro
    ports:
      - "9100:9100"
      - "9101:9999"
```

### 环境变量说明

| 变量 | 说明 |
|------|------|
| `TENSOR_PARALLEL_SIZE` | 张量并行GPU数量 |
| `MAX_MODEL_LEN` | 最大上下文长度（默认：65536） |
| `MAX_NUM_SEQS` | 最大并发请求数 |
| `GPU_MEMORY_UTILIZATION` | GPU显存使用比例 |

---

## 🤝 参与贡献

欢迎贡献代码！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📝 更新日志

### v1.1.0 (2026-01-18)
- ✨ 一体化 Docker 镜像，包含 vLLM + Web UI
- ✨ 支持4个85分钟音频同时并发处理
- ✨ 智能音频处理器，支持任意长度音频
- ✨ 完整的性能基准测试报告
- 🐛 修复长音频ASR截断问题
- 📚 多语言文档支持

### v1.0.0 (2026-01-14)
- 🎉 Step-Audio-R1.1 首次发布
- ✨ 5种处理模式（ASR、S2T、翻译、摘要、理解）
- ✨ Web UI 和 REST API

---

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [StepFun AI](https://github.com/stepfun-ai) 提供原始 Step-Audio-R1 模型
- [vLLM](https://github.com/vllm-project/vllm) 提供高性能推理引擎

---

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=neosu/Step-Audio-R1.1&type=Date)](https://star-history.com/#neosu/Step-Audio-R1.1)

---

## 📱 关注公众号

<p align="center">
  <img src="https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png" width="200" alt="微信公众号">
</p>

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/neosu">NeoSu</a>
</p>
