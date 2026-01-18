<p align="center">
  <img src="assets/logo.png" height="100">
</p>

<h1 align="center">Step-Audio-R1.1</h1>

<p align="center">
  <strong>The First Audio Language Model with Test-Time Compute Scaling</strong>
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

## 🎯 Overview

Step-Audio-R1.1 is a state-of-the-art audio language model that combines **real-time responsiveness** with **strong reasoning capability**. It's the first audio model to successfully unlock test-time compute scaling, surpassing Gemini 2.5 Pro on comprehensive audio benchmarks.

<p align="center">
  <img src="assets/ui-screenshot.png" width="80%" alt="Web UI Screenshot">
</p>

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎙️ **5 Processing Modes** | ASR, Speech-to-Text, Translation, Summarization, Understanding |
| ⚡ **High Concurrency** | Process 4× 85-minute audio files simultaneously |
| 🧠 **Deep Reasoning** | Chain-of-Thought reasoning grounded in acoustic features |
| 🐳 **All-in-One Docker** | Single container with vLLM + Web UI + API |
| 🌐 **Web UI** | Beautiful, responsive interface for audio processing |
| 📡 **REST API** | OpenAI-compatible API for integration |
| 🔧 **Long Audio Support** | Automatic segmentation for audio >85 minutes |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull the all-in-one image
docker pull neosun/step-audio-r1.1:latest

# Run (requires model files mounted)
docker run --gpus all \
  -v /path/to/Step-Audio-R1.1:/model:ro \
  -p 9100:9100 \
  -p 9101:9999 \
  neosun/step-audio-r1.1:latest
```

### Option 2: Docker Compose

```bash
# Clone the repository
git clone https://github.com/neosu/Step-Audio-R1.1.git
cd Step-Audio-R1.1

# Download model (requires ~65GB)
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1

# Start services
docker compose up -d
```

Access the Web UI at: **http://localhost:9100**

---

## 📦 Installation

### Prerequisites

- **GPU**: 4× NVIDIA GPUs with ≥40GB VRAM each (tested on L40S/H100/H800)
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Storage**: ~65GB for model files

### Download Model

```bash
# Method 1: Git LFS
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1

# Method 2: Hugging Face CLI
pip install huggingface_hub
huggingface-cli download stepfun-ai/Step-Audio-R1.1 --local-dir ./Step-Audio-R1.1
```

### Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_PORT` | 9100 | Web UI port |
| `VLLM_PORT` | 9101 | vLLM API port |
| `MODEL_PATH` | ./Step-Audio-R1.1 | Path to model files |
| `TENSOR_PARALLEL_SIZE` | 4 | Number of GPUs |
| `MAX_NUM_SEQS` | 4 | Max concurrent requests |
| `GPU_MEMORY_UTILIZATION` | 0.85 | GPU memory usage |

---

## 🎮 Usage

### Web UI

Navigate to `http://localhost:9100` and:
1. Upload an audio file (WAV, MP3, FLAC, M4A)
2. Select processing mode
3. Click "Process" and wait for results

### API

```bash
# Process audio file
curl -X POST http://localhost:9100/api/process \
  -F "audio=@your_audio.wav" \
  -F "mode=summarize"

# Available modes: asr, s2t, translate, summarize, understand
```

### Python SDK

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:9100/api/process",
        files={"audio": f},
        data={"mode": "understand", "question": "What is discussed?"}
    )
print(response.json()["answer"])
```

### Long Audio Processing

For audio longer than 85 minutes, use the smart processor:

```bash
python smart_audio_processor.py input.wav -m summarize -o output.json -p 4
```

---

## 📊 Benchmark Results

### Processing Modes Comparison (5-minute audio)

| Mode | Time | Output | Best For |
|------|------|--------|----------|
| ASR | 52.8s | 4,496 chars | Precise transcription |
| S2T | 46.8s | 3,713 chars | Structured notes |
| Translate | 51.7s | 1,690 chars | Cross-language |
| Summarize | 26.0s | 1,637 chars | Quick overview |
| Understand | 29.5s | 2,025 chars | Deep analysis |

### Concurrency Test (4× 85-minute audio)

| Mode | Success Rate | Total Time |
|------|--------------|------------|
| All modes | 20/20 ✅ | ~250s each |

### Processing Efficiency

| Audio Length | Efficiency (sec/min) |
|--------------|---------------------|
| 5 min | 5-13 |
| 30 min | 1-3 |
| 60-85 min | 0.5-0.8 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Step-Audio-R1.1 Container              │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────────┐ │
│  │   Web UI    │    │         vLLM Server         │ │
│  │  (Flask)    │───▶│  (Qwen2.5 32B + Audio Enc)  │ │
│  │  Port 9100  │    │        Port 9999            │ │
│  └─────────────┘    └─────────────────────────────┘ │
│         │                       │                   │
│         ▼                       ▼                   │
│  ┌─────────────┐    ┌─────────────────────────────┐ │
│  │  REST API   │    │    4× GPU Tensor Parallel   │ │
│  └─────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Step-Audio-R1.1/
├── app.py                    # Web UI + API server
├── smart_audio_processor.py  # Long audio processor
├── docker-compose.yml        # All-in-one deployment
├── Dockerfile.allinone       # All-in-one image
├── static/                   # Frontend assets
├── templates/                # HTML templates
├── test_audio/               # Test reports
│   ├── FULL_BENCHMARK_REPORT.md
│   └── CONCURRENCY_TEST_REPORT.md
└── assets/                   # Documentation images
```

---

## 🔧 Advanced Configuration

### Custom Docker Compose

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

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism |
| `MAX_MODEL_LEN` | Maximum context length (default: 65536) |
| `MAX_NUM_SEQS` | Maximum concurrent requests |
| `GPU_MEMORY_UTILIZATION` | GPU memory fraction to use |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Changelog

### v1.1.0 (2026-01-18)
- ✨ All-in-One Docker image with vLLM + Web UI
- ✨ Support for 4× concurrent 85-minute audio processing
- ✨ Smart audio processor for unlimited length audio
- ✨ Comprehensive benchmark reports
- 🐛 Fixed ASR truncation issues for long audio
- 📚 Multi-language documentation

### v1.0.0 (2026-01-14)
- 🎉 Initial release of Step-Audio-R1.1
- ✨ 5 processing modes (ASR, S2T, Translate, Summarize, Understand)
- ✨ Web UI and REST API

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [StepFun AI](https://github.com/stepfun-ai) for the original Step-Audio-R1 model
- [vLLM](https://github.com/vllm-project/vllm) for the high-performance inference engine

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosu/Step-Audio-R1.1&type=Date)](https://star-history.com/#neosu/Step-Audio-R1.1)

---

## 📱 Follow Us

<p align="center">
  <img src="https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png" width="200" alt="WeChat">
</p>

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/neosu">NeoSu</a>
</p>
