<p align="center">
  <img src="assets/logo.png" height="100">
</p>

<h1 align="center">Step-Audio-R1.1</h1>

<p align="center">
  <strong>首個支援測試時計算擴展的音訊語言模型</strong>
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

## 🎯 專案簡介

Step-Audio-R1.1 是一款先進的音訊語言模型，結合了**即時回應能力**與**強大的推理能力**。它是首個成功實現測試時計算擴展的音訊模型，在綜合音訊基準測試中超越了 Gemini 2.5 Pro。

<p align="center">
  <img src="assets/ui-screenshot.png" width="80%" alt="Web UI 截圖">
</p>

### ✨ 核心特性

| 特性 | 說明 |
|------|------|
| 🎙️ **5種處理模式** | ASR語音辨識、語音轉文字、翻譯、摘要、理解問答 |
| ⚡ **高併發處理** | 同時處理4個85分鐘音訊檔案 |
| 🧠 **深度推理** | 基於聲學特徵的思維鏈推理 |
| 🐳 **一體化Docker** | 單容器包含 vLLM + Web UI + API |
| 🌐 **Web介面** | 美觀、響應式的音訊處理介面 |
| 📡 **REST API** | OpenAI相容的API介面 |
| 🔧 **長音訊支援** | 自動分段處理超過85分鐘的音訊 |

---

## 🚀 快速開始

### 第一步：下載模型（必需，約65GB）

```bash
# 方法1：Git LFS（推薦）
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1

# 方法2：Hugging Face CLI
pip install huggingface_hub
huggingface-cli download stepfun-ai/Step-Audio-R1.1 --local-dir ./Step-Audio-R1.1
```

### 第二步：Docker 執行

```bash
# 拉取一體化映像
docker pull neosun/step-audio-r1.1:latest

# 執行（掛載模型目錄）
docker run --gpus all \
  -v $(pwd)/Step-Audio-R1.1:/model:ro \
  -p 9100:9100 \
  -p 9101:9999 \
  neosun/step-audio-r1.1:latest
```

### 或使用 Docker Compose

```bash
# 複製儲存庫
git clone https://github.com/neosun100/Step-Audio-R1.1.git
cd Step-Audio-R1.1

# 下載模型（如果還沒下載）
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1

# 啟動服務
docker compose up -d
```

存取 Web UI：**http://localhost:9100**

---

## 📦 安裝部署

### 環境需求

- **GPU**：4× NVIDIA GPU，每個≥40GB顯存（已測試 L40S/H100/H800）
- **Docker**：20.10+ 並安裝 NVIDIA Container Toolkit
- **儲存**：約65GB用於模型檔案

### 下載模型

```bash
# 方法1：Git LFS
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1

# 方法2：Hugging Face CLI
pip install huggingface_hub
huggingface-cli download stepfun-ai/Step-Audio-R1.1 --local-dir ./Step-Audio-R1.1
```

### 設定說明

從範本建立 `.env` 檔案：

```bash
cp .env.example .env
```

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `WEB_PORT` | 9100 | Web UI 連接埠 |
| `VLLM_PORT` | 9101 | vLLM API 連接埠 |
| `MODEL_PATH` | ./Step-Audio-R1.1 | 模型檔案路徑 |
| `TENSOR_PARALLEL_SIZE` | 4 | GPU數量 |
| `MAX_NUM_SEQS` | 4 | 最大併發請求數 |
| `GPU_MEMORY_UTILIZATION` | 0.85 | GPU顯存使用率 |

---

## 🎮 使用方法

### Web 介面

存取 `http://localhost:9100`：
1. 上傳音訊檔案（WAV、MP3、FLAC、M4A）
2. 選擇處理模式
3. 點擊「處理」並等待結果

### API 呼叫

```bash
# 處理音訊檔案
curl -X POST http://localhost:9100/api/process \
  -F "audio=@your_audio.wav" \
  -F "mode=summarize"

# 可用模式：asr, s2t, translate, summarize, understand
```

### Python 範例

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:9100/api/process",
        files={"audio": f},
        data={"mode": "understand", "question": "討論了什麼內容？"}
    )
print(response.json()["answer"])
```

### 長音訊處理

對於超過85分鐘的音訊，使用智慧處理器：

```bash
python smart_audio_processor.py input.wav -m summarize -o output.json -p 4
```

---

## 📊 效能基準

### 處理模式比較（5分鐘音訊）

| 模式 | 耗時 | 輸出 | 適用場景 |
|------|------|------|----------|
| ASR | 52.8秒 | 4,496字元 | 精確轉錄 |
| S2T | 46.8秒 | 3,713字元 | 結構化筆記 |
| Translate | 51.7秒 | 1,690字元 | 跨語言理解 |
| Summarize | 26.0秒 | 1,637字元 | 快速概覽 |
| Understand | 29.5秒 | 2,025字元 | 深度分析 |

### 併發測試（4×85分鐘音訊）

| 模式 | 成功率 | 總耗時 |
|------|--------|--------|
| 所有模式 | 20/20 ✅ | 約250秒/組 |

---

## 🤝 參與貢獻

歡迎貢獻程式碼！請隨時提交 Pull Request。

1. Fork 本儲存庫
2. 建立特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📝 更新日誌

### v1.1.0 (2026-01-18)
- ✨ 一體化 Docker 映像，包含 vLLM + Web UI
- ✨ 支援4個85分鐘音訊同時併發處理
- ✨ 智慧音訊處理器，支援任意長度音訊
- ✨ 完整的效能基準測試報告
- 🐛 修復長音訊ASR截斷問題
- 📚 多語言文件支援

### v1.0.0 (2026-01-14)
- 🎉 Step-Audio-R1.1 首次發布
- ✨ 5種處理模式（ASR、S2T、翻譯、摘要、理解）
- ✨ Web UI 和 REST API

---

## 📄 授權條款

本專案採用 Apache License 2.0 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

---

## 🙏 致謝

- [StepFun AI](https://github.com/stepfun-ai) 提供原始 Step-Audio-R1 模型
- [vLLM](https://github.com/vllm-project/vllm) 提供高效能推理引擎

---

## ⭐ Star 歷史

[![Star History Chart](https://api.star-history.com/svg?repos=neosu/Step-Audio-R1.1&type=Date)](https://star-history.com/#neosu/Step-Audio-R1.1)

---

## 📱 關注公眾號

<p align="center">
  <img src="https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png" width="200" alt="微信公眾號">
</p>

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/neosu">NeoSu</a>
</p>
