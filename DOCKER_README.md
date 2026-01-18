# Step-Audio-R1.1 Docker 部署

## 🚀 快速开始

### 前置要求
- 4 张 NVIDIA GPU（每张至少 35GB 显存）
- Docker + nvidia-docker
- 约 70GB 磁盘空间（模型）

### 一键启动
```bash
./start.sh
```

### 分步启动（推荐）

1. **启动 Web UI**（不需要 GPU）
```bash
docker compose -f docker-compose.web.yml up -d
```
访问: http://localhost:9100

2. **启动 vLLM 后端**（需要 4 张 GPU）
```bash
./start-vllm.sh
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `start.sh` | 一键启动脚本（Web + vLLM） |
| `start-vllm.sh` | 单独启动 vLLM 后端 |
| `docker-compose.yml` | 完整服务配置 |
| `docker-compose.web.yml` | 仅 Web UI 配置 |
| `app.py` | Flask Web 服务 |
| `mcp_server.py` | MCP 服务器 |
| `MCP_GUIDE.md` | MCP 使用指南 |

## 🌐 访问方式

### Web UI
- 地址: http://0.0.0.0:9100
- 功能: 上传音频、选择模式、查看结果

### REST API
- 文档: http://0.0.0.0:9100/docs
- 健康检查: `GET /health`
- 处理音频: `POST /api/process`
- 异步任务: `POST /api/task`

### MCP
参见 [MCP_GUIDE.md](MCP_GUIDE.md)

## ⚙️ 配置

### 环境变量
```bash
WEB_PORT=9100          # Web UI 端口
VLLM_PORT=9999         # vLLM API 端口
MODEL_PATH=./Step-Audio-R1.1  # 模型路径
```

### GPU 选择
脚本会自动选择显存最多的 4 张 GPU。如需手动指定：
```bash
export NVIDIA_VISIBLE_DEVICES=0,1,2,3
```

## 🔧 故障排除

### GPU 显存不足
```
ValueError: Free memory on device (X.XX/44.64 GiB) on startup is less than desired GPU memory utilization
```
解决方案：
1. 停止占用 GPU 的其他进程
2. 降低 `--gpu-memory-utilization` 参数
3. 减小 `--max-model-len` 参数

### 查看日志
```bash
docker logs -f step-audio-r1-vllm  # vLLM 日志
docker logs -f step-audio-r1-web   # Web UI 日志
```

## 📦 Docker Hub

构建并推送镜像：
```bash
docker build -t neosun/step-audio-r1:latest .
docker push neosun/step-audio-r1:latest
```

## 📝 License

Apache 2.0
