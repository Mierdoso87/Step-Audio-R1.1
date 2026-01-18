#!/bin/bash
# Step-Audio-R1.1 一键启动脚本
# 可在任意路径下运行，自动检测模型位置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 默认配置
IMAGE="neosun/step-audio-r1.1:latest"
CONTAINER_NAME="step-audio-r1.1"
WEB_PORT=${WEB_PORT:-9100}
VLLM_PORT=${VLLM_PORT:-9101}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}

# 模型搜索路径（按优先级）
MODEL_SEARCH_PATHS=(
    "./Step-Audio-R1.1"
    "../Step-Audio-R1.1"
    "$HOME/Step-Audio-R1.1"
    "$HOME/models/Step-Audio-R1.1"
    "/data/Step-Audio-R1.1"
    "/models/Step-Audio-R1.1"
    "/home/neo/upload/Step-Audio-R1/Step-Audio-R1.1"
)

echo -e "${GREEN}=============================================="
echo "  Step-Audio-R1.1 一键启动脚本"
echo -e "==============================================${NC}"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    exit 1
fi

# 检查 NVIDIA Docker
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "${YELLOW}警告: NVIDIA Container Toolkit 可能未安装${NC}"
fi

# 检查容器是否已运行
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✓ 容器已在运行${NC}"
    echo "  Web UI: http://localhost:${WEB_PORT}"
    echo "  vLLM API: http://localhost:${VLLM_PORT}"
    exit 0
fi

# 检查容器是否存在但已停止
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}发现已停止的容器，正在重启...${NC}"
    docker start ${CONTAINER_NAME}
    echo -e "${GREEN}✓ 容器已重启${NC}"
    echo "  Web UI: http://localhost:${WEB_PORT}"
    exit 0
fi

# 查找模型路径
MODEL_PATH=""
echo "正在搜索模型文件..."

# 首先检查环境变量
if [ -n "$MODEL_PATH_ENV" ] && [ -d "$MODEL_PATH_ENV" ]; then
    MODEL_PATH="$MODEL_PATH_ENV"
    echo -e "${GREEN}✓ 使用环境变量指定的模型路径: $MODEL_PATH${NC}"
fi

# 搜索预定义路径
if [ -z "$MODEL_PATH" ]; then
    for path in "${MODEL_SEARCH_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/config.json" ]; then
            MODEL_PATH="$(cd "$path" && pwd)"
            echo -e "${GREEN}✓ 找到模型: $MODEL_PATH${NC}"
            break
        fi
    done
fi

# 如果还没找到，搜索常见位置
if [ -z "$MODEL_PATH" ]; then
    echo "在常见位置搜索..."
    FOUND=$(find /home /data /models -maxdepth 4 -type d -name "Step-Audio-R1.1" 2>/dev/null | head -1)
    if [ -n "$FOUND" ] && [ -f "$FOUND/config.json" ]; then
        MODEL_PATH="$FOUND"
        echo -e "${GREEN}✓ 找到模型: $MODEL_PATH${NC}"
    fi
fi

# 未找到模型
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 未找到模型文件${NC}"
    echo ""
    echo "请先下载模型 (~65GB):"
    echo "  git lfs install"
    echo "  git clone https://huggingface.co/stepfun-ai/Step-Audio-R1.1"
    echo ""
    echo "或指定模型路径:"
    echo "  MODEL_PATH_ENV=/path/to/Step-Audio-R1.1 $0"
    exit 1
fi

# 拉取镜像（如果不存在）
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE}$"; then
    echo "正在拉取 Docker 镜像..."
    docker pull ${IMAGE}
fi

# 启动容器
echo ""
echo "正在启动容器..."
echo "  模型路径: $MODEL_PATH"
echo "  GPU数量: $TENSOR_PARALLEL_SIZE"
echo "  并发数: $MAX_NUM_SEQS"

docker run -d \
    --gpus all \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
    -e TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} \
    -e MAX_NUM_SEQS=${MAX_NUM_SEQS} \
    -e GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} \
    -v "${MODEL_PATH}:/model:ro" \
    -v /tmp/step-audio-r1:/tmp/step-audio-r1 \
    -p ${WEB_PORT}:9100 \
    -p ${VLLM_PORT}:9999 \
    ${IMAGE}

echo ""
echo -e "${GREEN}=============================================="
echo "  ✓ 启动成功！"
echo "=============================================="
echo ""
echo "  Web UI:   http://localhost:${WEB_PORT}"
echo "  vLLM API: http://localhost:${VLLM_PORT}"
echo ""
echo "  查看日志: docker logs -f ${CONTAINER_NAME}"
echo "  停止服务: docker stop ${CONTAINER_NAME}"
echo -e "==============================================${NC}"

# 等待服务就绪
echo ""
echo "等待服务启动 (约3-5分钟)..."
for i in {1..60}; do
    if curl -sf http://localhost:${WEB_PORT}/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 服务已就绪！${NC}"
        exit 0
    fi
    sleep 5
    echo -n "."
done

echo ""
echo -e "${YELLOW}服务仍在启动中，请稍后访问 http://localhost:${WEB_PORT}${NC}"
echo "可使用 'docker logs -f ${CONTAINER_NAME}' 查看启动进度"
