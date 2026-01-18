#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  Step-Audio-R1.1 Startup Script"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check nvidia-docker
echo -e "${YELLOW}[1/5] Checking nvidia-docker...${NC}"
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "${RED}Error: nvidia-docker not available${NC}"
    exit 1
fi
echo -e "${GREEN}✓ nvidia-docker available${NC}"

# Check model
echo -e "${YELLOW}[2/5] Checking model...${NC}"
MODEL_PATH="${MODEL_PATH:-./Step-Audio-R1.1}"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Model not found. Downloading Step-Audio-R1.1...${NC}"
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download stepfun-ai/Step-Audio-R1.1 --local-dir "$MODEL_PATH"
    else
        echo -e "${YELLOW}Installing huggingface-cli...${NC}"
        pip install -q huggingface_hub
        huggingface-cli download stepfun-ai/Step-Audio-R1.1 --local-dir "$MODEL_PATH"
    fi
fi
echo -e "${GREEN}✓ Model ready at $MODEL_PATH${NC}"

# Auto-select GPU with least memory usage
echo -e "${YELLOW}[3/5] Selecting optimal GPUs...${NC}"
# This model needs 4 GPUs - try to find 4 with most free memory
# Sort by free memory descending and take top 4
GPU_IDS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
          sort -t',' -k2 -rn | head -4 | cut -d',' -f1 | sort -n | tr '\n' ',' | sed 's/,$//')

GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -lt 4 ]; then
    echo -e "${RED}Error: Need 4 GPUs but only found $GPU_COUNT${NC}"
    exit 1
fi

# Check minimum free memory on selected GPUs
MIN_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
           sort -t',' -k2 -rn | head -4 | tail -1 | cut -d',' -f2 | tr -d ' ')
echo -e "${YELLOW}Minimum free memory on selected GPUs: ${MIN_FREE}MB${NC}"

if [ "$MIN_FREE" -lt 10000 ]; then
    echo -e "${RED}Warning: GPU with only ${MIN_FREE}MB free. May fail to load model.${NC}"
    echo -e "${YELLOW}Current GPU status:${NC}"
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv
    echo -e "${YELLOW}Attempting to start anyway with reduced settings...${NC}"
fi

export NVIDIA_VISIBLE_DEVICES=$GPU_IDS
echo -e "${GREEN}✓ Selected GPUs: $GPU_IDS${NC}"

# Check ports
echo -e "${YELLOW}[4/5] Checking ports...${NC}"
WEB_PORT=${WEB_PORT:-9100}
VLLM_PORT=${VLLM_PORT:-9101}

check_port() {
    if ss -tlnp | grep -q ":$1 "; then
        echo -e "${RED}Port $1 is in use, trying next...${NC}"
        return 1
    fi
    return 0
}

while ! check_port $WEB_PORT; do
    WEB_PORT=$((WEB_PORT + 1))
done
while ! check_port $VLLM_PORT; do
    VLLM_PORT=$((VLLM_PORT + 1))
done

export WEB_PORT VLLM_PORT MODEL_PATH
echo -e "${GREEN}✓ Using ports: Web=$WEB_PORT, vLLM=$VLLM_PORT${NC}"

# Create .env file
cat > .env << EOF
WEB_PORT=$WEB_PORT
VLLM_PORT=$VLLM_PORT
NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES
MODEL_PATH=$MODEL_PATH
EOF

# Start services
echo -e "${YELLOW}[5/5] Starting services...${NC}"
docker compose up -d --build

echo ""
echo -e "${GREEN}=========================================="
echo "  Step-Audio-R1.1 Started Successfully!"
echo "==========================================${NC}"
echo ""
echo -e "  🌐 Web UI:     http://0.0.0.0:${WEB_PORT}"
echo -e "  📚 API Docs:   http://0.0.0.0:${WEB_PORT}/docs"
echo -e "  🔧 vLLM API:   http://0.0.0.0:${VLLM_PORT}"
echo -e "  🤖 MCP:        See MCP_GUIDE.md"
echo ""
echo -e "  📁 Uploads:    /tmp/step-audio-r1/"
echo -e "  🎮 GPUs:       ${NVIDIA_VISIBLE_DEVICES}"
echo ""
echo -e "  View logs:     docker compose logs -f"
echo -e "  Stop:          docker compose down"
echo ""
