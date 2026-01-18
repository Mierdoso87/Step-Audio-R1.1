#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  Step-Audio-R1.1 vLLM Backend Startup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check model
MODEL_PATH="${MODEL_PATH:-./Step-Audio-R1.1}"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Model not found at $MODEL_PATH${NC}"
    exit 1
fi

# Select GPUs with most free memory
echo -e "${YELLOW}Selecting GPUs...${NC}"
GPU_IDS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
          sort -t',' -k2 -rn | head -4 | cut -d',' -f1 | sort -n | tr '\n' ',' | sed 's/,$//')
echo -e "${GREEN}Using GPUs: $GPU_IDS${NC}"

# Check port
PORT=${VLLM_PORT:-9999}
while ss -tlnp | grep -q ":$PORT "; do
    PORT=$((PORT + 1))
done
echo -e "${GREEN}Using port: $PORT${NC}"

# Start vLLM
echo -e "${YELLOW}Starting vLLM server...${NC}"
docker run --rm -d --gpus "\"device=$GPU_IDS\"" \
    --name step-audio-r1-vllm \
    -v "$(pwd)/$MODEL_PATH":/model:ro \
    -p "0.0.0.0:$PORT:9999" \
    stepfun2025/vllm:step-audio-2-v20250909 \
    vllm serve /model \
    --served-model-name Step-Audio-R1.1 \
    --port 9999 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --max-num-seqs 16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --enable-log-requests \
    --interleave-mm-strings \
    --chat-template '{%- macro render_content(content) -%}{%- if content is string -%}{{- content.replace("<audio_patch>\n", "<audio_patch>") -}}{%- elif content is mapping -%}{{- content['"'"'value'"'"'] if '"'"'value'"'"' in content else content['"'"'text'"'"'] -}}{%- elif content is iterable -%}{%- for item in content -%}{%- if item.type == '"'"'text'"'"' -%}{{- item['"'"'value'"'"'] if '"'"'value'"'"' in item else item['"'"'text'"'"'] -}}{%- elif item.type == '"'"'audio'"'"' -%}<audio_patch>{%- endif -%}{%- endfor -%}{%- endif -%}{%- endmacro -%}{%- if tools -%}{{- '"'"'<|BOT|>system\n'"'"' -}}{%- if messages[0]['"'"'role'"'"'] == '"'"'system'"'"' -%}{{- render_content(messages[0]['"'"'content'"'"']) + '"'"'<|EOT|>'"'"' -}}{%- endif -%}{{- '"'"'<|BOT|>tool_json_schemas\n'"'"' + tools|tojson + '"'"'<|EOT|>'"'"' -}}{%- else -%}{%- if messages[0]['"'"'role'"'"'] == '"'"'system'"'"' -%}{{- '"'"'<|BOT|>system\n'"'"' + render_content(messages[0]['"'"'content'"'"']) + '"'"'<|EOT|>'"'"' -}}{%- endif -%}{%- endif -%}{%- for message in messages -%}{%- if message["role"] == "user" -%}{{- '"'"'<|BOT|>human\n'"'"' + render_content(message["content"]) + '"'"'<|EOT|>'"'"' -}}{%- elif message["role"] == "assistant" -%}{{- '"'"'<|BOT|>assistant\n'"'"' + (render_content(message["content"]) if message["content"] else '"'"''"'"') -}}{%- set is_last_assistant = true -%}{%- for m in messages[loop.index:] -%}{%- if m["role"] == "assistant" -%}{%- set is_last_assistant = false -%}{%- endif -%}{%- endfor -%}{%- if not is_last_assistant -%}{{- '"'"'<|EOT|>'"'"' -}}{%- endif -%}{%- elif message["role"] == "function_output" -%}{%- else -%}{%- if not (loop.first and message["role"] == "system") -%}{{- '"'"'<|BOT|>'"'"' + message["role"] + '"'"'\n'"'"' + render_content(message["content"]) + '"'"'<|EOT|>'"'"' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '"'"'<|BOT|>assistant\n<think>\n'"'"' -}}{%- endif -%}'

echo ""
echo -e "${GREEN}vLLM server starting on port $PORT${NC}"
echo -e "View logs: docker logs -f step-audio-r1-vllm"
echo -e "Stop: docker stop step-audio-r1-vllm"
