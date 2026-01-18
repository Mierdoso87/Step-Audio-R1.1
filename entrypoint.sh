#!/bin/bash
set -e

echo "=============================================="
echo "Step-Audio-R1.1 All-in-One Container"
echo "=============================================="
echo "Configuration:"
echo "  - Model Path: /model"
echo "  - Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE:-4}"
echo "  - Max Model Length: ${MAX_MODEL_LEN:-65536}"
echo "  - Max Concurrent Requests: ${MAX_NUM_SEQS:-4}"
echo "  - GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION:-0.85}"
echo "  - Web UI Port: 9100"
echo "  - vLLM API Port: 9999"
echo "=============================================="

# Check if model exists
if [ ! -d "/model" ] || [ -z "$(ls -A /model 2>/dev/null)" ]; then
    echo "ERROR: Model not found at /model"
    echo "Please mount the model directory: -v /path/to/Step-Audio-R1.1:/model"
    exit 1
fi

echo "Starting services..."
exec /usr/local/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
