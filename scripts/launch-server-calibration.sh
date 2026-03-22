#!/bin/bash
# Launch vLLM in calibration mode for per-layer scale recording

set -e

cd /home/yeb/Developer/qwen3.5
source venv/bin/activate

# Clear triton cache for fresh kernel compilation
rm -rf ~/.cache/triton 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_KV_SCALES_RECORD=1
export VLLM_KV_SCALES_OUTPUT=/home/yeb/Developer/qwen3.5/scales/qwen35_27b_per_layer.json
export VLLM_INT8_V_FP8_EMUL=1

echo "Starting vLLM in CALIBRATION MODE"
echo "Scales will be recorded and saved to: $VLLM_KV_SCALES_OUTPUT"
echo ""

vllm serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
    --tensor-parallel-size 2 \
    --kv-cache-dtype int8 \
    --calculate-kv-scales \
    --kv-cache-memory-bytes 3221225472 \
    --max-model-len 8192 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
