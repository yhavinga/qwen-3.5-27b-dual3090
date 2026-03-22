#!/bin/bash
# Qwen 3.5 27B GPTQ-Int4 — Conservative 32K context
#
# Use this if you encounter OOM or stability issues with 65K.
# More headroom for concurrent requests.
#
# Expected: 45-65 tok/s, very stable

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

exec vllm serve "Qwen/Qwen3.5-27B-GPTQ-Int4" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --quantization gptq_marlin \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --max-num-seqs 128 \
    --port 8000 \
    "$@"
