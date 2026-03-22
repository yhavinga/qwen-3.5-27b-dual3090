#!/bin/bash
# Qwen 3.5 35B-A3B (MoE) — Dual GPU Data Parallelism for max batch throughput
#
# Uses both GPUs independently (DP=2) rather than splitting the model (TP).
# Better for serving multiple concurrent users.
#
# Expected: ~200+ tok/s aggregate throughput with batch

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# DP=2: each GPU runs full model independently
exec vllm serve "Qwen/Qwen3.5-35B-A3B-FP8" \
    --data-parallel-size 2 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}' \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 64 \
    --port 8000 \
    "$@"
