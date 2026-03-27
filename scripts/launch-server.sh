#!/bin/bash
# Qwen 3.5 27B GPTQ-Int4 — Dual RTX 3090 NVLink Maximum Performance
#
# TESTED PERFORMANCE (2026-03-25):
#   - 8K context: 56 tok/s
#   - 16K context: 63 tok/s
#   - 32K context: 55 tok/s
#
# Requirements:
#   - 2x RTX 3090 with NVLink bridge
#   - vLLM 0.17.1 (pip install vllm==0.17.1)
#   - Model: Qwen/Qwen3.5-27B-GPTQ-Int4 (~17GB)
#
# CRITICAL: Use minimal cudagraph_capture_sizes to avoid OOM!

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# GPU Selection
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1

# Note: NCCL auto-detects NVLink — no env vars needed.
# Testing shows ~5% speedup for single requests (not 50% as sometimes claimed).

# ============================================================================
# Memory Optimization
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# ============================================================================
# Activate venv if present
# ============================================================================
if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# ============================================================================
# Launch vLLM
# ============================================================================
exec vllm serve "Qwen/Qwen3.5-27B-GPTQ-Int4" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --port 8000 \
    "$@"
