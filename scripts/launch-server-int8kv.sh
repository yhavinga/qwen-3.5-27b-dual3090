#!/bin/bash
# Qwen 3.5 27B GPTQ-Int4 + INT8 KV Cache — Maximum Context (128K+)
#
# Requires: vLLM patched with INT8 KV support (see patches/)
#
# This config trades ~5-10% compute for 50% KV cache memory savings,
# enabling 128K context where BF16 KV would OOM.
#
# Expected performance: 35-55 tok/s (slightly lower due to INT8 overhead)
# Context: up to 131072 tokens

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# INT8 KV Cache Configuration
# ============================================================================
# FP8-E4M3 emulation for V (values) - handles 340x variance across layers
export VLLM_INT8_V_FP8_EMUL=1

# Per-layer calibrated scales (run calibration first!)
SCALES_FILE="${PROJECT_ROOT}/scales/qwen35_27b_per_layer.json"
if [ -f "$SCALES_FILE" ]; then
    export VLLM_KV_SCALES_FILE="$SCALES_FILE"
    echo "Using per-layer scales: $SCALES_FILE"
else
    echo "WARNING: No scales file found. Using dynamic calibration."
    echo "Run: python scripts/calibrate_kv_scales.py after server starts"
fi

# ============================================================================
# NVLink Optimization
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216

# ============================================================================
# Memory Optimization
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# ============================================================================
# Activate venv
# ============================================================================
if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# ============================================================================
# Launch vLLM with INT8 KV cache
# ============================================================================
exec vllm serve "Qwen/Qwen3.5-27B-GPTQ-Int4" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --quantization gptq_marlin \
    --kv-cache-dtype int8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.88 \
    --enable-chunked-prefill \
    --max-num-seqs 32 \
    --port 8000 \
    "$@"
