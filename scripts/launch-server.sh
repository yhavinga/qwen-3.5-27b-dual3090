#!/bin/bash
# Qwen 3.5 27B GPTQ-Int4 — Dual RTX 3090 NVLink Maximum Performance
#
# TESTED PERFORMANCE (2026-03-21):
#   - Peak decode: 49 tok/s (short context)
#   - 8K context: 29 tok/s
#   - 16K context: 20 tok/s
#   - 32K context: 12 tok/s
#
# Requirements:
#   - 2x RTX 3090 with NVLink bridge
#   - vLLM 0.18.1rc1 (patched: FakeTensorMode fix + INT8 KV cache)
#   - Model: Qwen/Qwen3.5-27B-GPTQ-Int4 (~17GB)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# NVLink Optimization (critical for 50% speedup over PCIe)
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL           # Force NVLink path
export NCCL_BUFF_SIZE=16777216      # 16MB buffer for large transfers

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
    --quantization gptq_marlin \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 64 \
    --port 8000 \
    "$@"
