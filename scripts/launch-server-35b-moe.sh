#!/bin/bash
# Qwen 3.5 35B-A3B (MoE) — FASTEST option: 100+ tok/s on single GPU!
#
# This is the MoE variant with only 3B active parameters per token.
# 30/40 layers use Gated DeltaNet (no KV cache), enabling:
#   - 112 tok/s on single RTX 3090
#   - Full 262K context in 22GB VRAM
#
# Trade-off: Slightly lower quality than dense 27B for some tasks,
# but 5x faster and better long-context handling.
#
# With dual 3090s, you could run DP=2 for even higher throughput.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Single GPU is optimal for this model (only 3B active params)
# With dual GPUs, use DP=2 for batch throughput instead of TP
export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# Single GPU config for maximum tok/s
exec vllm serve "Qwen/Qwen3.5-35B-A3B-FP8" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}' \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --max-num-seqs 32 \
    --port 8000 \
    "$@"
