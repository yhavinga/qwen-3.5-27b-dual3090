#!/usr/bin/env python3
"""
Deep analysis of bandwidth vs throughput trade-off
"""

RTX_3090_BW = 936.0  # GB/s
WEIGHTS_PER_GPU = 7.0  # GB
NUM_GPUS = 2

print("=" * 70)
print("DEEP BANDWIDTH ANALYSIS: The Real Story")
print("=" * 70)

print("""
## The Key Insight ##

For AUTO-REGRESSIVE decode:
- Each forward pass reads ALL model weights (7 GB per GPU)
- Single request: 1 forward pass = 1 token  
- Batched (32): 1 forward pass = 32 tokens (amortized!)

## What We Measured ##
""")

results = [
    (1, 40, 1.21, 33.2),
    (2, 80, 1.64, 48.9),
    (4, 160, 1.19, 134.3),
    (8, 320, 1.05, 305.1),
    (16, 640, 1.99, 321.5),
    (24, 960, 1.41, 681.2),
    (32, 1280, 1.46, 874.3),
]

print(f"{'Batch':<8} {'Tok/s':<12} {'FwdPass/s':<12} {'BW/GPU':<12} {'BW Util':<10} {'Throughput ×'}")
print("-" * 70)

single_tps = 33.2
for batch, tokens, time, tps in results:
    fwd_per_sec = tps / batch  # forward passes per second
    bw_per_gpu = fwd_per_sec * WEIGHTS_PER_GPU
    util = bw_per_gpu / RTX_3090_BW * 100
    speedup = tps / single_tps
    print(f"{batch:<8} {tps:<12.1f} {fwd_per_sec:<12.1f} {bw_per_gpu:.0f} GB/s     {util:.1f}%      {speedup:.1f}×")

print("""
## Interpretation ##

As batch size increases:
1. Throughput (tok/s) increases massively (26× at batch=32!)
2. But forward passes/sec DECREASES (33 → 27)
3. And bandwidth utilization DECREASES (25% → 20%)

This means: at high batch, we're COMPUTE-BOUND, not memory-bound!

## The Two Questions ##

Q1: "How do we get 80% bandwidth utilization?"
A1: This is about SINGLE REQUEST latency.
    - Currently: ~25% utilization, 33 tok/s
    - At 80%: would be ~107 tok/s single request
    - Requires: CUDA graphs, custom kernels, less overhead

Q2: "How do we maximize THROUGHPUT?"  
A2: Use BATCHING. We already achieve:
    - 874 tok/s at batch=32 (26× single request)
    - This is the better metric for serving workloads

## For 80% Single-Request Bandwidth ##

Need to eliminate overhead:
1. Kernel launch overhead: ~15-25%
2. TP sync overhead: ~10-15%  
3. Framework overhead: ~5-10%

CUDA graphs help with #1 (already enabled in vLLM V1!)
Better NVLink pipelining helps with #2
Custom kernels help with #3

But the honest answer: 80% utilization on single request
is very hard. Industry standard is 30-50%.
""")

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
SINGLE REQUEST:
  - Decode: 33 tok/s
  - Bandwidth: ~230 GB/s (25% utilization)
  - Bottleneck: kernel overhead, TP sync

BATCHED (32 concurrent):
  - Throughput: 874 tok/s (26× single!)
  - This is excellent for serving workloads

TO GET 80% BW UTILIZATION ON SINGLE REQUEST:
  1. This is a hard problem - even top providers hit ~50%
  2. Options: speculative decoding, custom CUDA kernels
  3. Speculative decoding is most practical
""")
