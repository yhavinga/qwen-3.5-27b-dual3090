#!/usr/bin/env python3
"""
First Principles Bandwidth Analysis for 80% Utilization
"""

# Hardware specs
RTX_3090_BW = 936.0  # GB/s per card
NVLINK_BW = 112.0    # GB/s bidirectional
NUM_GPUS = 2

# Model specs (GPTQ-Int4 with TP=2)
MODEL_SIZE_GB = 14.0
WEIGHTS_PER_GPU = MODEL_SIZE_GB / NUM_GPUS  # 7 GB

# Measured results
SINGLE_DECODE_TPS = 48.0  # tok/s single request
BATCH8_DECODE_TPS = 164.5 # tok/s total with 8 concurrent

print("=" * 70)
print("FIRST PRINCIPLES BANDWIDTH ANALYSIS")
print("=" * 70)

print("\n## Current State ##")
print(f"Single request decode: {SINGLE_DECODE_TPS:.1f} tok/s")
print(f"8 concurrent requests: {BATCH8_DECODE_TPS:.1f} tok/s total")
print()

# Bandwidth calculation for single request
single_bw = SINGLE_DECODE_TPS * WEIGHTS_PER_GPU
print(f"Single request bandwidth: {SINGLE_DECODE_TPS:.1f} × {WEIGHTS_PER_GPU:.1f} GB = {single_bw:.0f} GB/s ({single_bw/RTX_3090_BW*100:.1f}%)")

print("\n## Theoretical Limits ##")
theoretical_max_tps = RTX_3090_BW / WEIGHTS_PER_GPU
print(f"Max decode (100% BW): {RTX_3090_BW:.0f} / {WEIGHTS_PER_GPU:.1f} = {theoretical_max_tps:.0f} tok/s")
print(f"80% target: {theoretical_max_tps * 0.8:.0f} tok/s")
print(f"Currently achieving: {single_bw/RTX_3090_BW*100:.1f}% of theoretical")

print("\n## Why We're Not at 100% ##")
print("""
1. Kernel Launch Overhead (~15-25%)
   - ~500 kernel launches per token
   - Each launch: 5-20μs
   - Total: 2.5-10ms per token

2. Tensor Parallelism Sync (~10-15%)
   - 16 all-reduce ops per token (one per attention layer)
   - NVLink is 8x slower than HBM
   - Sync stalls pipeline

3. Memory Access Fragmentation (~5-10%)
   - INT4 dequantization requires gather ops
   - KV cache access is non-sequential
   - Small batch = poor coalescing

4. PyTorch/vLLM Framework (~5%)
   - Python dispatch overhead
   - Scheduler overhead
""")

print("\n## Paths to 80% Utilization ##")
print()

# Path 1: Batching
print("1. BATCHING (Proven - 3.4x improvement seen)")
print(f"   - Current 8-batch: {BATCH8_DECODE_TPS:.1f} tok/s total")
print(f"   - Effective per-output BW: {SINGLE_DECODE_TPS * 3.4:.0f} tok/s equivalent")
print(f"   - With larger batch (16+): Could reach 200+ tok/s total")
print()

# Path 2: CUDA Graphs
print("2. CUDA GRAPHS (vLLM V1 supports this)")
print("   - Eliminates kernel launch overhead")
print("   - Estimate: +15-20% improvement")
print(f"   - Single request: {SINGLE_DECODE_TPS * 1.17:.0f} tok/s")
print()

# Path 3: Speculative Decoding
print("3. SPECULATIVE DECODING")
print("   - Draft 4-8 tokens with small model")
print("   - Verify all in one forward pass")
print("   - 2-3x effective speedup for single request")
print()

# Path 4: Combine
print("4. COMBINED APPROACH")
target_improvement = 0.80 / 0.37
print(f"   - Need {target_improvement:.1f}x improvement to reach 80%")
print(f"   - Batching (2x) + CUDA Graphs (1.2x) = 2.4x")
print(f"   - That gives: {0.37 * 2.4 * 100:.0f}% utilization")
print()

print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
To achieve 80% bandwidth utilization:

IMMEDIATE (no code changes):
1. Increase --max-num-seqs from 4 to 16 or higher
2. Run with actual concurrent load (multiple users/requests)

MEDIUM TERM:
3. Enable CUDA graphs if available in vLLM V1
4. Enable chunked prefill for better batching

LONGER TERM:
5. Implement speculative decoding with small draft model
6. Custom CUDA kernels for fused operations

The key insight: 80% utilization is achievable with BATCHING.
Single-request decode will always be memory-bound at ~40% utilization.
""")
