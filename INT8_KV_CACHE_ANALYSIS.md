# INT8 KV Cache Analysis for Hybrid Models

## Summary

**INT8 KV cache provides NO speedup for Qwen3.5-27B** due to its hybrid architecture.
This analysis explains why and when INT8 KV cache is (not) beneficial.

## Hybrid Architecture

Qwen3.5-27B uses two types of layers:

| Layer Type | Count | KV Cache? | INT8 Benefit |
|------------|-------|-----------|--------------|
| Full Attention | 16 (25%) | Yes | Possible |
| Gated Delta Net (GDN) | 48 (75%) | No (fixed state) | None |

The GDN layers use a recurrent state (~50MB total, fixed size) instead of KV cache.
This means INT8 KV cache optimizations only affect 25% of the model.

## Benchmark: Qwen3.5-27B on 2x RTX 3090

| Context | FP16 KV | INT8 KV | Difference |
|--------:|--------:|--------:|------------|
| 8K | 51 tok/s | 50 tok/s | -2% |
| 64K | 39 tok/s | 33 tok/s | -15% |
| 128K | 40 tok/s | 26 tok/s | -35% |

INT8 is consistently **slower**, not faster.

## Why INT8 is Slower for Hybrid Models

1. **Limited scope**: Only 25% of layers benefit from reduced memory bandwidth
2. **Overhead**: Quantization/dequantization adds latency to every attention op
3. **Compute-bound**: GDN layers dominate compute time, not memory bandwidth

```
Memory bandwidth utilization: <1%
Conclusion: Hybrid models are COMPUTE-bound, not MEMORY-bound
```

## Comparison: Hybrid vs Pure Transformer

| Model Type | Architecture | INT8 KV @ 64K |
|------------|--------------|---------------|
| Qwen 3.5-27B | Hybrid (25% attention) | 15% slower |
| Gemma 3-27B | Pure transformer | 87% faster |

INT8 KV cache works excellently for pure transformers, but hurts hybrid models.

## When to Use INT8 KV Cache

| Scenario | Recommendation |
|----------|----------------|
| Pure transformer, long context | INT8 |
| Hybrid model (Qwen3.5, Jamba) | FP16 |
| Need >200K context, accept 4x slowdown | INT8 |
| Memory-constrained, OOM otherwise | INT8 |

## Memory Capacity

INT8 still provides 2x memory capacity for KV cache:

| KV Cache | Max Context | Trade-off |
|----------|-------------|-----------|
| FP16 | ~200K | Best speed |
| INT8 | ~400K | 4x slower |

Use INT8 only when FP16 causes OOM.

## Conclusion

For **pure transformers**: INT8 KV cache is a powerful optimization at long context.

For **hybrid models** like Qwen3.5: Use FP16 KV cache. The architecture already scales
efficiently at long context because 75% of layers have O(1) memory complexity.
