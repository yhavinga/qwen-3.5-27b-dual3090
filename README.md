# Qwen 3.5 27B — Dual RTX 3090 Optimized

## TL;DR

```bash
pip install vllm==0.17.1

vllm serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --disable-custom-all-reduce \
  --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

**55 tok/s at 8K-32K context.** Minimal degradation thanks to hybrid architecture.

**CRITICAL**: The `--compilation-config` with reduced CUDA graph sizes is essential! Default (51 sizes) causes OOM on RTX 3090.

---

## Performance Summary

| Context | TTFT | Decode (tok/s) |
|--------:|-----:|---------------:|
| 8K | 4s | 48 |
| 16K | 8s | 47 |
| 32K | 16s | 45 |
| 64K | 35s | 43 |
| 128K | 85s | 38 |

**Key finding**: Only 20% decode speed degradation from 8K to 128K, thanks to hybrid architecture (25% attention + 75% GDN layers).

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for full analysis including comparison with Gemma 3 and RTX 5090.

## vLLM Version & CUDA Graph Fix

**Recommended: vLLM 0.17.1** with **CUDA 12.4+** (PyTorch ships with CUDA 12.4+)

```bash
pip install vllm==0.17.1
```

CUDA 12.4+ reduces CUDA graph memory by ~70% compared to older versions.

**CRITICAL**: Both vLLM 0.17.1 and 0.18.x OOM during CUDA graph capture with default settings. The fix is to reduce cudagraph_capture_sizes from 51 to 6:

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

Without this, the model loads (~13.9 GiB) but CUDA graph profiling needs ~800 MiB more than available.

## Quick Start

```bash
# Test the server
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-27B-GPTQ-Int4", "prompt": "Hello", "max_tokens": 50}'
```

## Key Optimizations

### CUDA Graphs (3x Decode Speedup)

```bash
--disable-custom-all-reduce  # Required for RTX 3090
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", ...}'
```

### GPTQ-Marlin Backend

```bash
--quantization gptq_marlin  # Fast Marlin kernels for GPTQ
```

## INT8 KV Cache — NOT Recommended for Qwen 3.5

**INT8 KV cache makes Qwen 3.5 SLOWER**, not faster:

| Context | FP16 | INT8 | Change |
|--------:|-----:|-----:|-------:|
| 8K | 51 | 50 | -2% |
| 64K | 39 | 33 | -15% |
| 128K | 40 | 26 | -35% |

**Why?** Qwen 3.5 is a hybrid model with only 25% attention layers. INT8 quantization overhead affects all attention ops but only benefits 25% of the model.

**When to use INT8**: Only if FP16 causes OOM (>200K context). INT8 doubles KV capacity but at 4x speed penalty.

See [INT8_KV_CACHE_ANALYSIS.md](INT8_KV_CACHE_ANALYSIS.md) for detailed analysis.

## Single GPU?

**vLLM**: No. GDN buffers (~8.6GB) + model weights fill 24GB before KV cache allocation.

**llama.cpp**: Yes. Q4_K_M (~17GB) fits on RTX 3090 with ~7GB for context. Speed: 15-25 tok/s (vs 51 tok/s with vLLM TP=2).

## Hardware Comparison

| Setup | 8K | 32K | 128K |
|-------|---:|----:|-----:|
| 2x RTX 3090 (vLLM) | 48 tok/s | 45 tok/s | 38 tok/s |
| 1x RTX 5090 | 61 tok/s | 44 tok/s | ~35 tok/s |

Dual RTX 3090 competitive with RTX 5090, especially at long context.

## Troubleshooting

### Low Performance (<30 tok/s)

1. Verify GPUs: `nvidia-smi`
2. Check vLLM version: `pip show vllm`
3. Check topology: `nvidia-smi topo -m` (NV4 = NVLink, PHB = PCIe)

### OOM Errors

1. Reduce context: use `launch-server-32k.sh`
2. Lower GPU util: add `--gpu-memory-utilization 0.85`
3. Enable INT8 KV cache for 50% memory savings

### CUDA Graph Capture OOM

Default cudagraph_capture_sizes (51 entries) requires too much memory. Use minimal config:

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

Or disable CUDA graphs entirely (slower):
```bash
--enforce-eager
```

## Directory Structure

```
qwen3.5/
├── vllm-source/                     # vLLM 0.18.1rc1 source (editable install)
│   └── vllm/                        # Patched vLLM code
├── scripts/
│   ├── launch-server.sh             # Main launcher (27B, 65K)
│   ├── launch-server-32k.sh         # Conservative (27B, 32K)
│   ├── launch-server-int8kv.sh      # Long context (27B, 128K)
│   ├── launch-server-35b-moe.sh     # Fast MoE single GPU
│   ├── build_vllm_from_source.sh    # Build vLLM with patches
│   ├── patch_int8_kv.py             # INT8 KV cache patch tool
│   ├── patch_faketensor_fix.py      # FakeTensorMode fix tool
│   ├── test_int8_patch.py           # Verify patches
│   ├── benchmark.py                 # Performance benchmark
│   └── quick_test.py                # Sanity check
├── patches/
│   └── vllm-int8-kv-cache-with-fp8v.patch
├── results/                         # Benchmark outputs
├── BENCHMARK_RESULTS.md             # Detailed findings
└── venv/                            # Python environment
```

## References

- [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [Qwen 3.5 GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4)
- [35B-A3B 112 tok/s Analysis](https://medium.com/@CodePulse/one-rtx-3090-112-tokens-per-second-full-262k-context-no-api-bill-304f60029bb6)
