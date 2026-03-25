# Qwen 3.5 27B — Dual RTX 3090 NVLink Maximum Performance

Optimized setup for running Qwen 3.5 27B Q4 at maximum speed on dual RTX 3090s with NVLink.

## Performance Summary

| Context | Decode (tok/s) | Prefill (tok/s) | Notes |
|--------:|---------------:|----------------:|-------|
| 8K | 51 | 1912 | Peak decode speed |
| 32K | 45 | 1984 | Minimal degradation |
| 64K | 39 | 1821 | |
| 128K | 40 | 1548 | 83s TTFT |
| 200K | ~30 | 1262 | 158s TTFT |

**Key finding**: Only 22% speed degradation from 8K to 128K, thanks to hybrid architecture (25% attention + 75% GDN layers).

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for full analysis including comparison with Gemma 3 and RTX 5090.

## vLLM Version

**Recommended: vLLM 0.17.1**

```bash
pip install vllm==0.17.1
```

vLLM 0.18.x has a regression: CUDA graph profiling OOMs on RTX 3090 due to tight memory (model + GDN buffers = 22.5GB, only 80MB free). Use `enforce_eager=True` as workaround (4x slower).

## Quick Start

```bash
# Activate venv
source venv/bin/activate

# Verify NVLink connectivity
./scripts/check_nvlink.sh

# Start server (downloads model on first run)
./scripts/launch-server.sh

# In another terminal, test
python scripts/quick_test.py

# Full benchmark
python scripts/benchmark.py --output results/benchmark.json
```

## Available Launch Scripts

### Dense 27B (Highest Quality)

| Script | Context | Use Case |
|--------|---------|----------|
| `launch-server.sh` | 65K | **Default** - best balance |
| `launch-server-32k.sh` | 32K | Conservative, very stable |
| `launch-server-int8kv.sh` | 128K | Maximum context (requires patch) |

### MoE 35B-A3B (Highest Speed)

| Script | Config | Use Case |
|--------|--------|----------|
| `launch-server-35b-moe.sh` | Single GPU | Maximum tok/s (112+) |
| `launch-server-35b-moe-dp2.sh` | DP=2 | Batch serving |

## Key Optimizations

### NVLink (50% Speedup)

The scripts set these critical environment variables:
```bash
export NCCL_P2P_LEVEL=NVL        # Force NVLink path
export CUDA_FORCE_P2P_ACCESS=1   # Enable P2P
export VLLM_SKIP_P2P_CHECK=1     # Skip redundant checks
```

Verify NVLink is working:
```bash
nvidia-smi topo -m
# Should show NV4 between GPU0 and GPU1, not PHB
```

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
| 2x RTX 3090 (vLLM) | 51 tok/s | 45 tok/s | 40 tok/s |
| 1x RTX 5090 | 61 tok/s | 44 tok/s | ~35 tok/s |

Dual RTX 3090 matches or beats RTX 5090 at 32K+ context.

## Troubleshooting

### Low Performance (<30 tok/s)

1. Check NVLink: `./scripts/check_nvlink.sh`
2. Verify GPUs: `nvidia-smi`
3. Check vLLM version: `pip show vllm`

### OOM Errors

1. Reduce context: use `launch-server-32k.sh`
2. Lower GPU util: add `--gpu-memory-utilization 0.85`
3. Enable INT8 KV cache for 50% memory savings

### CUDA Graph Capture Fails

Known issue with AWQ models. Use GPTQ-Int4 instead:
- Model: `Qwen/Qwen3.5-27B-GPTQ-Int4`
- See: https://github.com/vllm-project/vllm/issues/35743

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
- [NVLink 50% Boost Benchmark](http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html)
- [Qwen 3.5 GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4)
- [35B-A3B 112 tok/s Analysis](https://medium.com/@CodePulse/one-rtx-3090-112-tokens-per-second-full-262k-context-no-api-bill-304f60029bb6)
