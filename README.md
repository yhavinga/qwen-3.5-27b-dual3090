# Qwen 3.5 27B — Dual RTX 3090 NVLink Maximum Performance

Optimized setup for running Qwen 3.5 27B Q4 at maximum speed on dual RTX 3090s with NVLink.

## Performance Summary (Tested 2026-03-21)

| Configuration | tok/s | Context | Notes |
|--------------|-------|---------|-------|
| **27B GPTQ-Int4 TP=2** | **49** | <1K | Peak decode speed |
| 27B GPTQ-Int4 TP=2 | 42 | 2K | |
| 27B GPTQ-Int4 TP=2 | 29 | 8K | |
| 27B GPTQ-Int4 TP=2 | 20 | 16K | |
| 27B GPTQ-Int4 TP=2 | 12 | 32K | Memory-bound |
| **35B-A3B MoE** | 100-112 | 262K | Maximum speed (untested) |

## vLLM Version

**Current: vLLM 0.18.1rc1 (built from source with patches)**

Two critical patches applied:
1. **FakeTensorMode fix** - Disables AOT compile for PyTorch <2.12 (prevents startup crash)
2. **INT8 KV cache** - 50% memory reduction for 128K+ context on Ampere GPUs

### Build from Source (Recommended)

```bash
./scripts/build_vllm_from_source.sh
```

### Alternative: vLLM 0.17.1 (Simple)

```bash
pip install vllm==0.17.1
```

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

## INT8 KV Cache (Extended Context)

For 128K+ context, apply the INT8 KV cache patch:

```bash
# Apply patch to vLLM
./scripts/apply_int8_patch.sh

# Launch with INT8 KV
./scripts/launch-server-int8kv.sh

# Calibrate scales
python scripts/calibrate_kv_scales.py
```

**How it works:**
- K (keys): INT8 symmetric quantization
- V (values): FP8-E4M3 emulated in INT8 storage
- Per-layer calibrated scales handle 340x variance

**Memory savings:**
- 65K context: 8.5GB → 4.25GB KV cache
- Enables 128K where BF16 would OOM

## Model Comparison

| Model | Params Active | VRAM (Q4) | Quality | Speed |
|-------|--------------|-----------|---------|-------|
| 27B Dense | 27B | ~17GB | Highest | Baseline |
| 35B-A3B MoE | 3B | ~19GB | Very Good | 5x faster |

The 35B-A3B uses Gated DeltaNet layers (30/40) with fixed-size recurrent state instead of KV cache, enabling constant memory at any context length for those layers.

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
