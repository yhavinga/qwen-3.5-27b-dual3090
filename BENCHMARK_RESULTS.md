# Qwen3.5-27B on Dual RTX 3090: Benchmark Results

## Hardware Setup
- 2x NVIDIA RTX 3090 (24GB each) with NVLink
- Power consumption: ~295W per GPU (undervolted)
- vLLM with tensor-parallel-size=2

## TL;DR

Qwen3.5-27B is a **hybrid model** (16 attention + 48 GDN layers). This changes everything:

| Finding | Implication |
|---------|-------------|
| Only 25% of layers use KV cache | INT8 KV cache overhead outweighs benefits |
| GDN layers don't degrade with context | Excellent long-context scaling |
| GDN state is fixed ~8.6GB | TP=2 required even with 4-bit quantization |

## Decode Speed by Context Length

| Context | Decode (tok/s) | Prefill (tok/s) | TTFT |
|--------:|---------------:|----------------:|-----:|
| 8K | 51 | 1912 | 4s |
| 32K | 45 | 1984 | 16s |
| 64K | 39 | 1821 | 35s |
| 128K | 40 | 1548 | 83s |
| 200K | ~30 | 1262 | 158s |

**Key observation**: Decode speed only drops 22% from 8K to 128K context. Pure transformers like Gemma 3 27B drop 69% over the same range.

## Qwen 3.5 vs Gemma 3 (27B models)

| Context | Gemma 3 (INT8 KV) | Qwen 3.5 (FP16 KV) |
|--------:|------------------:|-------------------:|
| 8K | 67 tok/s | 51 tok/s |
| 32K | 35 tok/s | 45 tok/s |
| 64K | 25 tok/s | 39 tok/s |
| 128K | 21 tok/s | 40 tok/s |

Qwen is slower at short context but **faster above 32K**. Hybrid architectures scale better.

## Single GPU Options

| Backend | Single RTX 3090? | Speed | Max Context |
|---------|------------------|-------|-------------|
| vLLM | No (OOM) | - | - |
| llama.cpp | Yes (Q4_K_M) | 15-25 tok/s | ~131K |

vLLM requires TP=2 because GDN buffers (~8.6GB) + model weights fill 24GB before KV cache allocation.

## RTX 3090 vs RTX 5090

| Setup | 8K | 32K | 128K |
|-------|---:|----:|-----:|
| 2x RTX 3090 (vLLM) | 51 tok/s | 45 tok/s | 40 tok/s |
| 1x RTX 5090 | 61 tok/s | 44 tok/s | ~35 tok/s |

Dual RTX 3090 matches or beats single RTX 5090 at long context, at lower cost.

## vLLM Configuration

```bash
vllm serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype auto \
  --enable-chunked-prefill
```

**Note**: Use `--kv-cache-dtype auto` (FP16). INT8 KV cache is slower for hybrid models because the quantization overhead affects 100% of attention ops but only benefits 25% of layers.

## Maximum Context

| KV Cache | Max Context | Use Case |
|----------|-------------|----------|
| FP16 | ~200K | Single user, best speed |
| INT8 | ~400K | When you need extreme context |

## Batched Throughput

| Concurrent Requests | Throughput |
|--------------------:|-----------:|
| 1 | 48 tok/s |
| 8 | 305 tok/s |
| 32 | 874 tok/s |
