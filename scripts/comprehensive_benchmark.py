#!/usr/bin/env python3
"""
Comprehensive benchmark for Qwen3.5-27B on dual RTX 3090.

Measures:
1. Decode speed at various context lengths
2. Prefill speed (throughput)
3. Quality assessment
4. Memory bandwidth utilization analysis
"""

import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

API_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"

# Hardware specs
RTX_3090_BW_GBS = 936.0  # GB/s per card
NVLINK_BW_GBS = 112.0    # GB/s bidirectional (NV3)
NUM_GPUS = 2

# Model specs (approximate for GPTQ-Int4)
MODEL_SIZE_GB = 14.0     # ~14GB for 4-bit quantized 27B
PARAMS_PER_GPU = MODEL_SIZE_GB / NUM_GPUS  # 7GB per GPU with TP=2

# KV cache specs for Qwen3.5-27B (16 attention layers, 8 KV heads per TP, 128 dim)
KV_HEADS_PER_GPU = 8
HEAD_DIM = 128
NUM_ATTN_LAYERS = 16
BYTES_PER_KV_ELEMENT = 1  # INT8


def request(prompt: str, max_tokens: int, temperature: float = 0) -> dict:
    """Make API request and return result."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def measure_decode_speed(context_tokens: int, gen_tokens: int = 100) -> dict:
    """Measure decode speed at given context length."""
    # Create prompt of approximate token count
    corpus = Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt").read_text()

    # Rough estimate: 4 chars per token
    chars_needed = context_tokens * 4
    prompt = corpus[:chars_needed] + "\nContinue: 1, 2, 3,"

    # Warmup
    request(prompt, 1)

    # Measure with two different gen lengths to isolate decode speed
    start1 = time.perf_counter()
    r1 = request(prompt, gen_tokens // 2)
    t1 = time.perf_counter() - start1

    start2 = time.perf_counter()
    r2 = request(prompt, gen_tokens)
    t2 = time.perf_counter() - start2

    actual_context = r1["usage"]["prompt_tokens"]
    d1 = r1["usage"]["completion_tokens"]
    d2 = r2["usage"]["completion_tokens"]

    # Decode speed = delta tokens / delta time (cancels prefill)
    decode_tps = (d2 - d1) / (t2 - t1) if t2 > t1 else 0

    # Prefill speed estimate
    prefill_time = t1 - (d1 / decode_tps) if decode_tps > 0 else t1
    prefill_tps = actual_context / prefill_time if prefill_time > 0 else 0

    return {
        "context_tokens": actual_context,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "total_time_100tok": t2,
    }


def measure_quality(prompt: str, max_tokens: int = 200) -> dict:
    """Measure output quality indicators."""
    result = request(prompt, max_tokens, temperature=0)
    output = result["choices"][0]["text"]

    # Quality checks
    issues = []

    # Check for repetition
    if len(output) > 50:
        # Check if any 20-char substring repeats more than 3 times
        for i in range(len(output) - 20):
            substr = output[i:i+20]
            if output.count(substr) > 3:
                issues.append("repetition")
                break

    # Check for garbage
    garbage_patterns = ["????", "####", "[[[[", "]]]]", "\x00"]
    for p in garbage_patterns:
        if p in output:
            issues.append(f"garbage:{p}")

    # Check for coherence (simple heuristic: has punctuation and reasonable length)
    if len(output.strip()) < 10:
        issues.append("too_short")
    if not any(c in output for c in ".!?,;:"):
        issues.append("no_punctuation")

    return {
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "output_tokens": result["usage"]["completion_tokens"],
        "output_preview": output[:200].replace("\n", " "),
        "issues": issues,
        "quality": "GOOD" if not issues else "ISSUES"
    }


def calculate_bandwidth_utilization(decode_tps: float, context_tokens: int) -> dict:
    """
    Calculate memory bandwidth utilization.

    For each decode step:
    1. Read all model weights (~7GB per GPU for TP=2)
    2. Read KV cache (grows with context)
    3. Write new KV cache entry
    4. NVLink communication for TP
    """
    # Model weight reads per token
    model_read_per_tok = PARAMS_PER_GPU  # GB

    # KV cache read per token (all previous K/V for attention)
    # Per layer: 2 * num_heads * head_dim * context_tokens * bytes
    kv_per_layer = 2 * KV_HEADS_PER_GPU * HEAD_DIM * context_tokens * BYTES_PER_KV_ELEMENT
    kv_read_per_tok = (kv_per_layer * NUM_ATTN_LAYERS) / 1e9  # GB

    # Total memory traffic per token
    total_read_per_tok = model_read_per_tok + kv_read_per_tok

    # Bandwidth used
    bw_used_per_gpu = decode_tps * total_read_per_tok  # GB/s

    # Utilization
    utilization = bw_used_per_gpu / RTX_3090_BW_GBS * 100

    return {
        "model_read_gb_per_tok": model_read_per_tok,
        "kv_read_gb_per_tok": kv_read_per_tok,
        "total_read_gb_per_tok": total_read_per_tok,
        "bandwidth_used_gbs": bw_used_per_gpu,
        "theoretical_max_gbs": RTX_3090_BW_GBS,
        "utilization_pct": utilization,
    }


def main():
    print("=" * 70)
    print("Comprehensive Benchmark: Qwen3.5-27B on Dual RTX 3090")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Hardware: 2x RTX 3090 ({RTX_3090_BW_GBS} GB/s each), NVLink ({NVLINK_BW_GBS} GB/s)")
    print()

    # 1. Decode speed at various context lengths
    print("=" * 70)
    print("1. DECODE SPEED vs CONTEXT LENGTH")
    print("=" * 70)
    print(f"{'Context':<12} {'Decode':<12} {'Prefill':<12} {'BW Used':<12} {'BW Util':<10}")
    print("-" * 70)

    results = []
    for target_ctx in [500, 1000, 2000, 4000, 8000, 16000]:
        try:
            r = measure_decode_speed(target_ctx)
            bw = calculate_bandwidth_utilization(r["decode_tps"], r["context_tokens"])

            print(f"{r['context_tokens']:<12} {r['decode_tps']:.1f} tok/s    {r['prefill_tps']:.0f} tok/s    "
                  f"{bw['bandwidth_used_gbs']:.0f} GB/s    {bw['utilization_pct']:.1f}%")

            results.append({**r, **bw})
        except Exception as e:
            print(f"{target_ctx:<12} ERROR: {e}")

    print()

    # 2. Quality tests
    print("=" * 70)
    print("2. QUALITY ASSESSMENT")
    print("=" * 70)

    quality_prompts = [
        ("Short EN", "Explain quantum computing in simple terms:"),
        ("Short NL", "Leg uit wat kunstmatige intelligentie is:"),
        ("Technical", "Write a Python function to implement binary search with detailed comments:"),
        ("Creative", "Write a short poem about the moon:"),
        ("Math", "Solve step by step: If 3x + 7 = 22, what is x?"),
        ("Long context", Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt").read_text()[:8000] + "\n\nSamenvatting:"),
    ]

    for name, prompt in quality_prompts:
        try:
            q = measure_quality(prompt)
            status = "✓" if q["quality"] == "GOOD" else "✗"
            print(f"{status} {name:<15} [{q['prompt_tokens']:>5} -> {q['output_tokens']:>3} tok] {q['quality']}")
            if q["issues"]:
                print(f"  Issues: {q['issues']}")
            print(f"  Output: {q['output_preview'][:60]}...")
        except Exception as e:
            print(f"✗ {name:<15} ERROR: {e}")
        print()

    # 3. Bandwidth analysis summary
    print("=" * 70)
    print("3. BANDWIDTH UTILIZATION ANALYSIS")
    print("=" * 70)

    if results:
        # Short context (model weight dominated)
        short = results[0]
        print(f"Short context ({short['context_tokens']} tok):")
        print(f"  Model weights: {short['model_read_gb_per_tok']:.2f} GB/tok")
        print(f"  KV cache: {short['kv_read_gb_per_tok']:.4f} GB/tok")
        print(f"  Total: {short['total_read_gb_per_tok']:.2f} GB/tok")
        print(f"  Bandwidth: {short['bandwidth_used_gbs']:.0f} / {short['theoretical_max_gbs']:.0f} GB/s = {short['utilization_pct']:.1f}%")
        print()

        # Long context (KV cache significant)
        long = results[-1]
        print(f"Long context ({long['context_tokens']} tok):")
        print(f"  Model weights: {long['model_read_gb_per_tok']:.2f} GB/tok")
        print(f"  KV cache: {long['kv_read_gb_per_tok']:.4f} GB/tok")
        print(f"  Total: {long['total_read_gb_per_tok']:.2f} GB/tok")
        print(f"  Bandwidth: {long['bandwidth_used_gbs']:.0f} / {long['theoretical_max_gbs']:.0f} GB/s = {long['utilization_pct']:.1f}%")
        print()

        # Theoretical max decode speed
        theoretical_max_tps = RTX_3090_BW_GBS / PARAMS_PER_GPU
        print(f"Theoretical max decode (ignoring KV): {theoretical_max_tps:.0f} tok/s")
        print(f"Achieved at short context: {short['decode_tps']:.1f} tok/s ({short['decode_tps']/theoretical_max_tps*100:.1f}% of theoretical)")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        avg_decode = sum(r["decode_tps"] for r in results) / len(results)
        avg_util = sum(r["utilization_pct"] for r in results) / len(results)
        print(f"Average decode speed: {avg_decode:.1f} tok/s")
        print(f"Average bandwidth utilization: {avg_util:.1f}%")
        print(f"Decode speed range: {min(r['decode_tps'] for r in results):.1f} - {max(r['decode_tps'] for r in results):.1f} tok/s")


if __name__ == "__main__":
    main()
