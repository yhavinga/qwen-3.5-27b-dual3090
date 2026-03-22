#!/usr/bin/env python3
"""Test maximum throughput with high concurrency to approach 80% bandwidth."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"

# Hardware specs
RTX_3090_BW = 936.0
WEIGHTS_PER_GPU = 7.0

def request(prompt: str, max_tokens: int = 50):
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["usage"]["completion_tokens"]

def test_concurrent(n: int, gen_tokens: int = 50):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(request, f"Count: {i},", gen_tokens) for i in range(n)]
        tokens = sum(f.result() for f in as_completed(futures))
    elapsed = time.perf_counter() - start
    return tokens, elapsed

print("Maximum Throughput Test (max-num-seqs=32)")
print("=" * 70)
print(f"Target: 80% of {RTX_3090_BW} GB/s = {RTX_3090_BW * 0.8:.0f} GB/s")
print(f"Theoretical max: {RTX_3090_BW / WEIGHTS_PER_GPU:.0f} tok/s decode")
print(f"80% target: {(RTX_3090_BW * 0.8) / WEIGHTS_PER_GPU:.0f} tok/s")
print()

# Warmup
request("Hi", 5)

print(f"{'Concurrent':<12} {'Tokens':<10} {'Time':<10} {'Throughput':<15} {'BW Used':<12} {'Utilization'}")
print("-" * 70)

for n in [1, 2, 4, 8, 16, 24, 32]:
    try:
        tokens, elapsed = test_concurrent(n, gen_tokens=40)
        tps = tokens / elapsed
        bw_used = (tps * WEIGHTS_PER_GPU)
        util = bw_used / RTX_3090_BW * 100
        print(f"{n:<12} {tokens:<10} {elapsed:.2f}s     {tps:.1f} tok/s     {bw_used:.0f} GB/s     {util:.1f}%")
    except Exception as e:
        print(f"{n:<12} ERROR: {e}")

print()
print("Note: At batch=32, we should see significant improvement towards 80%")
