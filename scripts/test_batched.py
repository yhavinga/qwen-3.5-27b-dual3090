#!/usr/bin/env python3
"""Test throughput with concurrent requests to measure batching effect."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"

def single_request(prompt: str, max_tokens: int = 50):
    """Make single request, return tokens and time."""
    start = time.perf_counter()
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
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    return result["usage"]["completion_tokens"], elapsed

def test_concurrent(num_concurrent: int, prompt: str = "Count from 1 to 100: 1, 2, 3,"):
    """Test with N concurrent requests."""
    start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(single_request, prompt) for _ in range(num_concurrent)]
        results = [f.result() for f in as_completed(futures)]
    
    total_time = time.perf_counter() - start
    total_tokens = sum(r[0] for r in results)
    
    return total_tokens, total_time

print("Batching Effect on Throughput")
print("=" * 60)
print("Testing how concurrent requests improve total throughput...")
print()

# Warmup
single_request("Hello", 5)

# Test single request baseline
tokens, elapsed = single_request("Count: 1, 2, 3,", 50)
single_tps = tokens / elapsed
print(f"Single request:     {tokens} tokens in {elapsed:.2f}s = {single_tps:.1f} tok/s")

# Test concurrent
for n in [2, 4, 8]:
    tokens, elapsed = test_concurrent(n)
    throughput = tokens / elapsed
    efficiency = throughput / single_tps
    print(f"{n} concurrent:       {tokens} tokens in {elapsed:.2f}s = {throughput:.1f} tok/s ({efficiency:.1f}x single)")

print()
print("Interpretation:")
print("- If N concurrent ~= N × single throughput: batching is working")
print("- If N concurrent ~= single throughput: requests are serialized")
