#!/usr/bin/env python3
"""Quick sanity test for the vLLM server.

Verifies:
1. Server is running and responding
2. Model generates reasonable output
3. Basic throughput measurement

Usage:
    python scripts/quick_test.py
"""

import json
import sys
import time
import requests

URL = "http://localhost:8000"
MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"


def check_health():
    """Check if server is healthy."""
    try:
        resp = requests.get(f"{URL}/health", timeout=5)
        if resp.status_code == 200:
            print("[OK] Server is healthy")
            return True
    except requests.RequestException:
        pass
    print("[FAIL] Server not responding")
    print(f"       Start with: ./scripts/launch-server.sh")
    return False


def check_models():
    """List available models."""
    try:
        resp = requests.get(f"{URL}/v1/models", timeout=10)
        models = resp.json()
        print(f"[OK] Available models: {[m['id'] for m in models.get('data', [])]}")
        return True
    except Exception as e:
        print(f"[WARN] Could not list models: {e}")
        return False


def test_generation():
    """Test basic generation."""
    prompt = "Write a haiku about GPU inference:"
    print(f"\n[TEST] Generating response...")
    print(f"       Prompt: {prompt}")

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{URL}/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.7,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")
        return False

    elapsed = time.perf_counter() - start
    text = data["choices"][0]["text"].strip()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 64)
    tok_s = completion_tokens / elapsed

    print(f"\n[OUTPUT]\n{text}\n")
    print(f"[OK] Generated {completion_tokens} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
    return True


def test_throughput():
    """Quick throughput test."""
    print("\n[TEST] Throughput measurement (3 runs)...")
    results = []

    for i in range(3):
        prompt = "Explain quantum computing in simple terms. " * 10  # ~100 tokens
        start = time.perf_counter()
        resp = requests.post(
            f"{URL}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": 128, "temperature": 0},
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        tok_s = 128 / elapsed
        results.append(tok_s)
        print(f"       Run {i+1}: {tok_s:.1f} tok/s")

    avg = sum(results) / len(results)
    print(f"\n[RESULT] Average: {avg:.1f} tok/s")

    if avg > 30:
        print("[OK] Performance looks good for dual 3090 + NVLink")
    elif avg > 15:
        print("[WARN] Performance is okay, but could be better. Check NVLink status.")
    else:
        print("[WARN] Performance is low. Run ./scripts/check_nvlink.sh")

    return True


def main():
    print("=" * 60)
    print("Qwen 3.5 27B Quick Test")
    print("=" * 60)
    print()

    if not check_health():
        sys.exit(1)

    check_models()

    if not test_generation():
        sys.exit(1)

    test_throughput()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
