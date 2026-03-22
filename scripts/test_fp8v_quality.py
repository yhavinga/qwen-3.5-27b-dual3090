#!/usr/bin/env python3
"""Test FP8-V emulation quality and speed at various context lengths."""

import requests
import time
import json
import sys

API_URL = "http://localhost:8000/v1/completions"

def test_completion(prompt: str, max_tokens: int = 128, temperature: float = 0.0):
    """Run completion and return (text, tokens/sec, prompt_tokens)."""
    start = time.perf_counter()
    resp = requests.post(API_URL, json={
        "model": "Qwen/Qwen3.5-27B-GPTQ-Int4",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    elapsed = time.perf_counter() - start

    data = resp.json()
    if "error" in data:
        return None, 0, 0

    text = data["choices"][0]["text"]
    usage = data["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    tok_per_sec = completion_tokens / elapsed

    return text, tok_per_sec, prompt_tokens

def load_context(path: str) -> str:
    with open(path) as f:
        return f.read()

def main():
    # Load contexts
    context_4k = load_context("/tmp/context_4k.txt")
    context_full = load_context("/tmp/context_full.txt")

    print("=" * 70)
    print("FP8-V Emulation Quality & Speed Test")
    print("=" * 70)

    # Test 1: Short context quality
    print("\n[Test 1] Short context - technical question")
    prompt1 = "Explain in detail how attention mechanisms work in transformers, including the query, key, value computations and softmax normalization:"
    text, tps, ptoks = test_completion(prompt1, max_tokens=200, temperature=0.0)
    print(f"  Prompt tokens: {ptoks}, Speed: {tps:.1f} tok/s")
    print(f"  Output:\n{text[:500]}...")

    # Test 2: Medium context (~4K tokens)
    print("\n[Test 2] ~4K context - summarization")
    prompt2 = context_4k + "\n\nSamenvatting van bovenstaande tekst in het Nederlands:"
    text, tps, ptoks = test_completion(prompt2, max_tokens=150, temperature=0.0)
    print(f"  Prompt tokens: {ptoks}, Speed: {tps:.1f} tok/s")
    print(f"  Output:\n{text[:500]}...")

    # Test 3: Long context quality check
    print("\n[Test 3] Long context (~8K) - comprehension")
    context_8k = context_full[:32000]  # ~8K tokens
    prompt3 = context_8k + "\n\nWat zijn de belangrijkste onderwerpen die in deze parlementaire tekst worden besproken?"
    text, tps, ptoks = test_completion(prompt3, max_tokens=200, temperature=0.0)
    print(f"  Prompt tokens: {ptoks}, Speed: {tps:.1f} tok/s")
    print(f"  Output:\n{text[:600]}...")

    # Test 4: Speed scaling test
    print("\n[Test 4] Speed vs context length")
    print("  Context | Tokens | Speed (tok/s)")
    print("  --------|--------|---------------")

    for ctx_chars, label in [(4000, "1K"), (8000, "2K"), (16000, "4K"), (32000, "8K")]:
        ctx = context_full[:ctx_chars]
        prompt = ctx + "\n\nVat samen:"
        _, tps, ptoks = test_completion(prompt, max_tokens=64, temperature=0.0)
        print(f"  {label:>7} | {ptoks:>6} | {tps:.1f}")

    # Test 5: Coherence at long generation
    print("\n[Test 5] Long generation coherence (256 tokens)")
    prompt5 = "Write a detailed technical explanation of how hybrid Mamba-Transformer architectures combine the benefits of both approaches:"
    text, tps, ptoks = test_completion(prompt5, max_tokens=256, temperature=0.0)
    print(f"  Speed: {tps:.1f} tok/s")
    print(f"  Output:\n{text}")

    print("\n" + "=" * 70)
    print("MANUAL CHECK: Review outputs for coherence degradation")
    print("If outputs become nonsensical at longer contexts, FP8-V scales need calibration")
    print("=" * 70)

if __name__ == "__main__":
    main()
