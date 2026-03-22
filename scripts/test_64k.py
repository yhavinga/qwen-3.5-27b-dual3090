#!/usr/bin/env python3
"""Test 64K+ context lengths."""
import time
import urllib.request
import json
from pathlib import Path

text = Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt").read_text()

def test_at_length(target_chars, gen_tokens=64):
    prompt = text[:target_chars] + "\n\nSamenvatting:"

    start = time.perf_counter()
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions",
        data=json.dumps({
            "model": "Qwen/Qwen3.5-27B-GPTQ-Int4",
            "prompt": prompt,
            "max_tokens": gen_tokens,
            "temperature": 0
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        elapsed = time.perf_counter() - start

        usage = result["usage"]
        prefill = usage["prompt_tokens"]
        decode = usage["completion_tokens"]
        output = result["choices"][0]["text"][:100]

        return prefill, decode, elapsed, output
    except Exception as e:
        return None, None, None, str(e)

print("64K Context Test with INT8 KV + FP8-V + Per-Layer Scales")
print("=" * 70)
print(f"Corpus size: {len(text)} characters")
print()

for chars in [16000, 64000, 128000]:
    if chars > len(text):
        print(f"{chars:>7} chars: Skipped (corpus only {len(text)} chars)")
        continue

    prefill, decode, elapsed, output = test_at_length(chars)
    if prefill is None:
        print(f"{chars:>7} chars: ERROR - {output}")
    else:
        print(f"{chars:>7} chars = {prefill:>5} tokens")
        print(f"         Time: {elapsed:.1f}s total")
        print(f"         Speed: ~{prefill/elapsed:.0f} tok/s prefill, {decode/elapsed:.1f} tok/s decode")
        clean_output = output[:70].replace("\n", " ")
        print(f"         Output: {clean_output}...")
        print()
