#!/usr/bin/env python3
"""Stress test FP8-V emulation at long context lengths."""

import urllib.request
import json
import time
from pathlib import Path

# Load the full parliamentary text
text = Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt").read_text()

API_URL = "http://localhost:8000/v1/completions"

def test_context(chars, gen_tokens=128):
    prompt = text[:chars] + "\n\nSamenvatting van bovenstaande tekst in maximaal 3 zinnen:"
    payload = {
        "model": "Qwen/Qwen3.5-27B-GPTQ-Int4",
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0.0,
    }
    start = time.time()
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - start

    usage = result["usage"]
    output = result["choices"][0]["text"]

    return {
        "context_chars": chars,
        "prompt_tokens": usage["prompt_tokens"],
        "gen_tokens": usage["completion_tokens"],
        "elapsed": elapsed,
        "tok_per_s": usage["completion_tokens"] / elapsed,
        "output": output[:300],
    }

print("Context Length Stress Test with FP8-V Emulation")
print("=" * 70)
print("Context    Tokens   Speed        Output Preview")
print("-" * 70)

for chars in [4000, 16000, 32000, 64000, 100000]:
    if chars > len(text):
        chars = len(text)
    try:
        r = test_context(chars)
        preview = r["output"].replace("\n", " ")[:50]
        ptoks = r["prompt_tokens"]
        tps = r["tok_per_s"]
        print(f"{chars:<10} {ptoks:<8} {tps:.1f} tok/s    {preview}...")
    except Exception as e:
        print(f"{chars:<10} ERROR: {e}")

print("-" * 70)
print("If outputs are coherent at all context lengths, FP8-V emulation is working!")
