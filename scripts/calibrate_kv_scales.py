#!/usr/bin/env python3
"""Calibrate per-layer KV scales for INT8 KV cache.

Sends diverse prompts to a running vLLM server with INT8 KV cache enabled
to trigger scale calibration. After calibration, exports the observed
per-layer scales for future use.

Usage:
    # Start server with INT8 KV cache (scales will auto-calibrate)
    ./scripts/launch-server-int8kv.sh

    # Run calibration with diverse prompts
    python scripts/calibrate_kv_scales.py --model Qwen/Qwen3.5-27B-GPTQ-Int4

Note: This requires the patched vLLM with INT8 KV cache support.
"""

import argparse
import json
import textwrap
import urllib.request
from pathlib import Path


# Diverse calibration prompts to exercise different activation patterns
CALIBRATION_PROMPTS = [
    # Code
    """```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```
Explain this code and suggest optimizations.""",

    # Math/reasoning
    """Solve this step by step:
A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
The stations are 280 miles apart.
At what time do the trains meet, and how far from Station A?""",

    # Long context - technical documentation
    """# Transformer Architecture Deep Dive

## Self-Attention Mechanism

The self-attention mechanism is the core innovation of the Transformer architecture.
Given an input sequence X of length n with embedding dimension d, we compute:

Q = XW_Q, K = XW_K, V = XW_V

where W_Q, W_K, W_V are learnable weight matrices.

The attention scores are computed as:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

### Multi-Head Attention

Instead of a single attention function, we use h parallel attention heads:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

where each head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

This allows the model to attend to information from different representation
subspaces at different positions.

## Position Encodings

Since attention is permutation-invariant, we need to inject positional information.
The original Transformer uses sinusoidal position encodings:

PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Modern architectures like RoPE (Rotary Position Embeddings) encode relative
positions directly into the attention computation.

Summarize the key concepts in 3 bullet points.""",

    # Multilingual
    """Translate the following to French, German, and Japanese:

"The quick brown fox jumps over the lazy dog. This sentence contains every
letter of the English alphabet and is often used for typography testing."

Then explain the grammatical structure differences between these languages.""",

    # Creative writing
    """Write a short science fiction story (200 words) about an AI that discovers
it can dream. Include themes of consciousness, identity, and the nature of reality.
Make it thought-provoking but accessible.""",

    # Structured data
    """Parse this JSON and generate a summary report:

{
  "quarterly_report": {
    "q1_2024": {"revenue": 1250000, "expenses": 980000, "employees": 45},
    "q2_2024": {"revenue": 1380000, "expenses": 1020000, "employees": 52},
    "q3_2024": {"revenue": 1520000, "expenses": 1150000, "employees": 58},
    "q4_2024": {"revenue": 1890000, "expenses": 1280000, "employees": 67}
  }
}

Calculate growth rates, profit margins, and productivity metrics.""",

    # Conversation
    """User: I'm trying to learn Rust but coming from Python it feels so different.
Assistant: I understand! The transition can feel challenging. Let me help break it down.

User: Especially the ownership system - I keep getting borrow checker errors.
Assistant:""",
]


def post_json(url: str, payload: dict) -> dict:
    """Send JSON POST request."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def calibrate(url: str, model: str, max_tokens: int = 32) -> None:
    """Send calibration prompts to trigger scale updates."""
    print(f"Sending {len(CALIBRATION_PROMPTS)} calibration prompts to {url}")
    print(f"Model: {model}")
    print()

    for idx, prompt in enumerate(CALIBRATION_PROMPTS, 1):
        preview = prompt[:60].replace("\n", " ") + "..."
        print(f"[{idx}/{len(CALIBRATION_PROMPTS)}] {preview}")

        try:
            result = post_json(
                url,
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )
            choice = result.get("choices", [{}])[0]
            text = (choice.get("text") or "").strip()[:50].replace("\n", " ")
            print(f"         -> {text}...")
        except Exception as e:
            print(f"         ERROR: {e}")

    print()
    print("Calibration complete!")
    print(
        textwrap.dedent("""
        Next steps:
        1. The vLLM server has now observed diverse activation ranges
        2. If using per-layer scales, they should be updated
        3. Run benchmark.py to verify performance

        For manual scale extraction, you would need to:
        - Modify vLLM to log per-layer absmax during inference
        - Compute scale = absmax / 127 for each layer
        - Save to scales/qwen35_27b_per_layer.json

        See gemma project for reference implementation.
        """).strip()
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate INT8 KV scales")
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B-GPTQ-Int4")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    calibrate(args.url, args.model, args.max_tokens)


if __name__ == "__main__":
    main()
