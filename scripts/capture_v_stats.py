#!/usr/bin/env python3
"""
Capture V value statistics from vLLM server by patching the attention kernel.

This adds temporary logging to track V absmax per layer during inference,
then restores the original kernel.

Usage:
    1. Start vLLM server with INT8 KV cache
    2. Run: python scripts/capture_v_stats.py
    3. Server will log V stats during calibration prompts
    4. Stats are saved to scales/v_stats_qwen35.json
"""

import json
import os
import sys
import time
from pathlib import Path

# Add vllm source to path
VLLM_SOURCE = Path("/home/yeb/Developer/qwen3.5/vllm-source")
sys.path.insert(0, str(VLLM_SOURCE))


def patch_kernel_for_logging():
    """
    Patch triton_reshape_and_cache_flash.py to log V absmax per layer.

    Returns the patch that was applied (for reverting).
    """
    kernel_file = VLLM_SOURCE / "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"

    # Read current content
    content = kernel_file.read_text()

    # Check if already patched for logging
    if "V_STATS_LOG" in content:
        print("Kernel already patched for logging")
        return None

    # We need to add logging after the scale is computed
    # This is tricky because it's a Triton kernel

    print("NOTE: Triton kernels can't do runtime logging directly.")
    print("Instead, we'll estimate V ranges from the server's dynamic scale computation.")
    return None


def estimate_v_ranges_from_api(
    base_url: str = "http://localhost:8000",
    model: str = "Qwen/Qwen3.5-27B-GPTQ-Int4",
    num_prompts: int = 5,
    prompt_length: int = 2000,
):
    """
    Run inference and estimate V ranges based on output quality.

    Theory: If V variance is high and global scale is wrong, we'll see:
    - Quality degradation at certain layers (later layers often have higher variance)
    - Repetition or incoherence at specific context lengths

    We can also check if vLLM exposes any internal metrics.
    """
    import urllib.request

    def post_json(url, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    # Load calibration text
    text_file = Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt")
    if text_file.exists():
        text = text_file.read_text()[:prompt_length * num_prompts]
    else:
        text = "The quick brown fox " * (prompt_length * num_prompts // 4)

    results = []
    url = f"{base_url}/v1/completions"

    print(f"Running {num_prompts} test prompts...")
    for i in range(num_prompts):
        start_idx = i * prompt_length
        prompt = text[start_idx:start_idx + prompt_length]

        payload = {
            "model": model,
            "prompt": prompt + "\n\nSummary:",
            "max_tokens": 64,
            "temperature": 0.0,
        }

        try:
            start = time.time()
            resp = post_json(url, payload)
            elapsed = time.time() - start

            output = resp["choices"][0]["text"]
            usage = resp["usage"]

            # Check for signs of quantization issues
            issues = []
            if output.count(output[:20]) > 2 and len(output) > 40:
                issues.append("repetition")
            if "????" in output or "####" in output:
                issues.append("garbage")

            results.append({
                "prompt_idx": i,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "elapsed_s": elapsed,
                "tok_per_s": usage["completion_tokens"] / elapsed,
                "output_preview": output[:100],
                "issues": issues,
            })
            print(f"  [{i+1}/{num_prompts}] {usage['prompt_tokens']} tokens, {usage['completion_tokens']/elapsed:.1f} tok/s")

        except Exception as e:
            print(f"  [{i+1}/{num_prompts}] Error: {e}")
            results.append({"prompt_idx": i, "error": str(e)})

    return results


def check_prometheus_metrics(base_url: str = "http://localhost:8000"):
    """Check if vLLM exposes any KV cache metrics via Prometheus."""
    import urllib.request

    metrics_url = f"{base_url}/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as resp:
            metrics = resp.read().decode()

        # Look for KV cache related metrics
        kv_metrics = [line for line in metrics.split("\n")
                      if "kv" in line.lower() or "cache" in line.lower()]

        if kv_metrics:
            print("\nKV-related metrics from Prometheus:")
            for m in kv_metrics[:20]:
                print(f"  {m}")
            return kv_metrics
        else:
            print("\nNo KV-related metrics found in Prometheus endpoint")
    except Exception as e:
        print(f"\nCouldn't fetch metrics: {e}")

    return []


def main():
    print("=" * 60)
    print("V Statistics Capture for Qwen3.5-27B")
    print("=" * 60)

    # Check Prometheus metrics
    kv_metrics = check_prometheus_metrics()

    # Run inference tests
    print("\nRunning inference tests to check for quantization issues...")
    results = estimate_v_ranges_from_api()

    # Save results
    output_path = Path("scales/v_inference_stats.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved inference stats to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    errors = [r for r in results if "error" in r]
    issues = [r for r in results if r.get("issues")]

    if errors:
        print(f"Errors: {len(errors)}/{len(results)} prompts failed")
    if issues:
        print(f"Quality issues detected in {len(issues)} prompts:")
        for r in issues:
            print(f"  Prompt {r['prompt_idx']}: {r['issues']}")
    else:
        print("No obvious quality issues detected!")
        print("\nThis suggests:")
        print("  1. V variance in Qwen3.5 may be lower than Gemma's 340x")
        print("  2. Or FP8-V emulation is handling the variance correctly")
        print("  3. Or the GPTQ quantization already constrains activation ranges")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
To definitively measure V ranges:

Option A: Modify vLLM triton kernel to record absmax
  - Add tl.atomic_max to track per-layer V absmax
  - Run calibration, extract logged values
  - Revert kernel to original

Option B: Profile original BF16 model with HuggingFace
  - Requires ~54GB VRAM (won't fit on 2x 3090)
  - Could use CPU offloading (slow)

Option C: Compare quality systematically
  - BF16 KV vs INT8 KV vs INT8+FP8V
  - Use perplexity or downstream task metrics
  - If quality matches, variance handling is adequate
""")


if __name__ == "__main__":
    main()
