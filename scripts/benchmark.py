#!/usr/bin/env python3
"""Benchmark Qwen 3.5 27B inference performance.

Measures:
- Tokens per second (decode throughput)
- Time to first token (TTFT)
- Performance across different prompt lengths

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --url http://localhost:8000 --output results/bench.json
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class BenchmarkResult:
    prompt_tokens: int
    gen_tokens: int
    elapsed_s: float
    ttft_s: float
    tokens_per_sec: float


def generate(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Send a completion request and measure performance."""
    start = time.perf_counter()

    resp = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()

    ttft = None
    tokens_generated = 0

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if ttft is None:
                    ttft = time.perf_counter() - start
                if chunk.get("choices"):
                    # Count actual tokens from finish_reason or estimate
                    tokens_generated += 1
            except json.JSONDecodeError:
                continue

    elapsed = time.perf_counter() - start

    # Use reported usage if available, otherwise estimate
    prompt_tokens = len(prompt.split())  # rough estimate

    return BenchmarkResult(
        prompt_tokens=prompt_tokens,
        gen_tokens=max_tokens,
        elapsed_s=elapsed,
        ttft_s=ttft or elapsed,
        tokens_per_sec=max_tokens / elapsed if elapsed > 0 else 0,
    )


def run_benchmark(
    url: str,
    model: str,
    prompt_lengths: list[int],
    gen_tokens: int,
    warmup_runs: int,
    bench_runs: int,
) -> dict:
    """Run full benchmark suite."""
    results = {}

    # Warmup - fills CUDA graphs
    print(f"Warming up ({warmup_runs} runs)...")
    warmup_prompt = "Hello " * 50
    for i in range(warmup_runs):
        generate(url, model, warmup_prompt, 32)
        print(f"  Warmup {i+1}/{warmup_runs}")

    print(f"\nBenchmarking with {gen_tokens} output tokens, {bench_runs} runs each\n")

    for prompt_len in prompt_lengths:
        prompt = "x " * prompt_len
        print(f"--- Prompt length: ~{prompt_len} tokens ---")

        run_results = []
        for i in range(bench_runs):
            result = generate(url, model, prompt, gen_tokens)
            run_results.append(result)
            print(f"  Run {i+1}: {result.tokens_per_sec:.1f} tok/s, TTFT: {result.ttft_s*1000:.0f}ms")

        tok_s_values = [r.tokens_per_sec for r in run_results]
        ttft_values = [r.ttft_s * 1000 for r in run_results]

        results[prompt_len] = {
            "prompt_tokens": prompt_len,
            "gen_tokens": gen_tokens,
            "runs": bench_runs,
            "tok_s_mean": statistics.mean(tok_s_values),
            "tok_s_std": statistics.stdev(tok_s_values) if len(tok_s_values) > 1 else 0,
            "tok_s_max": max(tok_s_values),
            "ttft_ms_mean": statistics.mean(ttft_values),
            "ttft_ms_p50": statistics.median(ttft_values),
        }

        print(f"  Mean: {results[prompt_len]['tok_s_mean']:.1f} tok/s "
              f"(max: {results[prompt_len]['tok_s_max']:.1f})\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen 3.5 27B")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B-GPTQ-Int4", help="Model name")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark runs per config")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[100, 500, 2000, 8000, 16000, 32000],
        help="Prompt token lengths to test",
    )
    args = parser.parse_args()

    # Check server is up
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error: Cannot connect to {args.url}")
        print(f"Start server with: ./scripts/launch-server.sh")
        sys.exit(1)

    print(f"Benchmarking {args.model}")
    print(f"Server: {args.url}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Output tokens: {args.gen_tokens}")
    print()

    results = run_benchmark(
        url=args.url,
        model=args.model,
        prompt_lengths=args.prompt_lengths,
        gen_tokens=args.gen_tokens,
        warmup_runs=args.warmup,
        bench_runs=args.runs,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Prompt':<10} {'Mean tok/s':<12} {'Max tok/s':<12} {'TTFT (ms)':<10}")
    print("-" * 60)
    for prompt_len, data in results.items():
        print(f"{prompt_len:<10} {data['tok_s_mean']:<12.1f} {data['tok_s_max']:<12.1f} {data['ttft_ms_mean']:<10.0f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "server_url": args.url,
            "config": {
                "gen_tokens": args.gen_tokens,
                "warmup_runs": args.warmup,
                "bench_runs": args.runs,
            },
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
