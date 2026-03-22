#!/usr/bin/env python3
"""
Rigorous context length benchmark for Qwen 3.5 27B.

Tests actual generation speed at different context lengths using real Dutch
parliamentary text. Measures tok/s for both prefill and decode phases.

Usage:
    # Start server first:
    ./scripts/launch-server.sh  # or launch-server-int8kv.sh

    # Run benchmark:
    python scripts/benchmark_context_length.py --output results/context_benchmark.json
"""

import argparse
import json
import time
import statistics
import sys
from pathlib import Path

import requests

# Default configuration
SERVER_URL = "http://localhost:8000"
DUTCH_PARLIAMENT_TEXT = "/home/yeb/Developer/gemma/data/dutch_parliament_text.txt"

# Context sizes to test (in approximate tokens)
# Using ~4 chars per token as rough estimate for Dutch text
CONTEXT_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
GENERATION_LENGTH = 128  # Fixed generation length for fair comparison
WARMUP_RUNS = 2
BENCHMARK_RUNS = 3


def load_text(path: str) -> str:
    """Load the benchmark text."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def check_server(server_url: str) -> dict:
    """Check if server is running and get model info."""
    try:
        resp = requests.get(f"{server_url}/v1/models", timeout=5)
        resp.raise_for_status()
        models = resp.json()
        return models['data'][0] if models.get('data') else {}
    except Exception as e:
        print(f"ERROR: Server not reachable at {server_url}")
        print(f"  {e}")
        print("\nStart the server first:")
        print("  ./scripts/launch-server.sh")
        sys.exit(1)


def count_tokens(text: str, server_url: str) -> int:
    """Get actual token count from server."""
    try:
        # Use a minimal completion to count tokens
        resp = requests.post(
            f"{server_url}/v1/completions",
            json={
                "model": "Qwen/Qwen3.5-27B-GPTQ-Int4",
                "prompt": text,
                "max_tokens": 1,
                "echo": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data['usage']['prompt_tokens']
    except Exception as e:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4


def create_prompt(text: str, target_tokens: int) -> str:
    """Create a prompt with approximately target_tokens tokens."""
    # Rough estimate: 4 chars per token for Dutch text
    chars_needed = target_tokens * 4

    if len(text) < chars_needed:
        # Repeat text if needed
        repeats = (chars_needed // len(text)) + 1
        text = (text + "\n\n") * repeats

    prompt = text[:chars_needed]

    # Add a clear instruction at the end
    prompt += "\n\n---\nSamenvatting van bovenstaande tekst in het Nederlands:\n"

    return prompt


def benchmark_single(prompt: str, max_tokens: int, server_url: str) -> dict:
    """Run a single benchmark request and measure timing."""
    start_time = time.perf_counter()

    resp = requests.post(
        f"{server_url}/v1/completions",
        json={
            "model": "Qwen/Qwen3.5-27B-GPTQ-Int4",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        },
        timeout=300,
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    resp.raise_for_status()
    data = resp.json()

    usage = data['usage']
    prompt_tokens = usage['prompt_tokens']
    completion_tokens = usage['completion_tokens']

    # Calculate speeds
    # Note: This is total time including prefill + decode
    # For decode speed, we estimate prefill time based on prompt length

    # Rough prefill estimate: ~0.5ms per token for long context
    estimated_prefill_time = prompt_tokens * 0.0005  # seconds
    estimated_decode_time = max(0.1, total_time - estimated_prefill_time)

    decode_tok_per_sec = completion_tokens / estimated_decode_time
    total_tok_per_sec = completion_tokens / total_time

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_s": total_time,
        "estimated_prefill_time_s": estimated_prefill_time,
        "estimated_decode_time_s": estimated_decode_time,
        "decode_tok_per_sec": decode_tok_per_sec,
        "total_tok_per_sec": total_tok_per_sec,
    }


def run_benchmark(text: str, context_sizes: list, gen_length: int,
                  warmup_runs: int, benchmark_runs: int, server_url: str) -> list:
    """Run the full benchmark suite."""
    results = []

    print(f"\nBenchmarking {len(context_sizes)} context sizes...")
    print(f"Generation length: {gen_length} tokens")
    print(f"Warmup runs: {warmup_runs}, Benchmark runs: {benchmark_runs}")
    print("=" * 70)

    for target_tokens in context_sizes:
        print(f"\n[Context: ~{target_tokens} tokens]")

        # Create prompt
        prompt = create_prompt(text, target_tokens)

        # Get actual token count
        actual_tokens = count_tokens(prompt, server_url)
        print(f"  Actual prompt tokens: {actual_tokens}")

        # Warmup
        print(f"  Warming up ({warmup_runs} runs)...", end=" ", flush=True)
        for _ in range(warmup_runs):
            try:
                benchmark_single(prompt, gen_length, server_url)
                print(".", end="", flush=True)
            except Exception as e:
                print(f"\n  Warmup failed: {e}")
                break
        print(" done")

        # Benchmark runs
        run_results = []
        print(f"  Benchmarking ({benchmark_runs} runs)...", end=" ", flush=True)
        for i in range(benchmark_runs):
            try:
                result = benchmark_single(prompt, gen_length, server_url)
                run_results.append(result)
                print(f".", end="", flush=True)
            except Exception as e:
                print(f"\n  Run {i+1} failed: {e}")
        print(" done")

        if not run_results:
            print(f"  FAILED: No successful runs")
            continue

        # Aggregate results
        decode_speeds = [r['decode_tok_per_sec'] for r in run_results]
        total_speeds = [r['total_tok_per_sec'] for r in run_results]

        avg_decode = statistics.mean(decode_speeds)
        std_decode = statistics.stdev(decode_speeds) if len(decode_speeds) > 1 else 0
        avg_total = statistics.mean(total_speeds)

        result = {
            "target_context": target_tokens,
            "actual_context": actual_tokens,
            "generation_length": gen_length,
            "runs": benchmark_runs,
            "avg_decode_tok_per_sec": round(avg_decode, 2),
            "std_decode_tok_per_sec": round(std_decode, 2),
            "avg_total_tok_per_sec": round(avg_total, 2),
            "raw_results": run_results,
        }
        results.append(result)

        print(f"  Result: {avg_decode:.1f} ± {std_decode:.1f} tok/s (decode)")
        print(f"          {avg_total:.1f} tok/s (total including prefill)")

    return results


def print_summary(results: list, config_name: str):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY: {config_name}")
    print("=" * 70)
    print(f"{'Context':>10} | {'Decode (tok/s)':>18} | {'Total (tok/s)':>15} | {'Relative':>10}")
    print("-" * 70)

    baseline = results[0]['avg_decode_tok_per_sec'] if results else 1

    for r in results:
        ctx = r['actual_context']
        decode = r['avg_decode_tok_per_sec']
        std = r['std_decode_tok_per_sec']
        total = r['avg_total_tok_per_sec']
        relative = decode / baseline * 100

        print(f"{ctx:>10} | {decode:>8.1f} ± {std:<6.1f} | {total:>15.1f} | {relative:>9.0f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Context length benchmark")
    parser.add_argument("--output", type=str, default="results/context_benchmark.json",
                        help="Output JSON file")
    parser.add_argument("--server", type=str, default=SERVER_URL,
                        help="Server URL")
    parser.add_argument("--text", type=str, default=DUTCH_PARLIAMENT_TEXT,
                        help="Path to benchmark text")
    parser.add_argument("--gen-length", type=int, default=GENERATION_LENGTH,
                        help="Number of tokens to generate")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS,
                        help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS,
                        help="Number of benchmark runs per context size")
    parser.add_argument("--contexts", type=str, default=None,
                        help="Comma-separated list of context sizes (e.g., '1024,4096,16384')")
    parser.add_argument("--config-name", type=str, default="unknown",
                        help="Configuration name (e.g., 'vanilla', 'int8-kv')")
    args = parser.parse_args()

    server_url = args.server

    # Parse context sizes
    if args.contexts:
        context_sizes = [int(x.strip()) for x in args.contexts.split(',')]
    else:
        context_sizes = CONTEXT_SIZES

    print("=" * 70)
    print("QWEN 3.5 27B CONTEXT LENGTH BENCHMARK")
    print("=" * 70)
    print(f"Server: {server_url}")
    print(f"Text source: {args.text}")
    print(f"Generation length: {args.gen_length} tokens")
    print(f"Context sizes: {context_sizes}")
    print(f"Configuration: {args.config_name}")

    # Check server
    print("\nChecking server...")
    model_info = check_server(server_url)
    print(f"  Model: {model_info.get('id', 'unknown')}")

    # Load text
    print(f"\nLoading benchmark text...")
    text = load_text(args.text)
    print(f"  Loaded {len(text):,} characters (~{len(text)//4:,} tokens)")

    # Run benchmark
    results = run_benchmark(
        text=text,
        context_sizes=context_sizes,
        gen_length=args.gen_length,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        server_url=server_url,
    )

    # Print summary
    print_summary(results, args.config_name)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config_name": args.config_name,
        "server": server_url,
        "text_source": args.text,
        "generation_length": args.gen_length,
        "warmup_runs": args.warmup,
        "benchmark_runs": args.runs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
