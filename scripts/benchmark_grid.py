#!/usr/bin/env python3
"""
Comprehensive benchmark grid for Qwen3.5-27B-GPTQ-Int4.

Tests single-user decode tok/s across:
- Context sizes: 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K
- Configurations:
  - TP=2 + INT8 KV (current setup)
  - TP=2 + FP16 KV  
  - TP=1 + INT8 KV
  - TP=1 + FP16 KV

Marks OOM conditions clearly.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Configuration
MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"
API_URL = "http://localhost:8000/v1/completions"
VENV_ACTIVATE = "source /home/yeb/Developer/qwen3.5/venv/bin/activate"
SCALES_FILE = "/home/yeb/Developer/qwen3.5/scales/qwen35_27b_per_layer.json"
CORPUS_FILE = "/home/yeb/Developer/gemma/data/dutch_parliament_text.txt"

# Test parameters
CONTEXT_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
GEN_TOKENS = 64  # Generate this many tokens for speed measurement

# Configurations to test
CONFIGS = [
    {"name": "TP2_INT8", "tp": 2, "kv_dtype": "int8", "gpus": "0,1", "int8_emul": True},
    {"name": "TP2_FP16", "tp": 2, "kv_dtype": "auto", "gpus": "0,1", "int8_emul": False},
    {"name": "TP1_INT8", "tp": 1, "kv_dtype": "int8", "gpus": "0", "int8_emul": True},
    {"name": "TP1_FP16", "tp": 1, "kv_dtype": "auto", "gpus": "0", "int8_emul": False},
]

# Results storage
results = {cfg["name"]: {} for cfg in CONFIGS}


def kill_vllm():
    """Kill any running vLLM processes."""
    subprocess.run("pkill -9 -f 'vllm serve'", shell=True, capture_output=True)
    subprocess.run("sleep 5", shell=True)
    # Clear GPU memory
    subprocess.run("nvidia-smi --gpu-reset 2>/dev/null || true", shell=True, capture_output=True)
    time.sleep(3)


def start_server(config: dict, max_model_len: int) -> bool:
    """Start vLLM server with given config. Returns True if successful."""
    kill_vllm()
    
    env_vars = [
        f"CUDA_VISIBLE_DEVICES={config['gpus']}",
    ]
    
    if config["int8_emul"]:
        env_vars.extend([
            f"VLLM_KV_SCALES_FILE={SCALES_FILE}",
            "VLLM_INT8_V_FP8_EMUL=1",
        ])
    
    cmd = f"""
{VENV_ACTIVATE} && \\
{' '.join(env_vars)} \\
nohup vllm serve {MODEL} \\
    --tensor-parallel-size {config['tp']} \\
    --kv-cache-dtype {config['kv_dtype']} \\
    --max-model-len {max_model_len} \\
    --max-num-seqs 4 \\
    --gpu-memory-utilization 0.95 \\
    --port 8000 \\
    > /tmp/vllm_bench.log 2>&1 &
"""
    
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # Wait for server to start (up to 3 minutes)
    for i in range(36):
        time.sleep(5)
        try:
            req = urllib.request.Request(
                "http://localhost:8000/v1/models",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("data"):
                    return True
        except:
            pass
        
        # Check for OOM in logs
        try:
            log = Path("/tmp/vllm_bench.log").read_text()
            if "OutOfMemoryError" in log or "CUDA out of memory" in log:
                return False
            if "Error" in log and "torch" in log.lower():
                # Check if it's a fatal error
                if "RuntimeError" in log:
                    return False
        except:
            pass
    
    return False


def measure_decode_speed(context_tokens: int) -> tuple:
    """
    Measure decode speed at given context length.
    Returns (decode_tps, prefill_tps) or (None, error_msg) on failure.
    """
    corpus = Path(CORPUS_FILE).read_text()
    
    # Approximate chars needed (4 chars per token rough estimate)
    chars_needed = context_tokens * 4
    if chars_needed > len(corpus):
        return None, "corpus_too_short"
    
    prompt = corpus[:chars_needed] + "\n\nContinue: 1, 2, 3,"
    
    try:
        # Warmup
        req = urllib.request.Request(
            API_URL,
            data=json.dumps({
                "model": MODEL,
                "prompt": "Hello",
                "max_tokens": 5,
                "temperature": 0
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            pass
        
        # Measure with two lengths to isolate decode speed
        half_gen = GEN_TOKENS // 2
        full_gen = GEN_TOKENS
        
        # First measurement
        start1 = time.perf_counter()
        req1 = urllib.request.Request(
            API_URL,
            data=json.dumps({
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": half_gen,
                "temperature": 0
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req1, timeout=300) as resp:
            r1 = json.loads(resp.read())
        t1 = time.perf_counter() - start1
        
        # Second measurement  
        start2 = time.perf_counter()
        req2 = urllib.request.Request(
            API_URL,
            data=json.dumps({
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": full_gen,
                "temperature": 0
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req2, timeout=300) as resp:
            r2 = json.loads(resp.read())
        t2 = time.perf_counter() - start2
        
        actual_context = r1["usage"]["prompt_tokens"]
        d1 = r1["usage"]["completion_tokens"]
        d2 = r2["usage"]["completion_tokens"]
        
        # Decode speed = delta tokens / delta time
        if t2 > t1 and d2 > d1:
            decode_tps = (d2 - d1) / (t2 - t1)
        else:
            decode_tps = d2 / t2
        
        # Prefill estimate
        prefill_time = t1 - (d1 / decode_tps) if decode_tps > 0 else t1
        prefill_tps = actual_context / prefill_time if prefill_time > 0 else 0
        
        return (decode_tps, prefill_tps, actual_context), None
        
    except urllib.error.URLError as e:
        return None, f"connection_error"
    except Exception as e:
        err_str = str(e).lower()
        if "out of memory" in err_str or "oom" in err_str:
            return None, "OOM"
        return None, f"error:{str(e)[:30]}"


def run_benchmark():
    """Run the full benchmark grid."""
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK GRID: Qwen3.5-27B-GPTQ-Int4")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Context sizes: {CONTEXT_SIZES}")
    print(f"Generation tokens: {GEN_TOKENS}")
    print()
    
    for config in CONFIGS:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"  TP={config['tp']}, KV={config['kv_dtype']}, GPUs={config['gpus']}")
        print("=" * 80)
        
        # Find max context that works
        max_working_context = 0
        
        for ctx in CONTEXT_SIZES:
            print(f"\n  Testing context={ctx}...", end=" ", flush=True)
            
            # Start server with appropriate max-model-len
            # Add some headroom for generation
            max_len = min(ctx + 1024, 131072)
            
            if not start_server(config, max_len):
                print("OOM (server start)")
                results[config["name"]][ctx] = {"status": "OOM", "decode_tps": None}
                continue
            
            # Measure speed
            result, error = measure_decode_speed(ctx)
            
            if error:
                print(f"FAILED ({error})")
                results[config["name"]][ctx] = {"status": error, "decode_tps": None}
                if error == "OOM":
                    # Skip larger contexts
                    for remaining_ctx in CONTEXT_SIZES[CONTEXT_SIZES.index(ctx)+1:]:
                        results[config["name"]][remaining_ctx] = {"status": "OOM (skipped)", "decode_tps": None}
                    break
            else:
                decode_tps, prefill_tps, actual_ctx = result
                print(f"OK - {decode_tps:.1f} tok/s decode, {prefill_tps:.0f} tok/s prefill (actual ctx={actual_ctx})")
                results[config["name"]][ctx] = {
                    "status": "OK",
                    "decode_tps": decode_tps,
                    "prefill_tps": prefill_tps,
                    "actual_context": actual_ctx
                }
                max_working_context = ctx
        
        print(f"\n  Max working context for {config['name']}: {max_working_context}")
    
    # Print summary grid
    print_summary_grid()
    
    # Save results
    with open("/home/yeb/Developer/qwen3.5/benchmark_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_grid_results.json")


def print_summary_grid():
    """Print formatted summary grid."""
    print("\n")
    print("=" * 100)
    print("BENCHMARK SUMMARY GRID - Single User Decode Speed (tok/s)")
    print("=" * 100)
    
    # Header
    header = f"{'Context':<10}"
    for cfg in CONFIGS:
        header += f" {cfg['name']:<15}"
    print(header)
    print("-" * 100)
    
    # Data rows
    for ctx in CONTEXT_SIZES:
        row = f"{ctx:<10}"
        for cfg in CONFIGS:
            r = results[cfg["name"]].get(ctx, {})
            if r.get("status") == "OK":
                row += f" {r['decode_tps']:>6.1f} tok/s   "
            elif "OOM" in str(r.get("status", "")):
                row += f" {'OOM':>12}   "
            else:
                row += f" {'-':>12}   "
        print(row)
    
    print("-" * 100)
    
    # Memory analysis
    print("\nKV Cache Memory Analysis (per token, per layer):")
    print("  FP16: 2 bytes × 2 (K+V) × 128 dim × 8 heads = 4096 bytes = 4 KB")
    print("  INT8: 1 byte × 2 (K+V) × 128 dim × 8 heads = 2048 bytes = 2 KB")
    print("  Savings: 50% KV cache memory with INT8")
    print()
    print("  At 64K context, 16 attention layers:")
    print("    FP16: 64K × 4KB × 16 = 4 GB KV cache per GPU")
    print("    INT8: 64K × 2KB × 16 = 2 GB KV cache per GPU")


if __name__ == "__main__":
    try:
        run_benchmark()
    finally:
        kill_vllm()
        print("\nBenchmark complete. Server stopped.")
