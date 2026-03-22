#!/usr/bin/env python3
"""
Final benchmark grid - fixed measurement approach.
Uses absolute timing rather than differential.
"""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"
API = "http://localhost:8000/v1/completions"
CORPUS = Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt").read_text()
SCALES = "/home/yeb/Developer/qwen3.5/scales/qwen35_27b_per_layer.json"

def kill_server():
    os.system("pkill -9 -f 'vllm serve' 2>/dev/null")
    time.sleep(3)
    result = subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader",
                          shell=True, capture_output=True, text=True)
    for pid in result.stdout.strip().split('\n'):
        if pid.strip() and pid.strip().isdigit():
            os.system(f"kill -9 {pid.strip()} 2>/dev/null")
    time.sleep(2)

def start_server(cfg):
    kill_server()

    env = f"CUDA_VISIBLE_DEVICES={cfg['gpus']}"
    if cfg["scales"]:
        env += f" VLLM_KV_SCALES_FILE={SCALES} VLLM_INT8_V_FP8_EMUL=1"

    cmd = f""". /home/yeb/Developer/qwen3.5/venv/bin/activate && \
{env} nohup vllm serve {MODEL} \
    --tensor-parallel-size {cfg['tp']} \
    --kv-cache-dtype {cfg['kv']} \
    --max-model-len {cfg['max_ctx']} \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    > /tmp/vllm_{cfg['name']}.log 2>&1 &
"""
    subprocess.run(cmd, shell=True, executable='/bin/bash')

    print(f"  Starting {cfg['name']}...", end=" ", flush=True)
    for i in range(90):
        time.sleep(2)
        try:
            req = urllib.request.Request("http://localhost:8000/v1/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                if json.loads(r.read()).get("data"):
                    print("OK")
                    return True
        except:
            pass

        try:
            log = Path(f"/tmp/vllm_{cfg['name']}.log").read_text()
            if "OutOfMemory" in log or "CUDA out of memory" in log:
                print("OOM")
                return False
        except:
            pass

    print("TIMEOUT")
    return False

def measure(ctx_tokens, gen_tokens=100):
    """Improved measurement - runs single request, estimates decode speed."""
    chars = min(ctx_tokens * 4, len(CORPUS) - 100)
    prompt = CORPUS[:chars] + "\n\nContinue counting: 1, 2, 3,"

    # Single request, measure total time
    start = time.perf_counter()
    req = urllib.request.Request(
        API,
        data=json.dumps({
            "model": MODEL, "prompt": prompt,
            "max_tokens": gen_tokens, "temperature": 0
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        result = json.loads(r.read())
    total_time = time.perf_counter() - start

    ctx = result["usage"]["prompt_tokens"]
    gen = result["usage"]["completion_tokens"]

    # Estimate prefill time based on ~2000 tok/s for TP2
    prefill_est = ctx / 2000 if ctx > 0 else 0
    decode_time = max(0.1, total_time - prefill_est)
    decode_tps = gen / decode_time

    # Also compute overall throughput
    overall_tps = gen / total_time

    return decode_tps, overall_tps, ctx, gen, total_time

# Configurations - only test what works
CONFIGS = [
    {"name": "TP2_INT8", "tp": 2, "kv": "int8", "gpus": "0,1", "scales": True, "max_ctx": 65536},
    {"name": "TP2_FP16", "tp": 2, "kv": "auto", "gpus": "0,1", "scales": False, "max_ctx": 65536},
]

CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
RESULTS = {}

print("=" * 90)
print("BENCHMARK GRID: Qwen3.5-27B-GPTQ-Int4 on Dual RTX 3090")
print("=" * 90)
print()
print("Note: TP=1 (single GPU) is ALWAYS OOM - model needs ~14GB, insufficient room for KV cache")
print()

for cfg in CONFIGS:
    print(f"\n{'='*70}")
    print(f"Configuration: {cfg['name']} (TP={cfg['tp']}, KV={cfg['kv']})")
    print("=" * 70)

    RESULTS[cfg["name"]] = {}

    if not start_server(cfg):
        for ctx in CONTEXTS:
            RESULTS[cfg["name"]][ctx] = {"status": "OOM"}
        continue

    # Warmup
    try:
        req = urllib.request.Request(
            API,
            data=json.dumps({"model": MODEL, "prompt": "Hi", "max_tokens": 10, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60):
            pass
    except:
        pass
    time.sleep(2)

    print()
    print(f"{'Target':>8} {'Actual':>8} {'GenTok':>8} {'Time':>8} {'Decode':>12} {'Overall':>12}")
    print("-" * 70)

    for ctx in CONTEXTS:
        if ctx > cfg["max_ctx"]:
            print(f"{ctx:>8} {'SKIP':>8} - exceeds max_model_len")
            RESULTS[cfg["name"]][ctx] = {"status": "SKIP"}
            continue

        try:
            decode_tps, overall_tps, actual_ctx, gen, total_time = measure(ctx, gen_tokens=100)
            print(f"{ctx:>8} {actual_ctx:>8} {gen:>8} {total_time:>7.1f}s {decode_tps:>8.1f} tok/s {overall_tps:>8.1f} tok/s")
            RESULTS[cfg["name"]][ctx] = {
                "status": "OK",
                "actual_ctx": actual_ctx,
                "gen_tokens": gen,
                "total_time": round(total_time, 2),
                "decode_tps": round(decode_tps, 1),
                "overall_tps": round(overall_tps, 1)
            }
        except Exception as e:
            err = str(e).lower()
            if "memory" in err or "oom" in err:
                print(f"{ctx:>8} OOM")
                RESULTS[cfg["name"]][ctx] = {"status": "OOM"}
                # Mark remaining as OOM
                idx = CONTEXTS.index(ctx)
                for remaining in CONTEXTS[idx+1:]:
                    RESULTS[cfg["name"]][remaining] = {"status": "OOM"}
                break
            else:
                print(f"{ctx:>8} ERROR: {str(e)[:40]}")
                RESULTS[cfg["name"]][ctx] = {"status": "ERROR", "error": str(e)[:100]}

# Summary grid
print("\n")
print("=" * 100)
print("SUMMARY GRID - Single User Decode Speed (tok/s)")
print("=" * 100)
print()
print("Note: TP1 (single GPU) configurations are ALWAYS OOM for this model")
print()

header = f"{'Context':<10} {'TP2_INT8':>15} {'TP2_FP16':>15} {'TP1_INT8':>15} {'TP1_FP16':>15}"
print(header)
print("-" * 100)

for ctx in CONTEXTS:
    row = f"{ctx:<10}"
    for name in ["TP2_INT8", "TP2_FP16", "TP1_INT8", "TP1_FP16"]:
        if name.startswith("TP1"):
            row += f"{'OOM':>15}"
        else:
            r = RESULTS.get(name, {}).get(ctx, {})
            if r.get("status") == "OK":
                row += f"{r['decode_tps']:>12.1f}   "
            elif r.get("status") == "OOM":
                row += f"{'OOM':>15}"
            elif r.get("status") == "SKIP":
                row += f"{'SKIP':>15}"
            else:
                row += f"{'-':>15}"
    print(row)

print("-" * 100)
print()
print("Legend:")
print("  TP2 = 2 GPUs with tensor parallelism via NVLink")
print("  TP1 = 1 GPU (ALWAYS OOM for 27B model)")
print("  INT8 = INT8 KV cache with per-layer FP8-V emulation")
print("  FP16 = Standard FP16 KV cache")
print("  OOM = Out of memory during model loading or inference")
print()
print("Hardware: 2x RTX 3090 (24GB each), NVLink")
print("Model: Qwen3.5-27B-GPTQ-Int4 (~14GB weights)")

# Save
with open("/home/yeb/Developer/qwen3.5/benchmark_grid_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\nResults saved to benchmark_grid_results.json")

kill_server()
