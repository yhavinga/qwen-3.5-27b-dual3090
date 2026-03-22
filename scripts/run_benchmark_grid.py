#!/usr/bin/env python3
"""
Comprehensive benchmark grid for Qwen3.5-27B-GPTQ-Int4.
Tests single-user decode tok/s across configurations and context sizes.
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

CONFIGS = [
    {"name": "TP2_INT8", "tp": 2, "kv": "int8", "gpus": "0,1", "scales": True, "max_ctx": 65536},
    {"name": "TP2_FP16", "tp": 2, "kv": "auto", "gpus": "0,1", "scales": False, "max_ctx": 65536},
    {"name": "TP1_INT8", "tp": 1, "kv": "int8", "gpus": "0", "scales": True, "max_ctx": 32768},
    {"name": "TP1_FP16", "tp": 1, "kv": "auto", "gpus": "0", "scales": False, "max_ctx": 16384},
]

CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
RESULTS = {}

def kill_server():
    os.system("pkill -9 -f 'vllm serve' 2>/dev/null; sleep 3")
    # Kill remaining workers
    result = subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader", 
                          shell=True, capture_output=True, text=True)
    for pid in result.stdout.strip().split('\n'):
        if pid.strip():
            os.system(f"kill -9 {pid.strip()} 2>/dev/null")
    time.sleep(3)

def start_server(cfg):
    kill_server()
    
    env = f"CUDA_VISIBLE_DEVICES={cfg['gpus']}"
    if cfg["scales"]:
        env += f" VLLM_KV_SCALES_FILE={SCALES} VLLM_INT8_V_FP8_EMUL=1"
    
    cmd = f"""
. /home/yeb/Developer/qwen3.5/venv/bin/activate && \
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
    
    # Wait for server
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
        
        # Check for OOM
        try:
            log = Path(f"/tmp/vllm_{cfg['name']}.log").read_text()
            if "OutOfMemory" in log or "CUDA out of memory" in log or "less than desired GPU memory" in log:
                print("OOM")
                return False
        except:
            pass
    
    print("TIMEOUT")
    return False

def measure(ctx_tokens):
    chars = min(ctx_tokens * 4, len(CORPUS) - 100)
    prompt = CORPUS[:chars] + "\n\nContinue:"
    
    # Two measurements to isolate decode
    times, gens, ctxs = [], [], []
    for gen in [32, 64]:
        start = time.perf_counter()
        req = urllib.request.Request(
            API,
            data=json.dumps({
                "model": MODEL, "prompt": prompt,
                "max_tokens": gen, "temperature": 0
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read())
        times.append(time.perf_counter() - start)
        gens.append(result["usage"]["completion_tokens"])
        ctxs.append(result["usage"]["prompt_tokens"])
    
    if times[1] > times[0] and gens[1] > gens[0]:
        decode_tps = (gens[1] - gens[0]) / (times[1] - times[0])
    else:
        decode_tps = gens[1] / times[1]
    
    return decode_tps, ctxs[0]

# Main benchmark
print("=" * 80)
print("BENCHMARK GRID: Qwen3.5-27B-GPTQ-Int4 Single User Decode Speed")
print("=" * 80)
print()

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Configuration: {cfg['name']} (TP={cfg['tp']}, KV={cfg['kv']})")
    print("=" * 60)
    
    RESULTS[cfg["name"]] = {}
    
    if not start_server(cfg):
        for ctx in CONTEXTS:
            RESULTS[cfg["name"]][ctx] = "OOM"
        continue
    
    # Warmup
    try:
        req = urllib.request.Request(
            API,
            data=json.dumps({"model": MODEL, "prompt": "Hi", "max_tokens": 5, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30):
            pass
    except:
        pass
    
    # Test each context
    oom_hit = False
    for ctx in CONTEXTS:
        if ctx > cfg["max_ctx"]:
            print(f"    {ctx:>6} tokens: SKIP (> max_ctx {cfg['max_ctx']})")
            RESULTS[cfg["name"]][ctx] = "SKIP"
            continue
        
        if oom_hit:
            print(f"    {ctx:>6} tokens: OOM (skipped)")
            RESULTS[cfg["name"]][ctx] = "OOM"
            continue
        
        print(f"    {ctx:>6} tokens: ", end="", flush=True)
        try:
            tps, actual = measure(ctx)
            print(f"{tps:>5.1f} tok/s (ctx={actual})")
            RESULTS[cfg["name"]][ctx] = round(tps, 1)
        except Exception as e:
            err = str(e).lower()
            if "memory" in err or "oom" in err:
                print("OOM")
                RESULTS[cfg["name"]][ctx] = "OOM"
                oom_hit = True
            else:
                print(f"ERR: {str(e)[:40]}")
                RESULTS[cfg["name"]][ctx] = "ERR"

# Summary grid
print("\n")
print("=" * 90)
print("SUMMARY GRID - Single User Decode Speed (tok/s)")
print("=" * 90)

header = f"{'Context':<10}"
for cfg in CONFIGS:
    header += f" {cfg['name']:>15}"
print(header)
print("-" * 90)

for ctx in CONTEXTS:
    row = f"{ctx:<10}"
    for cfg in CONFIGS:
        val = RESULTS.get(cfg["name"], {}).get(ctx, "-")
        if isinstance(val, (int, float)):
            row += f" {val:>12.1f}   "
        else:
            row += f" {str(val):>12}   "
    print(row)

print("-" * 90)
print()
print("Legend:")
print("  TP2 = 2 GPUs (tensor parallel)")
print("  TP1 = 1 GPU")
print("  INT8 = INT8 KV cache with FP8-V emulation")
print("  FP16 = FP16 KV cache (default)")
print("  OOM = Out of memory")
print("  SKIP = Context exceeds max_model_len for config")

# Save
with open("/home/yeb/Developer/qwen3.5/benchmark_grid_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\nResults saved to benchmark_grid_results.json")

kill_server()
print("\nBenchmark complete.")
