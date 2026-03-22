#!/usr/bin/env python3
"""
Efficient benchmark - one server start per config, test all contexts.
"""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"
API = "http://localhost:8000/v1/completions"
CORPUS = Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt").read_text()

# Results storage
RESULTS = {}

def kill_server():
    subprocess.run("pkill -9 -f 'vllm serve'", shell=True, capture_output=True)
    time.sleep(5)

def start_server(tp, kv_dtype, gpus, max_len, use_scales):
    kill_server()
    
    env = f"CUDA_VISIBLE_DEVICES={gpus}"
    if use_scales:
        env += " VLLM_KV_SCALES_FILE=/home/yeb/Developer/qwen3.5/scales/qwen35_27b_per_layer.json VLLM_INT8_V_FP8_EMUL=1"
    
    cmd = f"""
source venv/bin/activate && \
{env} nohup vllm serve {MODEL} \
    --tensor-parallel-size {tp} \
    --kv-cache-dtype {kv_dtype} \
    --max-model-len {max_len} \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    > /tmp/vllm_bench.log 2>&1 &
"""
    subprocess.run(cmd, shell=True)
    
    # Wait for server
    for _ in range(60):
        time.sleep(3)
        try:
            req = urllib.request.Request("http://localhost:8000/v1/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                if json.loads(r.read()).get("data"):
                    return True
        except:
            pass
        
        # Check OOM
        try:
            log = Path("/tmp/vllm_bench.log").read_text()
            if "OutOfMemory" in log or "CUDA out of memory" in log:
                return False
        except:
            pass
    
    return False

def measure(ctx_tokens, gen_tokens=64):
    """Measure decode speed. Returns (decode_tps, actual_ctx) or raises."""
    chars = ctx_tokens * 4
    if chars > len(CORPUS):
        chars = len(CORPUS) - 100
    
    prompt = CORPUS[:chars] + "\n\nContinue:"
    
    # Two measurements to isolate decode
    times, gens, ctxs = [], [], []
    for gen in [gen_tokens // 2, gen_tokens]:
        start = time.perf_counter()
        req = urllib.request.Request(
            API,
            data=json.dumps({
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": gen,
                "temperature": 0
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read())
        times.append(time.perf_counter() - start)
        gens.append(result["usage"]["completion_tokens"])
        ctxs.append(result["usage"]["prompt_tokens"])
    
    # Decode = delta_tokens / delta_time
    if times[1] > times[0] and gens[1] > gens[0]:
        decode_tps = (gens[1] - gens[0]) / (times[1] - times[0])
    else:
        decode_tps = gens[1] / times[1]
    
    return decode_tps, ctxs[0]

# Configurations
CONFIGS = [
    ("TP2_INT8", 2, "int8", "0,1", True, 65536),
    ("TP2_FP16", 2, "auto", "0,1", False, 65536),
    ("TP1_INT8", 1, "int8", "0", True, 32768),
    ("TP1_FP16", 1, "auto", "0", False, 32768),
]

CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

print("=" * 80)
print("BENCHMARK GRID: Qwen3.5-27B-GPTQ-Int4")
print("=" * 80)
print()

for name, tp, kv, gpus, scales, max_ctx in CONFIGS:
    print(f"\n{'='*80}")
    print(f"Configuration: {name} (TP={tp}, KV={kv})")
    print("=" * 80)
    
    RESULTS[name] = {}
    
    # Start server with max context
    print(f"  Starting server with max_len={max_ctx}...", end=" ", flush=True)
    if not start_server(tp, kv, gpus, max_ctx, scales):
        print("OOM - cannot start")
        for ctx in CONTEXTS:
            RESULTS[name][ctx] = "OOM"
        continue
    print("OK")
    
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
    for ctx in CONTEXTS:
        if ctx > max_ctx:
            print(f"  {ctx:>6} tokens: SKIP (> max_len)")
            RESULTS[name][ctx] = "SKIP"
            continue
        
        print(f"  {ctx:>6} tokens: ", end="", flush=True)
        try:
            tps, actual = measure(ctx)
            print(f"{tps:>5.1f} tok/s (actual ctx={actual})")
            RESULTS[name][ctx] = round(tps, 1)
        except Exception as e:
            err = str(e)
            if "memory" in err.lower():
                print("OOM")
                RESULTS[name][ctx] = "OOM"
                # Mark remaining as OOM
                idx = CONTEXTS.index(ctx)
                for remaining in CONTEXTS[idx+1:]:
                    RESULTS[name][remaining] = "OOM"
                break
            else:
                print(f"ERROR: {err[:40]}")
                RESULTS[name][ctx] = "ERR"

# Summary grid
print("\n")
print("=" * 100)
print("SUMMARY GRID - Single User Decode Speed (tok/s)")
print("=" * 100)

header = f"{'Context':<10}"
for name, *_ in CONFIGS:
    header += f" {name:>15}"
print(header)
print("-" * 100)

for ctx in CONTEXTS:
    row = f"{ctx:<10}"
    for name, *_ in CONFIGS:
        val = RESULTS.get(name, {}).get(ctx, "-")
        if isinstance(val, (int, float)):
            row += f" {val:>12.1f}   "
        else:
            row += f" {str(val):>12}   "
    print(row)

print("-" * 100)

# Save
with open("benchmark_grid_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\nResults saved to benchmark_grid_results.json")

kill_server()
