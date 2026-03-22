#!/usr/bin/env python3
"""Test one configuration at multiple context lengths."""
import json
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"
API = "http://localhost:8000/v1/completions"
CORPUS = Path("/home/yeb/Developer/gemma/data/dutch_parliament_text.txt").read_text()

def measure(ctx_tokens, gen_tokens=64):
    chars = min(ctx_tokens * 4, len(CORPUS) - 100)
    prompt = CORPUS[:chars] + "\n\nContinue:"
    
    times, gens, ctxs = [], [], []
    for gen in [gen_tokens // 2, gen_tokens]:
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

# Wait for server
print("Waiting for server...", end=" ", flush=True)
for _ in range(30):
    try:
        req = urllib.request.Request("http://localhost:8000/v1/models")
        with urllib.request.urlopen(req, timeout=5) as r:
            if json.loads(r.read()).get("data"):
                print("OK")
                break
    except:
        time.sleep(2)
else:
    print("TIMEOUT")
    sys.exit(1)

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

# Test contexts
results = {}
for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    print(f"  {ctx:>6} tokens: ", end="", flush=True)
    try:
        tps, actual = measure(ctx)
        print(f"{tps:>5.1f} tok/s (ctx={actual})")
        results[ctx] = round(tps, 1)
    except Exception as e:
        err = str(e).lower()
        if "memory" in err or "oom" in err:
            print("OOM")
            results[ctx] = "OOM"
            break
        else:
            print(f"ERROR: {str(e)[:50]}")
            results[ctx] = "ERR"

print(json.dumps(results))
