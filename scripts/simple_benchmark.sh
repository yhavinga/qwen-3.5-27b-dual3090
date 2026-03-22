#!/bin/bash
# Simple benchmark script - one config at a time

cd /home/yeb/Developer/qwen3.5
source venv/bin/activate

CORPUS="/home/yeb/Developer/gemma/data/dutch_parliament_text.txt"
MODEL="Qwen/Qwen3.5-27B-GPTQ-Int4"

measure_speed() {
    local ctx_chars=$1
    local gen_tok=${2:-64}
    
    # Create prompt
    local prompt=$(head -c $ctx_chars "$CORPUS")
    prompt="$prompt\n\nContinue: 1, 2, 3,"
    
    # Make request and measure
    python3 -c "
import json, time, urllib.request

prompt = '''$prompt'''
start = time.perf_counter()
req = urllib.request.Request(
    'http://localhost:8000/v1/completions',
    data=json.dumps({
        'model': '$MODEL',
        'prompt': prompt,
        'max_tokens': $gen_tok,
        'temperature': 0
    }).encode(),
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        r = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    ctx = r['usage']['prompt_tokens']
    gen = r['usage']['completion_tokens']
    tps = gen / elapsed
    print(f'{ctx},{gen},{elapsed:.2f},{tps:.1f}')
except Exception as e:
    print(f'ERROR,{e}')
" 2>/dev/null
}

wait_for_server() {
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "data"; then
            return 0
        fi
        sleep 3
    done
    return 1
}

start_server() {
    local tp=$1
    local kv_dtype=$2
    local gpus=$3
    local max_len=$4
    local use_scales=$5
    
    pkill -9 -f "vllm serve" 2>/dev/null
    sleep 5
    
    local env_cmd="CUDA_VISIBLE_DEVICES=$gpus"
    if [ "$use_scales" = "1" ]; then
        env_cmd="$env_cmd VLLM_KV_SCALES_FILE=/home/yeb/Developer/qwen3.5/scales/qwen35_27b_per_layer.json VLLM_INT8_V_FP8_EMUL=1"
    fi
    
    eval "$env_cmd nohup vllm serve $MODEL \
        --tensor-parallel-size $tp \
        --kv-cache-dtype $kv_dtype \
        --max-model-len $max_len \
        --max-num-seqs 4 \
        --gpu-memory-utilization 0.95 \
        --port 8000 \
        > /tmp/vllm_bench.log 2>&1 &"
    
    if wait_for_server; then
        return 0
    else
        return 1
    fi
}

echo "======================================================================"
echo "BENCHMARK GRID: Qwen3.5-27B-GPTQ-Int4 Single User Decode Speed"
echo "======================================================================"
echo ""

# Test configurations
# Format: name|tp|kv_dtype|gpus|use_scales|max_contexts
configs=(
    "TP2_INT8|2|int8|0,1|1|65536"
    "TP2_FP16|2|auto|0,1|0|32768"
    "TP1_INT8|1|int8|0|1|32768"
    "TP1_FP16|1|auto|0|0|16384"
)

# Context sizes (in approximate chars, ~4 chars per token)
contexts=(2048 4096 8192 16384 32768 65536 131072 262144)

for cfg in "${configs[@]}"; do
    IFS='|' read -r name tp kv gpus scales max_ctx <<< "$cfg"
    
    echo ""
    echo "======================================================================"
    echo "Configuration: $name (TP=$tp, KV=$kv)"
    echo "======================================================================"
    
    for ctx_chars in "${contexts[@]}"; do
        # Skip if beyond max
        ctx_tok=$((ctx_chars / 4))
        if [ $ctx_tok -gt $max_ctx ]; then
            echo "  ${ctx_tok} tokens: SKIP (beyond max $max_ctx)"
            continue
        fi
        
        echo -n "  Testing ~${ctx_tok} tokens... "
        
        # Start server
        if ! start_server $tp $kv $gpus $((ctx_tok + 1024)) $scales; then
            # Check for OOM
            if grep -q "OutOfMemory\|CUDA out of memory" /tmp/vllm_bench.log 2>/dev/null; then
                echo "OOM"
                echo "  Remaining contexts: OOM (skipped)"
                break
            else
                echo "FAILED (server start)"
                continue
            fi
        fi
        
        # Measure
        result=$(measure_speed $ctx_chars 64)
        if [[ "$result" == ERROR* ]]; then
            if [[ "$result" == *"memory"* ]]; then
                echo "OOM"
                break
            else
                echo "ERROR: $result"
            fi
        else
            IFS=',' read -r actual_ctx gen elapsed tps <<< "$result"
            echo "OK - ${actual_ctx} ctx, ${tps} tok/s decode"
        fi
    done
done

echo ""
echo "======================================================================"
echo "Benchmark complete"
echo "======================================================================"

pkill -9 -f "vllm serve" 2>/dev/null
