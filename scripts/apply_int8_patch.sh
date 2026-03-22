#!/bin/bash
# Apply the INT8 KV cache patch to vLLM
#
# This patch enables INT8-K + FP8-V (emulated) KV cache for Ampere GPUs (RTX 3090)
# that lack native FP8 hardware support.
#
# Benefits:
#   - 50% KV cache memory savings
#   - Enables 128K+ context where BF16 would OOM
#   - ~5-10% compute overhead (acceptable trade-off)
#
# Usage:
#   ./scripts/apply_int8_patch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PATCH_FILE="${PROJECT_ROOT}/patches/vllm-int8-kv-cache-with-fp8v.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

# Find vLLM installation path
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])" 2>/dev/null)

if [ -z "$VLLM_PATH" ]; then
    echo "ERROR: vLLM not found. Install with: pip install vllm"
    exit 1
fi

echo "vLLM path: $VLLM_PATH"
echo "Patch file: $PATCH_FILE"
echo ""

# Check if already patched
if grep -q "INT8_KV_CACHE" "$VLLM_PATH/v1/attention/ops/triton_reshape_and_cache_flash.py" 2>/dev/null; then
    echo "vLLM appears to already be patched for INT8 KV cache."
    read -p "Re-apply patch anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Apply patch
echo "Applying patch..."
cd "$(dirname "$VLLM_PATH")"

if patch -p1 --dry-run < "$PATCH_FILE" > /dev/null 2>&1; then
    patch -p1 < "$PATCH_FILE"
    echo ""
    echo "Patch applied successfully!"
    echo ""
    echo "To use INT8 KV cache:"
    echo "  export VLLM_INT8_V_FP8_EMUL=1"
    echo "  vllm serve ... --kv-cache-dtype int8"
    echo ""
    echo "Or use: ./scripts/launch-server-int8kv.sh"
else
    echo "ERROR: Patch failed to apply. vLLM version may be incompatible."
    echo "The patch was designed for vLLM 0.17.x"
    echo ""
    echo "Try manual inspection:"
    echo "  patch -p1 --dry-run -d \$(dirname $VLLM_PATH) < $PATCH_FILE"
    exit 1
fi
