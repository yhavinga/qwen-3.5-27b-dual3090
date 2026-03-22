#!/bin/bash
# Build vLLM from source with INT8 KV cache support and FakeTensorMode fix
#
# This script:
# 1. Builds vLLM 0.18.1rc0 from source (with precompiled wheels for speed)
# 2. Applies FakeTensorMode fix (for PyTorch 2.10-2.11 compatibility)
# 3. Applies INT8 KV cache patch (50% memory reduction)
#
# Usage:
#   ./scripts/build_vllm_from_source.sh [--full-build]
#
# Options:
#   --full-build  Compile CUDA kernels from source (takes 20+ minutes)
#                 Default uses precompiled wheels (fast)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_SOURCE="${PROJECT_ROOT}/vllm-source"
FULL_BUILD=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --full-build)
            FULL_BUILD=true
            shift
            ;;
    esac
done

echo "==================================================="
echo "Building vLLM from source"
echo "  - FakeTensorMode fix for PyTorch 2.10-2.11"
echo "  - INT8 KV cache for Ampere GPUs"
echo "==================================================="

# Activate venv
if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "Activated venv"
else
    echo "ERROR: venv not found at ${PROJECT_ROOT}/venv"
    exit 1
fi

# Check PyTorch version
echo ""
echo "Checking PyTorch version..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "  PyTorch: $TORCH_VERSION"

# Uninstall existing vLLM
echo ""
echo "Uninstalling existing vLLM..."
pip uninstall -y vllm 2>/dev/null || true

# Apply FakeTensorMode fix to source BEFORE building
echo ""
echo "Applying FakeTensorMode fix to vLLM source..."
cd "$VLLM_SOURCE"

# Patch envs.py in source to fix FakeTensorMode for PyTorch 2.10-2.11
ENVS_PY="vllm/envs.py"
if grep -q 'is_torch_equal_or_newer("2.10.0")' "$ENVS_PY" 2>/dev/null; then
    sed -i 's/is_torch_equal_or_newer("2\.10\.0")/is_torch_equal_or_newer("2.12.0")  # PATCHED: FakeTensorMode fix/' "$ENVS_PY"
    echo "  Patched $ENVS_PY: AOT compile requires PyTorch 2.12+ now"
else
    echo "  $ENVS_PY: Already patched or different version"
fi

# Build from source
echo ""
echo "Building vLLM from source..."
if [ "$FULL_BUILD" = true ]; then
    echo "  Mode: Full build (compiling CUDA kernels, 20+ minutes)"
    export MAX_JOBS=8
else
    echo "  Mode: Precompiled wheels (fast)"
    export VLLM_USE_PRECOMPILED=1
fi

pip install -e . 2>&1 | tail -50

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Apply INT8 KV cache patch to installed vLLM
echo ""
echo "Applying INT8 KV cache patch..."
python "${PROJECT_ROOT}/scripts/patch_int8_kv.py"

# Apply FakeTensorMode fix to installed vLLM (in case it wasn't in source)
echo ""
echo "Applying FakeTensorMode fix to installed vLLM..."
python "${PROJECT_ROOT}/scripts/patch_faketensor_fix.py" 2>/dev/null || echo "  (Skipped - may already be fixed)"

# Test torch.compile compatibility
echo ""
echo "Testing torch.compile compatibility..."
python << 'EOF'
import os
import sys

# Test if standalone_compile works without FakeTensorMode error
try:
    import torch._inductor
    if hasattr(torch._inductor, "standalone_compile"):
        print("  standalone_compile: available")
        # Try to access FakeTensorMode
        from torch._subclasses import FakeTensorMode
        print("  FakeTensorMode import: OK")
    else:
        print("  standalone_compile: NOT available (pre-2.10 PyTorch)")

    # Check vLLM AOT compile setting
    import vllm.envs as envs
    print(f"  VLLM_USE_AOT_COMPILE: {envs.VLLM_USE_AOT_COMPILE}")

    # Check if AOT compile would be enabled
    print(f"  use_aot_compile(): {envs.use_aot_compile()}")

except Exception as e:
    print(f"  Error: {e}")
    print("  Recommendation: set VLLM_USE_AOT_COMPILE=0")
    sys.exit(1)

print("torch.compile compatibility: OK")
EOF

# Test INT8 KV cache
echo ""
echo "Testing INT8 KV cache..."
python << 'EOF'
try:
    from vllm.config.cache import CacheConfig
    config = CacheConfig(cache_dtype='int8')
    print(f"  INT8 CacheConfig: OK")
except Exception as e:
    print(f"  INT8 CacheConfig: FAILED - {e}")
    exit(1)
EOF

echo ""
echo "==================================================="
echo "Build complete!"
echo ""
echo "Quick test:"
echo "  ./scripts/launch-server.sh"
echo ""
echo "With INT8 KV cache (50% memory savings, 128K context):"
echo "  ./scripts/launch-server-int8kv.sh"
echo ""
echo "If you see FakeTensorMode errors, set:"
echo "  export VLLM_USE_AOT_COMPILE=0"
echo "==================================================="
