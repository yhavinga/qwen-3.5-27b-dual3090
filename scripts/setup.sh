#!/bin/bash
# Setup script for Qwen 3.5 27B inference environment
#
# Usage:
#   ./scripts/setup.sh
#
# This will:
# 1. Create/activate venv
# 2. Install vLLM and dependencies
# 3. Download the model (optional)
# 4. Verify NVLink connectivity

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Qwen 3.5 27B Setup"
echo "=========================================="
echo ""

# ============================================================================
# Step 1: Python environment
# ============================================================================
echo "[1/4] Setting up Python environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "      Created new venv"
fi

source venv/bin/activate
echo "      Activated venv: $(which python)"

# ============================================================================
# Step 2: Install dependencies
# ============================================================================
echo ""
echo "[2/4] Installing dependencies..."

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "      vLLM version: $(pip show vllm | grep Version)"

# ============================================================================
# Step 3: Verify GPU setup
# ============================================================================
echo ""
echo "[3/4] Verifying GPU setup..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "      WARNING: nvidia-smi not found. CUDA may not be installed."
else
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
    echo "      Checking NVLink..."
    nvidia-smi topo -m | grep -E "GPU[0-9]|NV|PHB" | head -10
fi

# ============================================================================
# Step 4: Model download (optional)
# ============================================================================
echo ""
echo "[4/4] Model download..."
echo ""
echo "      Models will be downloaded automatically on first run."
echo "      To pre-download, run:"
echo ""
echo "      # Dense 27B (17GB)"
echo "      huggingface-cli download Qwen/Qwen3.5-27B-GPTQ-Int4"
echo ""
echo "      # MoE 35B (faster, 19GB)"
echo "      huggingface-cli download Qwen/Qwen3.5-35B-A3B-FP8"
echo ""

# ============================================================================
# Done
# ============================================================================
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "  # Start the server"
echo "  ./scripts/launch-server.sh"
echo ""
echo "  # Test in another terminal"
echo "  source venv/bin/activate"
echo "  python scripts/quick_test.py"
echo ""
echo "  # For maximum speed (MoE variant)"
echo "  ./scripts/launch-server-35b-moe.sh"
echo ""
