#!/bin/bash
# Check NVLink connectivity between GPUs
#
# Expected output for dual 3090 with NVLink:
#   GPU0 <-> GPU1: NV4 (NVLink 4 lanes)
#
# Without NVLink you'll see:
#   GPU0 <-> GPU1: PHB (PCIe Host Bridge)

echo "=== GPU Topology ==="
nvidia-smi topo -m

echo ""
echo "=== NVLink Status ==="
nvidia-smi nvlink --status 2>/dev/null || echo "NVLink query not available"

echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

echo ""
echo "=== P2P Bandwidth Test ==="
echo "If cuda-samples is installed, run: /usr/local/cuda/samples/1_Utilities/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest"
