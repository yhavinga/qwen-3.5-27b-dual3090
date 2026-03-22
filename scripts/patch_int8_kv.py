#!/usr/bin/env python3
"""
Patch vLLM 0.18.x to add INT8 KV cache support for Ampere GPUs (RTX 3090).

This enables 50% memory reduction in KV cache, allowing:
- Longer context at same batch size
- Or higher batch sizes at same context

INT8 KV cache uses simple linear INT8 quantization for both K and V values:
- K: scale = absmax(K) / 127
- V: scale = absmax(V) / 127

Usage:
    python scripts/patch_int8_kv.py [--vllm-path /path/to/vllm]

After patching, run with:
    vllm serve MODEL --kv-cache-dtype int8 --kv-cache-memory-bytes <bytes> ...

Note: For hybrid Mamba-Transformer models (e.g., Qwen3.5), use conservative
memory settings as Mamba state cache is separate from KV cache.

Example for Qwen3.5-27B on dual RTX 3090:
    vllm serve Qwen/Qwen3.5-27B-GPTQ-Int4 \\
        --tensor-parallel-size 2 \\
        --kv-cache-dtype int8 \\
        --kv-cache-memory-bytes 3221225472 \\
        --max-model-len 16384 \\
        --max-num-seqs 16
"""

import sys
import os
import re
from pathlib import Path

def get_vllm_path():
    """Get path to vllm installation."""
    try:
        import vllm
        return Path(vllm.__path__[0])
    except ImportError:
        print("ERROR: vllm not installed")
        sys.exit(1)

def backup_file(path: Path):
    """Create backup of file if not already backed up."""
    backup = path.with_suffix(path.suffix + '.orig')
    if not backup.exists():
        import shutil
        shutil.copy2(path, backup)
        print(f"  Backed up: {path.name}")
    return backup

def patch_cache_config(vllm_path: Path):
    """Add 'int8' to CacheDType literal."""
    cache_py = vllm_path / "config" / "cache.py"
    if not cache_py.exists():
        print(f"ERROR: {cache_py} not found")
        return False

    backup_file(cache_py)
    content = cache_py.read_text()

    # Check if already patched
    if '"int8"' in content:
        print("  cache.py: Already patched")
        return True

    # Add int8 to CacheDType - try different patterns
    patterns = [
        ('"fp8_ds_mla",\n]', '"fp8_ds_mla",\n    "int8",  # INT8 KV cache for Ampere GPUs (RTX 3090)\n]'),
        ('"fp8_inc",\n]', '"fp8_inc",\n    "int8",  # INT8 KV cache for Ampere GPUs (RTX 3090)\n]'),
    ]

    for old, new in patterns:
        if old in content:
            content = content.replace(old, new)
            cache_py.write_text(content)
            print("  cache.py: Patched CacheDType")
            return True

    print("  cache.py: Could not find CacheDType pattern")
    return False

def patch_backend(vllm_path: Path):
    """Update is_quantized_kv_cache to include int8."""
    backend_py = vllm_path / "v1" / "attention" / "backend.py"
    if not backend_py.exists():
        print(f"ERROR: {backend_py} not found")
        return False

    backup_file(backend_py)
    content = backend_py.read_text()

    # Check if already patched
    if 'or kv_cache_dtype == "int8"' in content:
        print("  backend.py: Already patched")
        return True

    # Update is_quantized_kv_cache function
    old = 'return kv_cache_dtype.startswith("fp8")'
    new = 'return kv_cache_dtype.startswith("fp8") or kv_cache_dtype == "int8"'

    if old in content:
        content = content.replace(old, new)
        backend_py.write_text(content)
        print("  backend.py: Patched is_quantized_kv_cache")
        return True
    else:
        print("  backend.py: Could not find is_quantized_kv_cache pattern")
        return False

def patch_reshape_and_cache(vllm_path: Path):
    """Add INT8 quantization to triton reshape_and_cache kernel."""
    kernel_py = vllm_path / "v1" / "attention" / "ops" / "triton_reshape_and_cache_flash.py"
    if not kernel_py.exists():
        print(f"ERROR: {kernel_py} not found")
        return False

    backup_file(kernel_py)
    content = kernel_py.read_text()

    # Check if already patched
    if 'torch.int8,  # INT8 KV cache' in content:
        print("  triton_reshape_and_cache_flash.py: Already patched")
        return True

    patches_applied = 0

    # Patch 1: Add int8 to assertion
    old_check = 'assert kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8")'
    new_check = 'assert kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8") or kv_cache_dtype == "int8"'
    if old_check in content:
        content = content.replace(old_check, new_check)
        patches_applied += 1

    # Patch 2: Update FP8_KV_CACHE to include INT8
    old_fp8 = 'FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")'
    new_fp8 = 'FP8_KV_CACHE = kv_cache_dtype.startswith("fp8") or kv_cache_dtype == "int8"'
    if old_fp8 in content:
        content = content.replace(old_fp8, new_fp8)
        patches_applied += 1

    # Patch 3: Update dtype handling - handle int8 like fp8 but with torch.int8
    # Find and replace the kv_cache_torch_dtype assignment
    old_dtype_pattern = r'''kv_cache_torch_dtype = \(\s*current_platform\.fp8_dtype\(\)\s*if kv_cache_dtype\.startswith\("fp8"\)\s*else key_cache\.dtype\s*\)'''
    new_dtype = '''kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if kv_cache_dtype.startswith("fp8")
        else torch.int8 if kv_cache_dtype == "int8"
        else key_cache.dtype
    )'''
    content, count = re.subn(old_dtype_pattern, new_dtype, content)
    patches_applied += count

    # Patch 4: Add int8 to valid dtype assertion
    old_assert = '''assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
        torch.float8_e4m3fnuz,
    ]'''
    new_assert = '''assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
        torch.float8_e4m3fnuz,
        torch.int8,  # INT8 KV cache for Ampere
    ]'''
    if old_assert in content:
        content = content.replace(old_assert, new_assert)
        patches_applied += 1

    kernel_py.write_text(content)
    print(f"  triton_reshape_and_cache_flash.py: Applied {patches_applied} patches")
    return patches_applied > 0

def patch_triton_unified_attention(vllm_path: Path):
    """Add INT8 dequantization to triton unified attention kernel."""
    kernel_py = vllm_path / "v1" / "attention" / "ops" / "triton_unified_attention.py"
    if not kernel_py.exists():
        print(f"ERROR: {kernel_py} not found")
        return False

    backup_file(kernel_py)
    content = kernel_py.read_text()

    # Check if already patched
    if 'K_load.dtype == tl.int8' in content:
        print("  triton_unified_attention.py: Already patched")
        return True

    # Patch: Add INT8 handling alongside FP8 for K and V dequantization
    old_dequant = '''        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load'''

    new_dequant = '''        # Handle FP8 and INT8 quantized KV cache dequantization
        if K_load.dtype.is_fp8() or K_load.dtype == tl.int8:
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        # Handle FP8 and INT8 quantized KV cache dequantization
        if V_load.dtype.is_fp8() or V_load.dtype == tl.int8:
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load'''

    if old_dequant in content:
        content = content.replace(old_dequant, new_dequant)
        kernel_py.write_text(content)
        print("  triton_unified_attention.py: Patched K/V dequantization for INT8")
        return True
    else:
        print("  triton_unified_attention.py: Could not find dequantization pattern (may already be patched)")
        return False

def patch_triton_attn_backend(vllm_path: Path):
    """Add int8 to supported_kv_cache_dtypes and handle INT8 in triton attention backend."""
    triton_attn = vllm_path / "v1" / "attention" / "backends" / "triton_attn.py"
    if not triton_attn.exists():
        print(f"  triton_attn.py: Not found (may not exist in this version)")
        return True

    backup_file(triton_attn)
    content = triton_attn.read_text()

    patches_applied = 0

    # Patch 1: Add int8 to supported dtypes list
    if '"int8"' not in content and 'supported_kv_cache_dtypes' in content:
        content = content.replace(
            '"fp8_e5m2",',
            '"fp8_e5m2",\n        "int8",  # INT8 KV cache for Ampere GPUs'
        )
        patches_applied += 1

    # Patch 2: Handle INT8 alongside FP8 in fused RoPE cache
    old_rope = '''        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )'''

    new_rope = '''        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        is_int8_kv_cache = self.kv_cache_dtype == "int8"
        is_quantized_kv_cache = is_fp8_kv_cache or is_int8_kv_cache

        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        # INT8 cache is already stored as int8, no view conversion needed

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_quantized_kv_cache,
        )'''

    if old_rope in content:
        content = content.replace(old_rope, new_rope)
        patches_applied += 1

    if patches_applied > 0:
        triton_attn.write_text(content)
        print(f"  triton_attn.py: Applied {patches_applied} patches")
    else:
        print("  triton_attn.py: Already patched or patterns not found")

    return True

def patch_flash_attn(vllm_path: Path):
    """Check if flash attention backend exists and needs patching."""
    flash_attn = vllm_path / "v1" / "attention" / "backends" / "flash_attn.py"
    if flash_attn.exists():
        backup_file(flash_attn)
        content = flash_attn.read_text()

        if 'supported_kv_cache_dtypes' in content and '"int8"' not in content:
            if '"fp8"' in content or '"fp8_e4m3"' in content:
                content = content.replace(
                    '"fp8_e5m2",',
                    '"fp8_e5m2",\n        "int8",'
                )
                flash_attn.write_text(content)
                print("  flash_attn.py: Added int8 to supported dtypes")
            return True
        print("  flash_attn.py: Already patched or no supported_kv_cache_dtypes")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Patch vLLM for INT8 KV cache")
    parser.add_argument("--vllm-path", type=Path, help="Path to vLLM installation")
    parser.add_argument("--restore", action="store_true", help="Restore from backups")
    args = parser.parse_args()

    if args.vllm_path:
        vllm_path = args.vllm_path
    else:
        vllm_path = get_vllm_path()

    print(f"vLLM path: {vllm_path}")
    print(f"vLLM version: ", end="")
    version_file = vllm_path / "_version.py"
    if version_file.exists():
        print(version_file.read_text().split('"')[1])
    else:
        print("unknown")

    if args.restore:
        print("\nRestoring from backups...")
        for ext in [".py"]:
            for f in vllm_path.rglob(f"*{ext}.orig"):
                original = f.with_suffix("")
                import shutil
                shutil.copy2(f, original)
                print(f"  Restored: {original.relative_to(vllm_path)}")
        print("Done!")
        return

    print("\nPatching vLLM for INT8 KV cache support...")
    print("=" * 60)

    success = True
    success &= patch_cache_config(vllm_path)
    success &= patch_backend(vllm_path)
    success &= patch_reshape_and_cache(vllm_path)
    success &= patch_triton_unified_attention(vllm_path)
    success &= patch_triton_attn_backend(vllm_path)
    success &= patch_flash_attn(vllm_path)

    print("=" * 60)
    if success:
        print("\nPatch applied successfully!")
        print("\nUsage:")
        print("  vllm serve MODEL --kv-cache-dtype int8 --kv-cache-memory-bytes <bytes> ...")
        print("\nExample for Qwen3.5-27B on dual RTX 3090:")
        print("  vllm serve Qwen/Qwen3.5-27B-GPTQ-Int4 \\")
        print("      --tensor-parallel-size 2 \\")
        print("      --kv-cache-dtype int8 \\")
        print("      --kv-cache-memory-bytes 3221225472 \\")
        print("      --max-model-len 16384")
    else:
        print("\nPatch failed! Check errors above.")
        print("You can restore with: python patch_int8_kv.py --restore")
        sys.exit(1)

if __name__ == "__main__":
    main()
