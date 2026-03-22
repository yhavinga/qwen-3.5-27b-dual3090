#!/usr/bin/env python3
"""Test that INT8 KV cache patch is applied correctly."""

import sys

def test_cache_dtype():
    """Test that int8 is in CacheDType."""
    from vllm.config.cache import CacheDType
    from typing import get_args
    dtypes = get_args(CacheDType)
    assert "int8" in dtypes, f"int8 not in CacheDType: {dtypes}"
    print(f"[PASS] CacheDType includes int8: {dtypes}")

def test_quantized_check():
    """Test that is_quantized_kv_cache returns True for int8."""
    from vllm.v1.attention.backend import is_quantized_kv_cache
    assert is_quantized_kv_cache("int8"), "is_quantized_kv_cache('int8') should be True"
    assert is_quantized_kv_cache("fp8"), "is_quantized_kv_cache('fp8') should be True"
    assert not is_quantized_kv_cache("auto"), "is_quantized_kv_cache('auto') should be False"
    print("[PASS] is_quantized_kv_cache handles int8 correctly")

def test_triton_kernel_import():
    """Test that triton kernels can be imported."""
    try:
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
        print("[PASS] triton_reshape_and_cache_flash imported")
    except Exception as e:
        print(f"[FAIL] Could not import triton kernel: {e}")
        return False
    return True

def test_vllm_import():
    """Test that vllm can be imported without errors."""
    try:
        import vllm
        print(f"[PASS] vllm imported (version: {vllm.__version__})")
        return True
    except Exception as e:
        print(f"[FAIL] Could not import vllm: {e}")
        return False

def main():
    print("Testing INT8 KV cache patch...")
    print("=" * 50)

    errors = 0

    if not test_vllm_import():
        errors += 1

    try:
        test_cache_dtype()
    except Exception as e:
        print(f"[FAIL] CacheDType test: {e}")
        errors += 1

    try:
        test_quantized_check()
    except Exception as e:
        print(f"[FAIL] is_quantized_kv_cache test: {e}")
        errors += 1

    if not test_triton_kernel_import():
        errors += 1

    print("=" * 50)
    if errors == 0:
        print("All tests passed! INT8 KV cache is ready.")
        print("\nUsage:")
        print("  vllm serve MODEL --kv-cache-dtype int8 ...")
        return 0
    else:
        print(f"FAILED: {errors} errors found")
        return 1

if __name__ == "__main__":
    sys.exit(main())
