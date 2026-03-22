#!/usr/bin/env python3
"""
Patch vLLM to disable AOT compile by default on PyTorch 2.10-2.11.

The FakeTensorMode AttributeError occurs when:
- PyTorch 2.10+ is used
- VLLM_USE_AOT_COMPILE is not explicitly set
- vLLM tries to use standalone_compile which has API incompatibilities

This patch changes the default to only enable AOT compile on PyTorch 2.12+
where the API is more stable.

Usage:
    python scripts/patch_faketensor_fix.py [--vllm-path /path/to/vllm]
"""

import sys
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
    backup = path.with_suffix(path.suffix + '.faketensor.orig')
    if not backup.exists():
        import shutil
        shutil.copy2(path, backup)
        print(f"  Backed up: {path.name}")
    return backup

def patch_envs_py(vllm_path: Path):
    """Patch envs.py to require PyTorch 2.12+ for AOT compile by default."""
    envs_py = vllm_path / "envs.py"
    if not envs_py.exists():
        print(f"ERROR: {envs_py} not found")
        return False

    backup_file(envs_py)
    content = envs_py.read_text()

    # Check if already patched
    if '2.12.0' in content and 'use_aot_compile' in content and 'PATCHED' in content:
        print("  envs.py: Already patched")
        return True

    # Find and patch the use_aot_compile function
    # Change from: is_torch_equal_or_newer("2.10.0")
    # To: is_torch_equal_or_newer("2.12.0")  # PATCHED: FakeTensorMode fix

    old_pattern = r'is_torch_equal_or_newer\("2\.10\.0"\)'
    new_pattern = 'is_torch_equal_or_newer("2.12.0")  # PATCHED: FakeTensorMode fix for PyTorch 2.10-2.11'

    if re.search(old_pattern, content):
        # Only replace the first occurrence (in use_aot_compile)
        content = re.sub(old_pattern, new_pattern, content, count=1)
        envs_py.write_text(content)
        print("  envs.py: Patched use_aot_compile to require PyTorch 2.12+")
        return True
    else:
        print("  envs.py: Pattern not found (may already be patched or different version)")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Patch vLLM for FakeTensorMode fix")
    parser.add_argument("--vllm-path", type=Path, help="Path to vLLM installation")
    parser.add_argument("--restore", action="store_true", help="Restore from backups")
    args = parser.parse_args()

    if args.vllm_path:
        vllm_path = args.vllm_path
    else:
        vllm_path = get_vllm_path()

    print(f"vLLM path: {vllm_path}")

    if args.restore:
        print("\nRestoring from backups...")
        for f in vllm_path.rglob("*.faketensor.orig"):
            original = Path(str(f).replace('.faketensor.orig', ''))
            import shutil
            shutil.copy2(f, original)
            print(f"  Restored: {original.name}")
        print("Done!")
        return

    print("\nPatching vLLM for FakeTensorMode fix...")
    print("=" * 50)

    success = patch_envs_py(vllm_path)

    print("=" * 50)
    if success:
        print("\nPatch applied successfully!")
        print("\nThis disables AOT compile by default on PyTorch 2.10-2.11")
        print("to work around the FakeTensorMode AttributeError.")
        print("\nNote: If you upgrade to PyTorch 2.12+, AOT compile will")
        print("be automatically re-enabled for better performance.")
    else:
        print("\nPatch may not be needed or failed.")
        print("Alternative: set VLLM_USE_AOT_COMPILE=0 environment variable")
        sys.exit(1)

if __name__ == "__main__":
    main()
