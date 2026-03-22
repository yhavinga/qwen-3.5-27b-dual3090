#!/usr/bin/env python3
"""
Profile K/V activation ranges per layer for Qwen3.5-27B.

Unlike pure transformers, Qwen3.5 is a hybrid Mamba-Transformer where
~50% of layers are Mamba (no KV cache) and ~50% are Attention (use KV cache).

This script profiles the attention layers to determine if the 340x V variance
seen in Gemma 3 27B also applies to Qwen3.5.

Usage:
    python scripts/profile_v_scales.py --text-file data/calibration_text.txt

Output:
    - CSV file with per-layer K/V absmax and scales
    - Summary statistics
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def profile_via_vllm_api(
    base_url: str,
    model: str,
    prompts: List[str],
) -> Dict[int, Dict[str, float]]:
    """
    Profile by running inference and extracting scales from server.

    NOTE: This requires the vLLM server to expose internal scales via metrics or API.
    For now, this is a placeholder - we'd need to patch vLLM to export this.
    """
    import urllib.request

    print("NOTE: vLLM doesn't expose per-layer scales by default.")
    print("Use --method hooks for actual profiling.")
    return {}


def profile_via_hooks(
    model_name: str,
    prompts: List[str],
    device: str = "cuda",
    max_length: int = 512,
) -> Dict[int, Dict[str, float]]:
    """
    Profile by loading model with HuggingFace and using forward hooks.

    This captures K/V values at each attention layer before caching.
    Works for Qwen3.5 which has both Mamba and Attention layers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    print("This may take a while for large models...")

    # Load with reduced precision for memory efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Multi-GPU support
        trust_remote_code=True,
    )
    model.eval()

    # Storage for K/V statistics per layer
    layer_stats: Dict[int, Dict[str, List[float]]] = {}

    def make_hook(layer_idx: int):
        """Create a forward hook that captures K/V activation ranges."""
        def hook(module, args, kwargs, output):
            # Try to extract K/V from different attention module patterns
            # Qwen3.5 attention modules may have different output formats

            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {"k_absmax": [], "v_absmax": []}

            # For most attention implementations, we need to look at module internals
            # This is a simplified version that captures output stats
            if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                # If we have direct access to projections, we can hook them separately
                pass

            # Capture from output if it's a tuple (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 3:
                past_kv = output[2]
                if past_kv is not None and len(past_kv) == 2:
                    k, v = past_kv
                    k_abs = k.abs().max().item()
                    v_abs = v.abs().max().item()
                    layer_stats[layer_idx]["k_absmax"].append(k_abs)
                    layer_stats[layer_idx]["v_absmax"].append(v_abs)

            return output
        return hook

    # Register hooks on attention layers
    hooks = []
    attention_layer_idx = 0

    for name, module in model.named_modules():
        # Match attention modules (varies by model architecture)
        if any(pattern in name.lower() for pattern in ["self_attn", "attention"]):
            if hasattr(module, "forward"):
                hook = module.register_forward_hook(make_hook(attention_layer_idx), with_kwargs=True)
                hooks.append(hook)
                print(f"  Registered hook on layer {attention_layer_idx}: {name}")
                attention_layer_idx += 1

    print(f"Registered {len(hooks)} attention hooks")

    # Run inference on prompts
    print(f"Running {len(prompts)} calibration prompts...")
    with torch.no_grad():
        for i, prompt in enumerate(prompts, 1):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            except Exception as e:
                print(f"  [{i}] Error: {e}")
                continue

            print(f"  [{i}/{len(prompts)}] Processed {len(inputs['input_ids'][0])} tokens")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Aggregate statistics
    results = {}
    for layer_idx, stats in layer_stats.items():
        if stats["k_absmax"] and stats["v_absmax"]:
            k_max = max(stats["k_absmax"])
            v_max = max(stats["v_absmax"])
            results[layer_idx] = {
                "k_absmax": k_max,
                "v_absmax": v_max,
                "k_scale": k_max / 127.0,
                "v_scale": v_max / 127.0,
            }

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def profile_via_simple_forward(
    model_name: str,
    text: str,
    device: str = "cuda",
) -> Dict[int, Dict[str, float]]:
    """
    Simplified profiling: run a single forward pass and capture KV cache.

    This is more reliable than hooks for complex architectures.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Try loading with quantization for memory efficiency
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"4-bit loading failed: {e}")
        print("Trying BF16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()

    # Tokenize
    inputs = tokenizer(text[:4096], return_tensors="pt", truncation=True)  # Limit to 4K
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")

    # Run forward with KV cache
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)

    # Extract KV cache if available
    results = {}
    past_key_values = outputs.get("past_key_values", None)

    if past_key_values is not None:
        print(f"Found {len(past_key_values)} cache entries")
        for layer_idx, kv in enumerate(past_key_values):
            if kv is None:
                continue
            if isinstance(kv, tuple) and len(kv) >= 2:
                k, v = kv[0], kv[1]
                k_abs = k.float().abs().max().item()
                v_abs = v.float().abs().max().item()
                results[layer_idx] = {
                    "k_absmax": k_abs,
                    "v_absmax": v_abs,
                    "k_scale": k_abs / 127.0,
                    "v_scale": v_abs / 127.0,
                }
    else:
        print("No past_key_values found - model may use different caching")

    # Cleanup
    del model, outputs
    torch.cuda.empty_cache()

    return results


def load_calibration_text(path: Path) -> str:
    """Load calibration text from file."""
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        # Default calibration text
        return """
The European Union's annual budget process involves complex negotiations between
member states, the European Parliament, and the European Commission. The budget
framework is established through multi-annual financial frameworks (MFFs) that
span seven-year periods. Revenue comes primarily from customs duties, VAT-based
contributions, and GNI-based contributions from member states.

In the field of artificial intelligence, transformer architectures have revolutionized
natural language processing. The attention mechanism allows models to weigh the
importance of different parts of the input sequence. Key-value caching enables
efficient autoregressive generation by storing computed states.

Python programming has become the dominant language for machine learning and data
science. Libraries like PyTorch and TensorFlow provide automatic differentiation
and GPU acceleration. The ecosystem includes tools for data manipulation (pandas),
visualization (matplotlib), and scientific computing (numpy).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="Model to profile")
    parser.add_argument("--text-file", type=Path, default=Path("data/dutch_parliament_text.txt"))
    parser.add_argument("--output", "-o", type=Path, default=Path("scales/qwen35_27b_per_layer_scales.csv"))
    parser.add_argument("--method", choices=["hooks", "forward", "api"], default="forward")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load calibration text
    text = load_calibration_text(args.text_file)
    print(f"Calibration text: {len(text)} characters")

    # Profile
    start = time.time()
    if args.method == "hooks":
        prompts = [text[:4000], text[4000:8000], text[8000:12000]]
        prompts = [p for p in prompts if p.strip()]
        results = profile_via_hooks(args.model, prompts, args.device)
    elif args.method == "forward":
        results = profile_via_simple_forward(args.model, text, args.device)
    else:
        results = profile_via_vllm_api("http://localhost:8000", args.model, [text[:4000]])

    elapsed = time.time() - start
    print(f"Profiling took {elapsed:.1f}s")

    if not results:
        print("No results captured. Check if model has KV cache or try different method.")
        return

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer_idx', 'k_absmax', 'v_absmax', 'k_scale', 'v_scale'])
        writer.writeheader()
        for layer_idx in sorted(results.keys()):
            row = {'layer_idx': layer_idx, **results[layer_idx]}
            writer.writerow(row)

    print(f"\nSaved to: {args.output}")

    # Summary statistics
    k_absmax_list = [r['k_absmax'] for r in results.values()]
    v_absmax_list = [r['v_absmax'] for r in results.values()]

    print("\n" + "=" * 60)
    print("SUMMARY: Per-Layer K/V Activation Ranges")
    print("=" * 60)
    print(f"Layers profiled: {len(results)}")
    print()
    print(f"K absmax:")
    print(f"  Min: {min(k_absmax_list):.2f}")
    print(f"  Max: {max(k_absmax_list):.2f}")
    print(f"  Range: {max(k_absmax_list) / max(min(k_absmax_list), 1e-10):.1f}x")
    print()
    print(f"V absmax:")
    print(f"  Min: {min(v_absmax_list):.2f}")
    print(f"  Max: {max(v_absmax_list):.2f}")
    print(f"  Range: {max(v_absmax_list) / max(min(v_absmax_list), 1e-10):.1f}x")
    print()

    # Recommendation
    v_range = max(v_absmax_list) / max(min(v_absmax_list), 1e-10)
    if v_range > 100:
        print("RECOMMENDATION: V variance is HIGH (>100x)")
        print("  -> Use FP8-V emulation with per-layer scales")
    elif v_range > 10:
        print("RECOMMENDATION: V variance is MODERATE (10-100x)")
        print("  -> FP8-V emulation may help, test quality first")
    else:
        print("RECOMMENDATION: V variance is LOW (<10x)")
        print("  -> Simple INT8 should work fine")


if __name__ == "__main__":
    main()
