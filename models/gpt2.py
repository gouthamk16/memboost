"""
GPT-2 Weight Quantization using Memboost
=========================================

Quantizes GPT-2 linear layer weights to 2/4-bit mixed precision using
memboost, reports compression statistics, and optionally saves the
quantized checkpoint.

Usage:
    python models/gpt2.py --model gpt2 --ratio-4bit 0.1
    python models/gpt2.py --model gpt2-medium --save quantized_gpt2_medium.pt

Dependencies: torch, memboost, transformers (weight download)
"""

import argparse
import time

import torch

import memboost


# ---------------------------------------------------------------------------
# Weight Loading + Quantization
# ---------------------------------------------------------------------------

def _get_linear_keys():
    """Return (hf_suffix, description) pairs for linear layers."""
    return [
        ("attn.c_attn",  "attention QKV"),
        ("attn.c_proj",  "attention output"),
        ("mlp.c_fc",     "MLP up"),
        ("mlp.c_proj",   "MLP down"),
    ]


GPT2_CONFIGS = {
    "gpt2":        dict(n_embd=768,  n_layer=12, n_head=12, n_inner=3072),
    "gpt2-medium": dict(n_embd=1024, n_layer=24, n_head=16, n_inner=4096),
    "gpt2-large":  dict(n_embd=1280, n_layer=36, n_head=20, n_inner=5120),
    "gpt2-xl":     dict(n_embd=1600, n_layer=48, n_head=25, n_inner=6400),
}


def quantize_gpt2(model_name: str = "gpt2", ratio_4bit: float = 0.1):
    """Download GPT-2 weights and quantize all linear layers.

    Returns:
        quantized_layers: dict mapping layer name to QuantizedTensor
        stats: dict with compression statistics
    """
    from transformers import GPT2LMHeadModel

    if model_name not in GPT2_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(GPT2_CONFIGS)}")

    cfg = GPT2_CONFIGS[model_name]
    n_layer = cfg["n_layer"]

    print(f"Loading {model_name} weights (n_embd={cfg['n_embd']}, n_layer={n_layer})...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()
    del hf_model

    quantized_layers = {}
    total_fp16_bytes = 0
    total_quantized_bytes = 0
    total_layers = 0

    for i in range(n_layer):
        prefix = f"transformer.h.{i}"

        for hf_suffix, desc in _get_linear_keys():
            key = f"{prefix}.{hf_suffix}.weight"
            # GPT-2 uses Conv1D layout (transposed), so .T to get [out, in]
            w = hf_sd[key].T.contiguous()

            fp16_bytes = w.numel() * 2
            total_fp16_bytes += fp16_bytes

            # Quantize
            w_fp16 = w.to(dtype=torch.float16, device="cuda").contiguous()
            qt = memboost.quantize(w_fp16, ratio_4bit=ratio_4bit)

            layer_name = f"h.{i}.{hf_suffix}"
            quantized_layers[layer_name] = qt
            total_quantized_bytes += qt.total_bytes
            total_layers += 1

        print(f"  Block {i:2d}: quantized 4 layers", end="\r")

    print(f"\nQuantized {total_layers} linear layers")
    print(f"  FP16 weight memory:       {total_fp16_bytes / 1024**2:.1f} MiB")
    print(f"  Quantized weight memory:  {total_quantized_bytes / 1024**2:.1f} MiB")
    print(f"  Compression ratio:        {total_fp16_bytes / total_quantized_bytes:.2f}x")

    # Compute average bits across all quantized layers
    total_elements = 0
    weighted_bits = 0.0
    for qt in quantized_layers.values():
        n = qt.M * qt.K
        total_elements += n
        weighted_bits += qt.avg_bits * n
    avg_bits = weighted_bits / total_elements if total_elements > 0 else 0
    print(f"  Average bits per weight:  {avg_bits:.2f}")

    stats = {
        "fp16_bytes": total_fp16_bytes,
        "quantized_bytes": total_quantized_bytes,
        "compression_ratio": total_fp16_bytes / total_quantized_bytes,
        "avg_bits": avg_bits,
        "num_layers": total_layers,
    }

    return quantized_layers, stats


def verify_roundtrip(quantized_layers, model_name: str = "gpt2", max_layers: int = 4):
    """Verify quantization quality by dequantizing and measuring error."""
    from transformers import GPT2LMHeadModel

    print(f"\nVerifying roundtrip quality (first {max_layers} layers)...")

    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()
    del hf_model

    cfg = GPT2_CONFIGS[model_name]
    checked = 0

    for i in range(min(max_layers, cfg["n_layer"])):
        for hf_suffix, desc in _get_linear_keys():
            layer_name = f"h.{i}.{hf_suffix}"
            if layer_name not in quantized_layers:
                continue

            key = f"transformer.h.{i}.{hf_suffix}.weight"
            w_orig = hf_sd[key].T.contiguous().to(dtype=torch.float16, device="cuda")

            qt = quantized_layers[layer_name]
            w_deq = memboost.dequantize(qt)

            # Compute relative error
            mse = ((w_orig.float() - w_deq.float()) ** 2).mean().item()
            orig_norm = (w_orig.float() ** 2).mean().item()
            rel_error = (mse / orig_norm) ** 0.5 if orig_norm > 0 else 0

            print(f"  {layer_name:30s}  RMSE={mse**0.5:.6f}  RelErr={rel_error:.4f}  AvgBits={qt.avg_bits:.2f}")
            checked += 1

    print(f"  Verified {checked} layers")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quantize GPT-2 weights with memboost")
    parser.add_argument("--model", type=str, default="gpt2",
                        choices=list(GPT2_CONFIGS.keys()),
                        help="GPT-2 model variant")
    parser.add_argument("--ratio-4bit", type=float, default=0.1,
                        help="Fraction of groups at 4-bit precision")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save quantized checkpoint (.pt)")
    parser.add_argument("--verify", action="store_true",
                        help="Run roundtrip verification after quantization")
    args = parser.parse_args()

    t0 = time.time()
    quantized_layers, stats = quantize_gpt2(args.model, ratio_4bit=args.ratio_4bit)
    elapsed = time.time() - t0
    print(f"Quantization completed in {elapsed:.1f}s")

    if args.verify:
        verify_roundtrip(quantized_layers, model_name=args.model)

    if args.save:
        print(f"\nSaving quantized checkpoint to {args.save}...")
        save_dict = {}
        for name, qt in quantized_layers.items():
            save_dict[name] = qt.state_dict()
        save_dict["_stats"] = stats
        save_dict["_model_name"] = args.model
        save_dict["_ratio_4bit"] = args.ratio_4bit
        torch.save(save_dict, args.save)
        print(f"  Saved ({stats['quantized_bytes'] / 1024**2:.1f} MiB quantized weights)")

    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
