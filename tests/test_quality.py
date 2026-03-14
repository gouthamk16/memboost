"""Diagnose GPT-2 quantization quality."""
import torch
import memboost
from transformers import GPT2LMHeadModel

# Load HF model
print("Loading GPT-2...")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = hf_model.state_dict()

# Test a few linear weights
test_keys = [
    "transformer.h.0.attn.c_attn.weight",
    "transformer.h.0.mlp.c_fc.weight",
    "transformer.h.6.attn.c_attn.weight",
]

for key in test_keys:
    w = sd[key].T.contiguous()  # Conv1D -> Linear
    w_fp16 = w.to(torch.float16).to("cuda")

    print(f"\n{key}")
    print(f"  Shape: {w_fp16.shape}, range: [{w_fp16.min().item():.4f}, {w_fp16.max().item():.4f}]")

    # Pure 2-bit quantization
    qt = memboost.quantize(w_fp16, ratio_4bit=0.0)
    w_deq = memboost.dequantize(qt)

    err = (w_fp16.float() - w_deq.float()).abs()
    print(f"  Pure 2-bit: mean_err={err.mean().item():.6f}, max_err={err.max().item():.4f}, "
          f"rel_err={(err / (w_fp16.float().abs() + 1e-6)).mean().item():.4f}")

    # Verify matmul quality via dequantize path
    x = torch.randn(4, w_fp16.shape[1], dtype=torch.float16, device="cuda")
    y_ref = x @ w_fp16.T
    y_deq = x @ w_deq.T

    print(f"  y_ref vs y_deq:  max_diff={((y_ref - y_deq).float().abs().max().item()):.4f}")

    # Mixed precision (10% 4-bit)
    qt_mix = memboost.quantize(w_fp16, ratio_4bit=0.1)
    w_deq_mix = memboost.dequantize(qt_mix)
    err_mix = (w_fp16.float() - w_deq_mix.float()).abs()
    print(f"  Mixed 2/4-bit: mean_err={err_mix.mean().item():.6f}, avg_bits={qt_mix.avg_bits:.2f}")
    print(f"  Compression: {w_fp16.numel() * 2 / qt.total_bytes:.2f}x (2-bit), "
          f"{w_fp16.numel() * 2 / qt_mix.total_bytes:.2f}x (mixed)")

print("\nDiagnostic complete.")
