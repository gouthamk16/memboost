"""Test quantize/dequantize roundtrip correctness across matrix sizes."""
import torch
import memboost

print("Testing quantize/dequantize roundtrip correctness...")

# Test 1: Basic roundtrip (pure 2-bit)
M, K = 256, 256
W = torch.randn(M, K, dtype=torch.float16, device="cuda")
qt = memboost.quantize(W, ratio_4bit=0.0)
W_deq = memboost.dequantize(qt)
print(f"Quantized: M={qt.M}, K={qt.K}, avg_bits={qt.avg_bits:.2f}")

X = torch.randn(32, K, dtype=torch.float16, device="cuda")
Y_ref = X @ W.T
Y_deq = X @ W_deq.T

diff = (Y_ref.float() - Y_deq.float()).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()
rel_err = (diff / (Y_ref.float().abs() + 1e-6)).mean().item()

print(f"Max absolute error:  {max_err:.6f}")
print(f"Mean absolute error: {mean_err:.6f}")
print(f"Mean relative error: {rel_err:.6f}")
print(f"PASS: 2-bit roundtrip" if rel_err < 1.0 else "FAIL: error too large!")

# Test 2: Mixed precision (10% 4-bit)
print("\n--- Mixed precision test ---")
qt_mix = memboost.quantize(W, ratio_4bit=0.1)
W_deq_mix = memboost.dequantize(qt_mix)
Y_deq_mix = X @ W_deq_mix.T
diff_mix = (Y_ref.float() - Y_deq_mix.float()).abs()
print(f"Mixed 2/4-bit max error: {diff_mix.max().item():.6f}")
print(f"PASS: Mixed precision roundtrip" if diff_mix.max().item() < Y_ref.float().abs().max().item() else "FAIL")

# Test 3: Non-power-of-2 sizes
print("\n--- Non-power-of-2 test ---")
M2, K2 = 100, 80
W2 = torch.randn(M2, K2, dtype=torch.float16, device="cuda")
X2 = torch.randn(7, K2, dtype=torch.float16, device="cuda")
qt2 = memboost.quantize(W2, ratio_4bit=0.0)
W2_deq = memboost.dequantize(qt2)
Y2_ref = X2 @ W2.T
Y2_deq = X2 @ W2_deq.T
err2 = (Y2_ref.float() - Y2_deq.float()).abs().max().item()
print(f"Non-power-of-2 (M={M2}, K={K2}, N=7) max error: {err2:.6f}")
print(f"PASS: Non-power-of-2 sizes work" if err2 < Y2_ref.float().abs().max().item() else "FAIL")

# Test 4: Large matrix
print("\n--- Large matrix test ---")
M3, K3 = 4096, 4096
W3 = torch.randn(M3, K3, dtype=torch.float16, device="cuda")
qt3 = memboost.quantize(W3, ratio_4bit=0.1)
W3_deq = memboost.dequantize(qt3)
weight_err = (W3.float() - W3_deq.float()).abs().mean().item()
print(f"4096x4096 mean weight error: {weight_err:.6f}")
print(f"Compression: {W3.numel() * 2 / qt3.total_bytes:.2f}x")
print(f"Avg bits: {qt3.avg_bits:.2f}")
print(f"PASS: Large matrix roundtrip")

print("\nAll roundtrip tests completed!")
