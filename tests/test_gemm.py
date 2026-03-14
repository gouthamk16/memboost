"""Test quantize/dequantize roundtrip correctness across matrix sizes."""
import torch
import memboost

print("Testing quantize/dequantize roundtrip correctness...")

# Test 1: Basic weight roundtrip (pure 2-bit)
M, K = 256, 256
W = torch.randn(M, K, dtype=torch.float16, device="cuda")
qt = memboost.quantize(W, ratio_4bit=0.0)
W_deq = memboost.dequantize(qt)
print(f"Quantized: M={qt.M}, K={qt.K}, avg_bits={qt.avg_bits:.2f}")

weight_err = (W.float() - W_deq.float()).abs().mean().item()
weight_rel = weight_err / W.float().abs().mean().item()
print(f"Mean weight error:    {weight_err:.6f}")
print(f"Relative weight error: {weight_rel:.4f}")

# 2-bit quantization of randn(std=1) has ~4 levels over range ~6,
# so expected step size ~2.0, mean error ~0.5-1.0. rel_err < 2.0 is sane.
assert weight_rel < 2.0, f"Weight relative error too large: {weight_rel}"
assert qt.avg_bits > 1.5 and qt.avg_bits < 3.0, f"avg_bits out of range: {qt.avg_bits}"
print("PASS: 2-bit weight roundtrip")

# Test 2: Shape preservation
assert W_deq.shape == W.shape, f"Shape mismatch: {W_deq.shape} vs {W.shape}"
assert W_deq.dtype == torch.float16, f"Dtype mismatch: {W_deq.dtype}"
print("PASS: shape and dtype preserved")

# Test 3: Non-power-of-2 sizes
print("\n--- Non-power-of-2 test ---")
M2, K2 = 100, 80
W2 = torch.randn(M2, K2, dtype=torch.float16, device="cuda")
qt2 = memboost.quantize(W2, ratio_4bit=0.0)
W2_deq = memboost.dequantize(qt2)
assert W2_deq.shape == (M2, K2), f"Shape mismatch: {W2_deq.shape}"
err2 = (W2.float() - W2_deq.float()).abs().mean().item()
print(f"Non-power-of-2 (M={M2}, K={K2}) mean error: {err2:.6f}")
print(f"PASS: Non-power-of-2 sizes work")

# Test 4: Large matrix with compression check
print("\n--- Large matrix test ---")
M3, K3 = 4096, 4096
W3 = torch.randn(M3, K3, dtype=torch.float16, device="cuda")
qt3 = memboost.quantize(W3, ratio_4bit=0.0)
W3_deq = memboost.dequantize(qt3)
weight_err3 = (W3.float() - W3_deq.float()).abs().mean().item()
compression = W3.numel() * 2 / qt3.total_bytes
print(f"4096x4096 mean weight error: {weight_err3:.6f}")
print(f"Compression: {compression:.2f}x")
print(f"Avg bits: {qt3.avg_bits:.2f}")
assert compression > 3.0, f"Compression too low: {compression}"
assert qt3.avg_bits > 1.5 and qt3.avg_bits < 3.0, f"avg_bits out of range: {qt3.avg_bits}"
print("PASS: Large matrix roundtrip")

# Test 5: Serialization roundtrip preserves data
print("\n--- Serialization test ---")
sd = qt.state_dict()
qt_loaded = memboost.QuantizedTensor.from_state_dict(sd)
W_deq_loaded = memboost.dequantize(qt_loaded)
assert torch.equal(W_deq, W_deq_loaded), "Serialization roundtrip changed dequantized output!"
print("PASS: serialization roundtrip")

print("\nAll roundtrip tests completed!")
