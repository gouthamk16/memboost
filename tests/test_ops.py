"""Quick smoke test for the memboost Python interface."""
import torch
import memboost

print("memboost version:", memboost.__version__)

# Test 1: Pack/Unpack 2-bit roundtrip
v = torch.tensor([0,1,2,3]*4, dtype=torch.uint8)
p = memboost.pack_2bit(v)
u = memboost.unpack_2bit(p, 16)
assert torch.equal(v, u), f"2-bit roundtrip FAILED: {v} != {u}"
print("PASS: pack_2bit / unpack_2bit roundtrip")

# Test 2: Pack/Unpack 4-bit roundtrip
v4 = torch.tensor([0,3,7,15,1,8,12,4], dtype=torch.uint8)
p4 = memboost.pack_4bit(v4)
u4 = memboost.unpack_4bit(p4, 8)
assert torch.equal(v4, u4), f"4-bit roundtrip FAILED"
print("PASS: pack_4bit / unpack_4bit roundtrip")

# Test 3: Quantize / Dequantize roundtrip (pure 2-bit)
w = torch.randn(64, 64, dtype=torch.float16, device="cuda")
qt = memboost.quantize(w, ratio_4bit=0.0)
print(f"QuantizedTensor: M={qt.M}, K={qt.K}, avg_bits={qt.avg_bits:.2f}, nnz_outliers={qt.nnz}")
w_hat = memboost.dequantize(qt)
assert w_hat.shape == w.shape, f"Shape mismatch: {w_hat.shape} vs {w.shape}"
err = (w.float() - w_hat.float()).abs().mean().item()
print(f"PASS: quantize/dequantize roundtrip, mean_abs_error={err:.4f}")

# Test 4: Mixed precision with 4-bit ratio (no hessian = all 2-bit still)
qt2 = memboost.quantize(w, ratio_4bit=0.2)
w_hat2 = memboost.dequantize(qt2)
err2 = (w.float() - w_hat2.float()).abs().mean().item()
print(f"PASS: mixed 2/4-bit request, avg_bits={qt2.avg_bits:.2f}, error={err2:.4f}")

# Test 5: Serialization roundtrip
sd = qt.state_dict()
qt_loaded = memboost.QuantizedTensor.from_state_dict(sd)
w_hat3 = memboost.dequantize(qt_loaded)
assert torch.equal(w_hat, w_hat3), "Serialization roundtrip FAILED"
print("PASS: state_dict / from_state_dict serialization roundtrip")

print()
print("All tests passed!")
