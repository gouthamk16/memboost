# Memboost: Mixed-Precision 2/4-bit Weight Quantization Engine

CUDA-accelerated quantization engine that compresses LLM weights from FP16 to mixed-precision 2-bit/4-bit representations.

Based on ["Fast and Efficient 2-bit LLM Inference on GPU" (arXiv:2311.16442)](https://arxiv.org/abs/2311.16442).

![Range-aware quantization](assets/image-1.png)
![Calculation of quantization parameters](assets/image.png)

## What it does

Memboost takes FP16 weight matrices and quantizes them to ~2-3 bits per weight using:

- **Intra-matrix mixed-precision** — each column group is independently assigned 2-bit or 4-bit precision
- **Hessian-based sensitivity selection** — the most sensitive groups get 4-bit, the rest get 2-bit
- **Hierarchical scale quantization** — 1st-order scales are further compressed via 2nd-order 4-bit quantization
- **Sparse outlier extraction** — high-error weights are stored separately in CSR format

Typical compression: **5-6x** over FP16 with minimal quality loss.

## What it does NOT do

Memboost is a quantization engine, not an inference engine. It compresses weights, it does not run models. Use the quantized output with your inference engine of choice (vLLM, TensorRT-LLM, llama.cpp, etc.).

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- PyTorch (with CUDA support)

### Install from Source

```bash
export CUDA_HOME=/usr/local/cuda
pip install -e .
```

## Quick Start

```python
import torch
import memboost

# FP16 weights on CUDA
weights = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

# Quantize (10% of groups at 4-bit, rest at 2-bit)
qt = memboost.quantize(weights, ratio_4bit=0.1)

print(f"Average bits: {qt.avg_bits:.2f}")
print(f"Compression:  {weights.numel() * 2 / qt.total_bytes:.2f}x")

# Dequantize to verify quality
w_hat = memboost.dequantize(qt)
error = (weights - w_hat).abs().mean()
print(f"Mean error:   {error.item():.6f}")

# Save / Load
torch.save(qt.state_dict(), "quantized.pt")
loaded = memboost.QuantizedTensor.from_state_dict(torch.load("quantized.pt"))
```

## Quantizing Real Models

### GPT-2

Quantize all linear layers from a HuggingFace GPT-2 checkpoint:

```bash
python models/gpt2.py --model gpt2 --ratio-4bit 0.1 --verify
python models/gpt2.py --model gpt2-xl --save quantized_gpt2_xl.pt
```

### Llama 2 (GPTQ)

Layer-streaming GPTQ quantization with Hessian-based precision selection. Fits in 8GB VRAM:

```bash
python models/llama.py --model meta-llama/Llama-2-7b-hf --ratio-4bit 0.1
python models/llama.py --model meta-llama/Llama-2-7b-hf --save quantized_llama2.pt
```

The Llama quantizer uses calibration data from WikiText-2 to compute per-layer Hessians, then applies GPTQ optimal rounding before packing into the memboost format.

## API Reference

### Core Functions

| Function | Description |
|---|---|
| `memboost.quantize(weights, ratio_4bit, hessian_diag)` | Quantize FP16 `[M, K]` tensor to mixed 2/4-bit |
| `memboost.dequantize(qtensor)` | Dequantize back to FP16 (without outlier reconstruction) |

### QuantizedTensor

The `QuantizedTensor` dataclass holds all packed components:

| Field | Description |
|---|---|
| `packed_2bit` | `[M, num_groups]` int32 — 16 weights per word |
| `packed_4bit` | `[M, num_groups*2]` int32 — 8 weights per word |
| `scales_1st` / `zeros_1st` | Per-group scale and zero point |
| `scales_2nd` / `zeros_2nd` | 2nd-order scale quantization parameters |
| `group_precision` | `[num_groups]` uint8 — 0=2-bit, 1=4-bit |
| `outlier_values` / `outlier_col_indices` / `outlier_row_ptrs` | CSR sparse outliers |

Properties: `avg_bits`, `total_bytes`, `total_mb`, `nnz`, `memory_breakdown()`

### Low-Level Pack/Unpack

```python
# Pack 16 uint8 values (0-3) into one int32
values = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.uint8)
packed = memboost.pack_2bit(values)
unpacked = memboost.unpack_2bit(packed, num_elements=16)

# Pack 8 uint8 values (0-15) into one int32
packed_4 = memboost.pack_4bit(values_4bit)
unpacked_4 = memboost.unpack_4bit(packed_4, num_elements=8)
```

## Tests

```bash
# Python smoke test (pack/unpack roundtrip, quantize/dequantize, serialization)
python test_ops.py

# Quantization roundtrip correctness across matrix sizes
python tests/test_gemm.py

# GPT-2 weight quantization quality diagnostic (requires transformers)
python tests/test_quality.py
```

### C++ CUDA Tests

```bash
nvcc -DTEST_QUANTIZE -o test_quantize core/quantize.cu
./test_quantize
```

## Project Structure

```
memboost/
├── memboost/
│   ├── __init__.py          # Public API exports
│   ├── ops.py               # Python bindings to CUDA ops
│   └── formats.py           # QuantizedTensor dataclass
├── core/
│   ├── bindings.cpp          # PyBind11 module definition
│   ├── torch_ops.cu          # Torch tensor <-> CUDA bridge
│   ├── torch_ops.h
│   ├── quantize.cu           # CUDA quantization kernels
│   └── quantize.cuh          # Kernel declarations + structs
├── models/
│   ├── gpt2.py               # GPT-2 weight quantization
│   └── llama.py              # Llama 2 GPTQ quantization
├── examples/
│   └── example.py            # Basic compression demo
└── tests/
    ├── test_ops.py            # Smoke tests
    ├── test_gemm.py           # Roundtrip correctness
    └── test_quality.py        # GPT-2 quality diagnostic
```
