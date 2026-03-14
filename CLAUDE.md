# Memboost — Development Guide

## Project Identity

Memboost is a **quantization engine**, not an inference engine. It compresses LLM weights from FP16 to mixed-precision 2/4-bit using custom CUDA kernels. It does not run models.

Based on: "Fast and Efficient 2-bit LLM Inference on GPU" (arXiv:2311.16442)

## Build & Test

```bash
# Build (requires CUDA toolkit + PyTorch with CUDA)
export CUDA_HOME=/usr/local/cuda
pip install -e .

# WSL note: if torch.cuda.is_available() returns False, restart WSL
wsl --shutdown

# Tests
python test_ops.py              # smoke test (pack/unpack, quantize/dequantize, serialization)
python tests/test_gemm.py       # roundtrip correctness across matrix sizes
python tests/test_quality.py    # GPT-2 weight quality diagnostic (needs transformers)

# C++ CUDA tests (no Python)
nvcc -DTEST_QUANTIZE -o test_quantize core/quantize.cu
./test_quantize
```

## Architecture

```
memboost/          Python API (quantize, dequantize, pack/unpack)
core/              CUDA kernels + PyBind11 bindings
models/            Quantization scripts for real models (GPT-2, Llama 2 GPTQ)
examples/          Basic usage demos
tests/             Correctness and quality tests
```

Key data flow: `FP16 weights → quantize() → QuantizedTensor → dequantize() → FP16`

The `QuantizedTensor` holds packed 2-bit/4-bit weights, hierarchical scales, group precision map, and sparse outliers in CSR format.

## Hard Rules

- **No inference code.** No generate(), no KV-cache, no attention kernels, no text generation. Memboost quantizes weights — inference engines consume the output.
- **No phantom APIs.** Never reference functions that don't exist. Every `memboost.something()` in Python must trace to a real binding in `core/bindings.cpp`.
- **No unnecessary code.** Every line must be defensible. No one-line wrappers, no "future use" functions, no premature abstractions.
- **No README/doc generation** unless explicitly requested.
- **No boilerplate.** Structure serves clarity, not aesthetics.

## CUDA Development

- All quantization kernels live in `core/quantize.cu` with declarations in `core/quantize.cuh`.
- Torch tensor ↔ CUDA bridge code lives in `core/torch_ops.cu`. This is where tensor checks, allocations, and kernel launches happen.
- Python-facing bindings are in `core/bindings.cpp` — one `m.def()` per exposed function.
- Use `CUDA_CHECK()` macro for all CUDA API calls in torch_ops.cu.
- Use `TORCH_CHECK()` for input validation (dtype, device, shape).
- Pack/unpack ops run on CPU (torch_ops.cu). Quantize/dequantize run on GPU.
- Group size constants: `GROUP_SIZE_1ST = 16`, `GROUP_SIZE_2ND = 256`. These must match between `.cuh` and `formats.py`.
- When adding a new kernel: add declaration in `.cuh`, implement in `.cu`, add host wrapper in `.cu`, add torch bridge in `torch_ops.cu`, add binding in `bindings.cpp`, add Python wrapper in `ops.py`.

## Python Conventions

- `memboost/__init__.py` defines `__all__` — every public API must be listed there.
- `formats.py` owns `QuantizedTensor`. Metrics like `avg_bits` and `total_bytes` must account for the fact that column groups span all M rows.
- `total_bytes` reports effective storage (not GPU allocation). `gpu_bytes` reports actual allocation.
- When quantizing without `hessian_diag`, all groups default to 2-bit (sensitivity computation is skipped).

## Code Quality

- Prefer the simplest correct solution. If logic fits in-place, keep it there.
- Functions exist only if reused or if they meaningfully improve clarity.
- Names: short, precise, intuitive. Prefer `n2`, `idx`, `cfg`, `err` over verbose alternatives.
- Comments: rare and intentional. Explain *why*, never *what* when the code is obvious.
- Do not sacrifice correctness for micro-optimizations.
- When performance matters (CUDA kernels), optimize deliberately: `#pragma unroll`, `__shfl_down_sync`, shared memory, coalesced access.

## Testing

- `test_ops.py` is the smoke test — must always pass. Tests pack/unpack roundtrip, quantize/dequantize, and serialization.
- 2-bit quantization of randn data has inherently high error (~1.0 mean abs). Test thresholds must reflect this — don't assert tight tolerances on 2-bit quantized random data.
- Every new Python API function needs a test in `test_ops.py` or a dedicated test file.
- CUDA kernel tests go in the `#ifdef TEST_QUANTIZE` block at the bottom of `quantize.cu`.

## Git

- Remote is HTTPS: `https://github.com/gouthamk16/memboost.git`
- Don't commit `.pyc` files or `.venv/`.
- Commit messages: lead with what changed and why, not how.
