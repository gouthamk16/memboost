from typing import Optional
import torch

from . import _C
from .formats import QuantizedTensor

# Pack/unpack ops
def pack_2bit(values: torch.Tensor) -> torch.Tensor:
    """Pack uint8 2-bit values into int32 words (16 values per word).

    Args:
        values: 1-D uint8 tensor with values in [0, 3]. Length must be a
                multiple of 16.

    Returns:
        1-D int32 tensor of length ``len(values) // 16``.
    """
    return _C.pack_2bit(values)


def unpack_2bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Unpack int32 words into uint8 2-bit values.

    Args:
        packed: 1-D int32 tensor produced by :func:`pack_2bit`.
        num_elements: Number of uint8 values to extract.

    Returns:
        1-D uint8 tensor of length *num_elements*.
    """
    return _C.unpack_2bit(packed, num_elements)


def pack_4bit(values: torch.Tensor) -> torch.Tensor:
    """Pack uint8 4-bit values into int32 words (8 values per word).

    Args:
        values: 1-D uint8 tensor with values in [0, 15]. Length must be a
                multiple of 8.

    Returns:
        1-D int32 tensor of length ``len(values) // 8``.
    """
    return _C.pack_4bit(values)


def unpack_4bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Unpack int32 words into uint8 4-bit values.

    Args:
        packed: 1-D int32 tensor produced by :func:`pack_4bit`.
        num_elements: Number of uint8 values to extract.

    Returns:
        1-D uint8 tensor of length *num_elements*.
    """
    return _C.unpack_4bit(packed, num_elements)


# Quantize / Dequantize
def quantize(
    weights: torch.Tensor,
    ratio_4bit: float = 0.1,
    hessian_diag: Optional[torch.Tensor] = None,
) -> QuantizedTensor:
    """Quantize an fp16 weight matrix to 2/4-bit mixed precision.

    This runs the full quantization pipeline on GPU:
      1. Compute per-group min/max statistics
      2. (Optional) Sensitivity-based precision assignment via Hessian
      3. Quantize weights to packed 2-bit or 4-bit per group
      4. 2nd-order scale quantization
      5. Sparse outlier extraction (CSR)

    Args:
        weights: ``[M, K]`` fp16 tensor **on CUDA**.
        ratio_4bit: Fraction of groups to keep at 4-bit precision
                    (most-sensitive groups). Set to 0 for pure 2-bit.
        hessian_diag: Optional ``[K]`` fp32 tensor **on CUDA** with the
                      diagonal of the inverse Hessian. Required for
                      sensitivity-based precision assignment.

    Returns:
        A :class:`QuantizedTensor` holding all packed components.
    """
    if hessian_diag is None:
        hessian_diag = torch.Tensor()  # empty tensor signals "not provided"

    tensors = _C.quantize(weights, ratio_4bit, hessian_diag)

    M, K = weights.shape
    return QuantizedTensor(
        packed_2bit=tensors[0],
        packed_4bit=tensors[1],
        scales_1st=tensors[2],
        zeros_1st=tensors[3],
        scales_2nd=tensors[4],
        zeros_2nd=tensors[5],
        scales_1st_quant=tensors[6],
        group_precision=tensors[7],
        outlier_values=tensors[8],
        outlier_col_indices=tensors[9],
        outlier_row_ptrs=tensors[10],
        M=M,
        K=K,
    )


def dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize packed weights back to fp16.

    Applies ``w = scale * (q - zero)`` per group using the 1st-order
    scales and zeros stored in the :class:`QuantizedTensor`.

    .. note::
        This does **not** add back sparse outliers. The outlier values
        are available in ``qtensor.outlier_values`` /
        ``qtensor.outlier_col_indices`` / ``qtensor.outlier_row_ptrs``
        for manual reconstruction if needed.

    Args:
        qtensor: A :class:`QuantizedTensor` on CUDA.

    Returns:
        ``[M, K]`` fp16 tensor.
    """
    return _C.dequantize(
        qtensor.packed_2bit,
        qtensor.packed_4bit,
        qtensor.scales_1st,
        qtensor.zeros_1st,
        qtensor.group_precision,
        qtensor.M,
        qtensor.K,
    )
