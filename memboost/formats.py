from dataclasses import dataclass, field
from typing import Optional

import torch

# Match CUDA constants from quantize.cuh
GROUP_SIZE_1ST = 16
GROUP_SIZE_2ND = 256


@dataclass
class QuantizedTensor:
    """Complete quantized weight representation.

    Mirrors the C++ QuantizedTensor / QuantizedWeights / SparseOutliers structs.
    All tensor fields live on the same CUDA device as the original weights.
    """

    # Core packed weights — int32 because torch has no uint32
    packed_2bit: torch.Tensor  # [M, num_groups] int32
    packed_4bit: torch.Tensor  # [M, num_groups * 2] int32 (2 uint32 per 4-bit group)

    # 1st-order quantization parameters
    scales_1st: torch.Tensor  # [M, num_groups] fp16
    zeros_1st: torch.Tensor  # [M, num_groups] int8

    # 2nd-order quantization parameters (scales of scales)
    scales_2nd: torch.Tensor  # [num_groups_2nd] fp16
    zeros_2nd: torch.Tensor  # [num_groups_2nd] int8
    scales_1st_quant: torch.Tensor  # [M, num_groups] uint8

    # Per-group precision map: 0 = 2-bit, 1 = 4-bit
    group_precision: torch.Tensor  # [num_groups] uint8

    # Sparse outliers in CSR format
    outlier_values: torch.Tensor  # [nnz] fp16
    outlier_col_indices: torch.Tensor  # [nnz] int32
    outlier_row_ptrs: torch.Tensor  # [M+1] int32

    # Original dimensions
    M: int = 0
    K: int = 0

    @property
    def num_groups(self) -> int:
        return (self.K + GROUP_SIZE_1ST - 1) // GROUP_SIZE_1ST

    @property
    def nnz(self) -> int:
        return self.outlier_values.numel()

    @property
    def avg_bits(self) -> float:
        """Approximate average bits per weight element."""
        n2 = int((self.group_precision == 0).sum().item()) * GROUP_SIZE_1ST
        n4 = int((self.group_precision == 1).sum().item()) * GROUP_SIZE_1ST
        total = self.M * self.K
        if total == 0:
            return 0.0
        bits_w = n2 * 2.0 + n4 * 4.0
        bits_scale = (n2 + n4) / GROUP_SIZE_1ST * 6  # 4-bit scale + 2-bit zero
        bits_outlier = self.nnz * 32  # 16-bit value + 16-bit position
        return (bits_w + bits_scale + bits_outlier) / total

    @property
    def total_bytes(self) -> int:
        """Exact total memory footprint in bytes across all stored tensors."""
        return (
            self.packed_2bit.nbytes
            + self.packed_4bit.nbytes
            + self.scales_1st.nbytes
            + self.zeros_1st.nbytes
            + self.scales_2nd.nbytes
            + self.zeros_2nd.nbytes
            + self.scales_1st_quant.nbytes
            + self.group_precision.nbytes
            + self.outlier_values.nbytes
            + self.outlier_col_indices.nbytes
            + self.outlier_row_ptrs.nbytes
        )

    @property
    def total_mb(self) -> float:
        """Exact total memory footprint in megabytes."""
        return self.total_bytes / (1024 * 1024)

    def memory_breakdown(self) -> dict[str, float]:
        """Return per-component memory usage in MiB for debugging."""
        components = {
            "packed_2bit": self.packed_2bit.nbytes,
            "packed_4bit": self.packed_4bit.nbytes,
            "scales_1st": self.scales_1st.nbytes,
            "zeros_1st": self.zeros_1st.nbytes,
            "scales_2nd": self.scales_2nd.nbytes,
            "zeros_2nd": self.zeros_2nd.nbytes,
            "scales_1st_quant": self.scales_1st_quant.nbytes,
            "group_precision": self.group_precision.nbytes,
            "outlier_values": self.outlier_values.nbytes,
            "outlier_col_indices": self.outlier_col_indices.nbytes,
            "outlier_row_ptrs": self.outlier_row_ptrs.nbytes,
        }
        mib = {k: v / (1024 * 1024) for k, v in components.items()}
        for name, size in sorted(mib.items(), key=lambda x: -x[1]):
            print(f"  {name:25s} {size:8.3f} MiB")
        print(f"  {'TOTAL':25s} {sum(mib.values()):8.3f} MiB")
        return mib

    @property
    def device(self) -> torch.device:
        return self.packed_2bit.device

    def to(self, device: torch.device) -> "QuantizedTensor":
        """Move all tensors to the given device."""
        return QuantizedTensor(
            packed_2bit=self.packed_2bit.to(device),
            packed_4bit=self.packed_4bit.to(device),
            scales_1st=self.scales_1st.to(device),
            zeros_1st=self.zeros_1st.to(device),
            scales_2nd=self.scales_2nd.to(device),
            zeros_2nd=self.zeros_2nd.to(device),
            scales_1st_quant=self.scales_1st_quant.to(device),
            group_precision=self.group_precision.to(device),
            outlier_values=self.outlier_values.to(device),
            outlier_col_indices=self.outlier_col_indices.to(device),
            outlier_row_ptrs=self.outlier_row_ptrs.to(device),
            M=self.M,
            K=self.K,
        )

    def state_dict(self) -> dict:
        """Serialize to a dict of tensors (for torch.save)."""
        return {
            "packed_2bit": self.packed_2bit,
            "packed_4bit": self.packed_4bit,
            "scales_1st": self.scales_1st,
            "zeros_1st": self.zeros_1st,
            "scales_2nd": self.scales_2nd,
            "zeros_2nd": self.zeros_2nd,
            "scales_1st_quant": self.scales_1st_quant,
            "group_precision": self.group_precision,
            "outlier_values": self.outlier_values,
            "outlier_col_indices": self.outlier_col_indices,
            "outlier_row_ptrs": self.outlier_row_ptrs,
            "M": torch.tensor(self.M),
            "K": torch.tensor(self.K),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "QuantizedTensor":
        """Deserialize from a dict produced by state_dict()."""
        return cls(
            packed_2bit=d["packed_2bit"],
            packed_4bit=d["packed_4bit"],
            scales_1st=d["scales_1st"],
            zeros_1st=d["zeros_1st"],
            scales_2nd=d["scales_2nd"],
            zeros_2nd=d["zeros_2nd"],
            scales_1st_quant=d["scales_1st_quant"],
            group_precision=d["group_precision"],
            outlier_values=d["outlier_values"],
            outlier_col_indices=d["outlier_col_indices"],
            outlier_row_ptrs=d["outlier_row_ptrs"],
            M=int(d["M"].item()),
            K=int(d["K"].item()),
        )
