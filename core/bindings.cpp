#include <torch/extension.h>
#include "torch_ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Memboost: mixed-precision 2/4-bit weight quantization engine";

    // Pack / Unpack
    m.def("pack_2bit", &memboost::pack_2bit_op,
          "Pack uint8 values (0-3) → int32 (16 values per word)",
          py::arg("values"));
    m.def("unpack_2bit", &memboost::unpack_2bit_op,
          "Unpack int32 → uint8 2-bit values",
          py::arg("packed"), py::arg("num_elements"));

    m.def("pack_4bit", &memboost::pack_4bit_op,
          "Pack uint8 values (0-15) → int32 (8 values per word)",
          py::arg("values"));
    m.def("unpack_4bit", &memboost::unpack_4bit_op,
          "Unpack int32 → uint8 4-bit values",
          py::arg("packed"), py::arg("num_elements"));

    // Quantize / Dequantize
    m.def("quantize", &memboost::quantize_op,
          "Quantize fp16 weights [M,K] → mixed 2/4-bit packed representation",
          py::arg("weights"),
          py::arg("ratio_4bit") = 0.1,
          py::arg("hessian_diag") = torch::Tensor());

    m.def("dequantize", &memboost::dequantize_op,
          "Dequantize packed weights → fp16 [M,K]",
          py::arg("packed_2bit"),
          py::arg("packed_4bit"),
          py::arg("scales_1st"),
          py::arg("zeros_1st"),
          py::arg("group_precision"),
          py::arg("M"), py::arg("K"));
}
