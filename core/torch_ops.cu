// Torch tensor <-> CUDA kernel bridge for memboost operations
#include "quantize.cuh"
#include "torch_ops.h"

#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <numeric>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while(0)

namespace memboost {

// Pack / Unpack — CPU bitwise operations on tensors
torch::Tensor pack_2bit_op(torch::Tensor values) {
    TORCH_CHECK(values.dtype() == torch::kUInt8, "values must be uint8");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");
    TORCH_CHECK(values.size(0) % 16 == 0, "length must be multiple of 16");

    values = values.cpu().contiguous();
    int64_t n = values.size(0);
    int64_t num_packed = n / 16;

    auto packed = torch::zeros({num_packed}, torch::dtype(torch::kInt32));
    auto* v = values.data_ptr<uint8_t>();
    auto* p = packed.data_ptr<int32_t>();

    for (int64_t i = 0; i < num_packed; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 16; j++) {
            word |= ((uint32_t)(v[i * 16 + j] & 0x3)) << (j * 2);
        }
        p[i] = static_cast<int32_t>(word);
    }
    return packed;
}

torch::Tensor unpack_2bit_op(torch::Tensor packed, int64_t num_elements) {
    TORCH_CHECK(packed.dtype() == torch::kInt32, "packed must be int32");
    TORCH_CHECK(packed.dim() == 1, "packed must be 1D");
    TORCH_CHECK(num_elements <= packed.size(0) * 16, "num_elements too large");

    packed = packed.cpu().contiguous();
    auto output = torch::zeros({num_elements}, torch::dtype(torch::kUInt8));
    auto* p = packed.data_ptr<int32_t>();
    auto* o = output.data_ptr<uint8_t>();

    for (int64_t i = 0; i < packed.size(0); i++) {
        uint32_t word = static_cast<uint32_t>(p[i]);
        for (int j = 0; j < 16 && i * 16 + j < num_elements; j++) {
            o[i * 16 + j] = (word >> (j * 2)) & 0x3;
        }
    }
    return output;
}

torch::Tensor pack_4bit_op(torch::Tensor values) {
    TORCH_CHECK(values.dtype() == torch::kUInt8, "values must be uint8");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");
    TORCH_CHECK(values.size(0) % 8 == 0, "length must be multiple of 8");

    values = values.cpu().contiguous();
    int64_t n = values.size(0);
    int64_t num_packed = n / 8;

    auto packed = torch::zeros({num_packed}, torch::dtype(torch::kInt32));
    auto* v = values.data_ptr<uint8_t>();
    auto* p = packed.data_ptr<int32_t>();

    for (int64_t i = 0; i < num_packed; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 8; j++) {
            word |= ((uint32_t)(v[i * 8 + j] & 0xF)) << (j * 4);
        }
        p[i] = static_cast<int32_t>(word);
    }
    return packed;
}

torch::Tensor unpack_4bit_op(torch::Tensor packed, int64_t num_elements) {
    TORCH_CHECK(packed.dtype() == torch::kInt32, "packed must be int32");
    TORCH_CHECK(packed.dim() == 1, "packed must be 1D");
    TORCH_CHECK(num_elements <= packed.size(0) * 8, "num_elements too large");

    packed = packed.cpu().contiguous();
    auto output = torch::zeros({num_elements}, torch::dtype(torch::kUInt8));
    auto* p = packed.data_ptr<int32_t>();
    auto* o = output.data_ptr<uint8_t>();

    for (int64_t i = 0; i < packed.size(0); i++) {
        uint32_t word = static_cast<uint32_t>(p[i]);
        for (int j = 0; j < 8 && i * 8 + j < num_elements; j++) {
            o[i * 8 + j] = (word >> (j * 4)) & 0xF;
        }
    }
    return output;
}

// Quantize — Full pipeline on CUDA
std::vector<torch::Tensor> quantize_op(
    torch::Tensor weights,
    double ratio_4bit,
    torch::Tensor hessian_diag
) {
    TORCH_CHECK(weights.dtype() == torch::kFloat16, "weights must be float16");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2D [M, K]");
    TORCH_CHECK(weights.is_cuda(), "weights must be on CUDA");

    weights = weights.contiguous();
    int M = weights.size(0);
    int K = weights.size(1);
    int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    int num_groups_2nd = (M * num_groups + GROUP_SIZE_2ND - 1) / GROUP_SIZE_2ND;

    auto dev = weights.device();
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto opts_f16  = torch::TensorOptions().dtype(torch::kFloat16).device(dev);
    auto opts_i32  = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opts_i8   = torch::TensorOptions().dtype(torch::kInt8).device(dev);
    auto opts_u8   = torch::TensorOptions().dtype(torch::kUInt8).device(dev);

    // Allocate intermediates and outputs
    auto group_min       = torch::empty({num_groups}, opts_f32);
    auto group_max       = torch::empty({num_groups}, opts_f32);
    auto group_precision = torch::zeros({num_groups}, opts_u8);
    auto packed_2bit     = torch::zeros({M, num_groups}, opts_i32);
    auto packed_4bit     = torch::zeros({M, num_groups * 2}, opts_i32);
    auto scales_1st      = torch::empty({M, num_groups}, opts_f16);
    auto zeros_1st       = torch::empty({M, num_groups}, opts_i8);
    auto outlier_mask    = torch::zeros({M, K}, opts_i32);

    auto* w_ptr = reinterpret_cast<const half*>(weights.data_ptr<at::Half>());

    // Step 1: Group statistics (min/max per column group)
    CUDA_CHECK(compute_group_stats(
        w_ptr,
        group_min.data_ptr<float>(),
        group_max.data_ptr<float>(),
        M, K
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Sensitivity-based precision assignment (optional)
    if (hessian_diag.defined() && hessian_diag.numel() > 0 && ratio_4bit > 0.0) {
        TORCH_CHECK(hessian_diag.dtype() == torch::kFloat32, "hessian_diag must be float32");
        TORCH_CHECK(hessian_diag.is_cuda(), "hessian_diag must be on CUDA");
        hessian_diag = hessian_diag.contiguous();

        auto sensitivity = torch::empty({num_groups}, opts_f32);
        CUDA_CHECK(compute_group_sensitivity(
            w_ptr,
            hessian_diag.data_ptr<float>(),
            sensitivity.data_ptr<float>(),
            M, K
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Sort on CPU and mark most-sensitive groups as 4-bit
        auto sens_cpu = sensitivity.cpu();
        auto* s = sens_cpu.data_ptr<float>();
        std::vector<int> idx(num_groups);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return s[a] > s[b];
        });

        int num_4bit = static_cast<int>(ratio_4bit * num_groups);
        auto prec_cpu = torch::zeros({num_groups}, torch::dtype(torch::kUInt8));
        auto* pp = prec_cpu.data_ptr<uint8_t>();
        for (int i = 0; i < num_4bit; i++) {
            pp[idx[i]] = 1;
        }
        group_precision = prec_cpu.to(dev);
    }

    // Step 3: Quantize weights
    CUDA_CHECK(memboost::quantize_weights(
        w_ptr,
        group_min.data_ptr<float>(),
        group_max.data_ptr<float>(),
        group_precision.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(packed_2bit.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(packed_4bit.data_ptr<int32_t>()),
        reinterpret_cast<half*>(scales_1st.data_ptr<at::Half>()),
        zeros_1st.data_ptr<int8_t>(),
        outlier_mask.data_ptr<int32_t>(),
        M, K
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: 2nd order scale quantization
    auto scales_2nd      = torch::empty({num_groups_2nd}, opts_f16);
    auto zeros_2nd       = torch::empty({num_groups_2nd}, opts_i8);
    auto scales_1st_quant = torch::empty({M, num_groups}, opts_u8);

    CUDA_CHECK(memboost::quantize_scales_2nd_order(
        reinterpret_cast<const half*>(scales_1st.data_ptr<at::Half>()),
        reinterpret_cast<half*>(scales_2nd.data_ptr<at::Half>()),
        zeros_2nd.data_ptr<int8_t>(),
        scales_1st_quant.data_ptr<uint8_t>(),
        M, num_groups
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Count outliers
    auto row_counts = torch::empty({M}, opts_i32);
    CUDA_CHECK(memboost::count_outliers(
        outlier_mask.data_ptr<int32_t>(),
        row_counts.data_ptr<int32_t>(),
        M, K
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Build row pointers on CPU (prefix sum)
    auto rc_cpu = row_counts.cpu();
    auto* rc = rc_cpu.data_ptr<int32_t>();
    auto row_ptrs_cpu = torch::empty({M + 1}, torch::dtype(torch::kInt32));
    auto* rp = row_ptrs_cpu.data_ptr<int32_t>();
    rp[0] = 0;
    for (int i = 0; i < M; i++) {
        rp[i + 1] = rp[i] + rc[i];
    }
    int nnz = rp[M];
    auto row_ptrs = row_ptrs_cpu.to(dev);

    // Step 6: Extract outlier values
    torch::Tensor outlier_values, outlier_col_indices;
    if (nnz > 0) {
        outlier_values     = torch::empty({nnz}, opts_f16);
        outlier_col_indices = torch::empty({nnz}, opts_i32);
        CUDA_CHECK(memboost::extract_outliers(
            w_ptr,
            outlier_mask.data_ptr<int32_t>(),
            row_ptrs.data_ptr<int32_t>(),
            reinterpret_cast<half*>(outlier_values.data_ptr<at::Half>()),
            outlier_col_indices.data_ptr<int32_t>(),
            M, K
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        outlier_values      = torch::empty({0}, opts_f16);
        outlier_col_indices = torch::empty({0}, opts_i32);
    }

    // Return all components as a vector of tensors
    return {
        packed_2bit,          
        packed_4bit,          
        scales_1st,           
        zeros_1st,            
        scales_2nd,           
        zeros_2nd,            
        scales_1st_quant,     
        group_precision,      
        outlier_values,       
        outlier_col_indices,  
        row_ptrs,             
    };
}

// Dequantize — CUDA kernel call
torch::Tensor dequantize_op(
    torch::Tensor packed_2bit,
    torch::Tensor packed_4bit,
    torch::Tensor scales_1st,
    torch::Tensor zeros_1st,
    torch::Tensor group_precision,
    int64_t M, int64_t K
) {
    TORCH_CHECK(packed_2bit.is_cuda(), "packed_2bit must be on CUDA");
    TORCH_CHECK(packed_2bit.dtype() == torch::kInt32, "packed_2bit must be int32");

    packed_2bit     = packed_2bit.contiguous();
    packed_4bit     = packed_4bit.contiguous();
    scales_1st      = scales_1st.contiguous();
    zeros_1st       = zeros_1st.contiguous();
    group_precision = group_precision.contiguous();

    auto output = torch::empty(
        {M, K},
        torch::TensorOptions().dtype(torch::kFloat16).device(packed_2bit.device())
    );

    CUDA_CHECK(memboost::dequantize_weights(
        reinterpret_cast<const uint32_t*>(packed_2bit.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(packed_4bit.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales_1st.data_ptr<at::Half>()),
        zeros_1st.data_ptr<int8_t>(),
        group_precision.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, K
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}
}
