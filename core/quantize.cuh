#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace memboost {

constexpr int GROUP_SIZE_1ST = 16; // 1st order quantization group size
constexpr int GROUP_SIZE_2ND = 256; // 2nd order quantization group size  
constexpr int WARP_SIZE = 32;
constexpr float OUTLIER_THRESHOLD = 0.005f; // 0.5% outlier ratio max

/**
 * Packed quantized weights structure
 * Stores 2-bit and 4-bit weights in separate packed arrays
 */
struct QuantizedWeights {
    uint32_t* weights_2bit;      
    uint32_t* weights_4bit;      
    
    // 1st order scales and zeros (per group)
    half* scales_1st;            
    int8_t* zeros_1st;           
    
    // 2nd order scales and zeros (for scale quantization)
    half* scales_2nd;            
    int8_t* zeros_2nd;           
    
    // Group precision map: 0 = 2-bit, 1 = 4-bit
    uint8_t* group_precision;    
    
    int num_rows;
    int num_cols;
    int num_groups;
    int num_4bit_groups;
};

/**
 * CSR format for sparse outliers
 * Stores outliers that exceed quantization error threshold
 */
struct SparseOutliers {
    half* values; // Outlier values (FP16)
    int* col_indices; // Column indices
    int* row_ptrs; // Row pointers
    int nnz; // Number of non-zeros
    int num_rows;
};

/**
 * Complete quantized tensor with all components
 */
struct QuantizedTensor {
    QuantizedWeights weights;
    SparseOutliers outliers;
    
    // Original dimensions
    int M; // Output dimension (rows)
    int K; // Input dimension (cols)
    
    // Statistics
    float avg_bits;
    float outlier_ratio;
};

struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
};

/**
 * Quantize FP16 weights to 2/4-bit mixed precision
 * 
 * @param weights_fp16   Input FP16 weights [M x K]
 * @param hessian_inv    Hessian inverse diagonal for sensitivity [K]
 * @param output         Output quantized tensor
 * @param M              Number of rows (output dim)
 * @param K              Number of cols (input dim)
 * @param ratio_4bit     Target ratio of 4-bit groups (e.g., 0.1 = 10%)
 * @param stream         CUDA stream
 */
cudaError_t quantize_mixed_precision(
    const half* weights_fp16,
    const float* hessian_inv,
    QuantizedTensor* output,
    int M, int K,
    float ratio_4bit,
    cudaStream_t stream = 0
);

cudaError_t compute_group_sensitivity(
    const half* weights,
    const float* hessian_inv,
    float* sensitivity,
    int M, int K,
    cudaStream_t stream = 0
);

cudaError_t allocate_quantized_tensor(QuantizedTensor* tensor, int M, int K);
cudaError_t free_quantized_tensor(QuantizedTensor* tensor);

inline float calculate_avg_bits(int num_2bit, int num_4bit, int group_size, int nnz_outliers, int total) {
    float bits_2 = num_2bit * 2.0f;
    float bits_4 = num_4bit * 4.0f;
    float bits_scale = (num_2bit + num_4bit) / group_size * (4 + 2);
    float bits_outlier = nnz_outliers * (16 + 16);
    return (bits_2 + bits_4 + bits_scale + bits_outlier) / total;
}

//Host wrappers for individual kernels (used by Python bindings)
cudaError_t compute_group_stats(
    const half* weights, float* group_min, float* group_max,
    int M, int K, cudaStream_t stream = 0
);

cudaError_t quantize_weights(
    const half* weights, const float* group_min, const float* group_max,
    const uint8_t* group_precision,
    uint32_t* packed_2bit, uint32_t* packed_4bit,
    half* scales_1st, int8_t* zeros_1st, int* outlier_mask,
    int M, int K, cudaStream_t stream = 0
);

cudaError_t quantize_scales_2nd_order(
    const half* scales_1st, half* scales_2nd, int8_t* zeros_2nd,
    uint8_t* scales_1st_quant,
    int M, int num_groups_1st, cudaStream_t stream = 0
);

cudaError_t count_outliers(
    const int* outlier_mask, int* row_counts,
    int M, int K, cudaStream_t stream = 0
);

cudaError_t extract_outliers(
    const half* weights, const int* outlier_mask, const int* row_ptrs,
    half* values, int* col_indices,
    int M, int K, cudaStream_t stream = 0
);

cudaError_t dequantize_weights(
    const uint32_t* packed_2bit, const uint32_t* packed_4bit,
    const half* scales_1st, const int8_t* zeros_1st,
    const uint8_t* group_precision, half* output,
    int M, int K, cudaStream_t stream = 0
);

} 
