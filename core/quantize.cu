/* Generally - weights are represented in fp16 (16 bits)
 What we plan to do now - represent weights in int2 (2 bits)

 Goal of this file:  Take weights in fp16 and convert them to int2
 Algo: Intra-matrix mixed-precision quantization based on: "Fast and Efficient 2-bit LLM Inference on GPU" (arxiv 2311.16442)

 Quantization basic theory
A group of weights with range (min, max) can be quantized to int2 (2 bits) in the range of (0, 2^N-1), zero point(N bits int) and scale factor(half)

Components:
 1. Intra-weight mixed-precision quantization (2/4-bit)
 2. Hierarchical 2nd-order scale quantization
 3. Sparse outlier detection and extraction (CSR format)
 4. Dequantization for roundtrip verification
*/

#include "quantize.cuh"
#include <cuda_fp16.h>
#include <cstdio>

namespace memboost {

// Get the absolute value of fp16
__device__ __forceinline__ half habs(half x) {
    unsigned short bits = reinterpret_cast<unsigned short&>(x);
    bits &= 0x7FFF;
    return reinterpret_cast<half&>(bits);
}
// Convert fp16 to fp32
__device__ __forceinline__ float hto_float(half x) {
    return __half2float(x);
}
// Convert fp32 to fp16
__device__ __forceinline__ half float_to_half(float x) {
    return __float2half(x);
}

// Pack 16 2 bit values into a single uint32
__device__ __forceinline__ uint32_t pack_2bit(uint8_t vals[16]) {
    uint32_t packed = 0;
    #pragma unroll
    for (int i=0; i<16; i++) {
        packed |= ((uint32_t)(vals[i] & 0x3)) << (i*2);
    }
    return packed;
}
// Unpack 16 2 bit values from a single uint32
__device__ __forceinline__ void unpack_2bit(uint32_t packed, uint8_t vals[16]) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        vals[i] = (packed >> (i * 2)) & 0x3;
    }
}
// Pack 8 4 bit values into a single uint32
__device__ __forceinline__ uint32_t pack_4bit(uint8_t vals[8]) {
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        packed |= ((uint32_t)(vals[i] & 0xF)) << (i * 4);
    }
    return packed;
}
// Unpack 8 4 bit values from a uint32
__device__ __forceinline__ void unpack_4bit(uint32_t packed, uint8_t vals[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        vals[i] = (packed >> (i * 4)) & 0xF;
    }
}

// Kernel to compute group sensitivity
__global__ void compute_sensitivity_kernel(
    const half* __restrict__ weights,
    const float* __restrict__ hessian_inv_diag,
    float* __restrict__ sensitivity,
    int M, int K, int num_groups
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= num_groups) return;
    
    // Each group covers GROUP_SIZE_1ST columns
    int col_start = group_idx * GROUP_SIZE_1ST;
    int col_end = min(col_start + GROUP_SIZE_1ST, K);
    
    float sum = 0.0f;
    
    for (int row = 0; row < M; row++) {
        for (int col = col_start; col < col_end; col++) {
            float w = hto_float(weights[row * K + col]);
            float h_inv = hessian_inv_diag[col];
            // Sensitivity = (w / sqrt(H^-1))^2 = w^2 * H
            float s = (h_inv > 1e-8f) ? (w * w / h_inv) : (w * w);
            sum += s;
        }
    }
    
    sensitivity[group_idx] = sum;
}

// Main kernels related to quantization

// Find min and max per group 
__global__ void compute_group_stats_kernel(
    const half* __restrict__ weights,
    float* __restrict__ group_min,
    float* __restrict__ group_max,
    int M, int K, int num_groups
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= num_groups) return;
    
    int col_start = group_idx * GROUP_SIZE_1ST;
    int col_end = min(col_start + GROUP_SIZE_1ST, K);
    
    float min_val = 1e10f;
    float max_val = -1e10f;
    
    for (int row = 0; row < M; row++) {
        for (int col = col_start; col < col_end; col++) {
            float w = hto_float(weights[row * K + col]);
            min_val = fminf(min_val, w);
            max_val = fmaxf(max_val, w);
        }
    }
    
    group_min[group_idx] = min_val;
    group_max[group_idx] = max_val;
}

// Kernel to quantizze weights to 2 bit (or 4 bits based on gp)
// Here the data type used to store the 2 bit weights will be uint8 since cuda does not have any int2 or int4 dtype, later these weights are packed to store them as groups of 2 bits (n * 2 total bits)
__global__ void quantize_weights_kernel(
    const half* __restrict__ weights,
    const float* __restrict__ group_min,
    const float* __restrict__ group_max,
    const uint8_t* __restrict__ group_precision, // 0=2bit, 1=4bit
    uint32_t* __restrict__ packed_2bit,
    uint32_t* __restrict__ packed_4bit,
    half* __restrict__ scales_1st,
    int8_t* __restrict__ zeros_1st,
    int* __restrict__ outlier_mask, 
    int M, int K, int num_groups
) {
    int row = blockIdx.y;
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx >= num_groups || row >= M) return;
    
    int col_start = group_idx * GROUP_SIZE_1ST;
    int col_end = min(col_start + GROUP_SIZE_1ST, K);
    int group_size = col_end - col_start;
    
    float gmin = group_min[group_idx];
    float gmax = group_max[group_idx];
    
    uint8_t precision = group_precision[group_idx];
    int num_levels = (precision == 0) ? 4 : 16; // 2^2 or 2^4
    
    // Compute scale and zero point
    float scale = (gmax - gmin) / (float)(num_levels - 1);
    if (scale < 1e-8f) scale = 1e-8f;
    
    int zero = (int)roundf(-gmin / scale);
    zero = max(0, min(num_levels - 1, zero));
    
    // Store scale and zero for this group
    int scale_idx = row * num_groups + group_idx;
    scales_1st[scale_idx] = float_to_half(scale);
    zeros_1st[scale_idx] = (int8_t)zero;
    
    // Quantize each weight in the group
    // Outlier threshold: must be > max rounding error for that precision.
    // 2-bit (4 levels): max rounding error = range / (2*(4-1)) = range/6 ≈ 0.167*range
    // 4-bit (16 levels): max rounding error = range / (2*(16-1)) = range/30 ≈ 0.033*range
    float range = gmax - gmin;
    float outlier_thresh = (precision == 0) ? 0.4f * range : 0.15f * range;

    if (precision == 0) {
        // 2-bit quantization
        uint8_t quant_vals[16] = {0};
        
        for (int i = 0; i < group_size; i++) {
            int col = col_start + i;
            float w = hto_float(weights[row * K + col]);
            
            // Quantize
            int q = (int)roundf((w - gmin) / scale);
            q = max(0, min(3, q));
            quant_vals[i] = (uint8_t)q;
            
            // Check for outliers (dequant error > threshold)
            float w_dequant = scale * (q - zero);
            float error = fabsf(w - w_dequant);
            if (error > outlier_thresh) {
                // Mark as outlier
                atomicOr(&outlier_mask[row * K + col], 1);
            }
        }
        
        // Pack 16 2-bit values into contiguous storage
        packed_2bit[row * num_groups + group_idx] = pack_2bit(quant_vals);
        
    } else {
        // 4-bit quantization — 16 values per group, packed into 2 uint32 words
        uint8_t quant_vals_lo[8] = {0};  // first 8 values
        uint8_t quant_vals_hi[8] = {0};  // second 8 values
        
        for (int i = 0; i < min(8, group_size); i++) {
            int col = col_start + i;
            float w = hto_float(weights[row * K + col]);
            
            int q = (int)roundf((w - gmin) / scale);
            q = max(0, min(15, q));
            quant_vals_lo[i] = (uint8_t)q;

            // Check for outliers
            float w_dequant = scale * (q - zero);
            float error = fabsf(w - w_dequant);
            if (error > outlier_thresh) {
                atomicOr(&outlier_mask[row * K + col], 1);
            }
        }
        
        for (int i = 8; i < min(16, group_size); i++) {
            int col = col_start + i;
            float w = hto_float(weights[row * K + col]);
            
            int q = (int)roundf((w - gmin) / scale);
            q = max(0, min(15, q));
            quant_vals_hi[i - 8] = (uint8_t)q;

            float w_dequant = scale * (q - zero);
            float error = fabsf(w - w_dequant);
            if (error > outlier_thresh) {
                atomicOr(&outlier_mask[row * K + col], 1);
            }
        }
        
        packed_4bit[row * num_groups * 2 + group_idx * 2]     = pack_4bit(quant_vals_lo);
        packed_4bit[row * num_groups * 2 + group_idx * 2 + 1] = pack_4bit(quant_vals_hi);
    }
}

// Kernel to quantize the 1st order scales to 4 bits
__global__ void quantize_scales_2nd_order_kernel(
    const half* __restrict__ scales_1st,
    half* __restrict__ scales_2nd,
    int8_t* __restrict__ zeros_2nd,
    uint8_t* __restrict__ scales_1st_quant,
    int M, int num_groups_1st, int num_groups_2nd
) {
    int group_2nd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_2nd_idx >= num_groups_2nd) return;
    
    int start = group_2nd_idx * GROUP_SIZE_2ND;
    int end = min(start + GROUP_SIZE_2ND, M * num_groups_1st);
    
    // Find min/max of scales in this 2nd order group
    float min_s = 1e10f, max_s = -1e10f;
    for (int i = start; i < end; i++) {
        float s = hto_float(scales_1st[i]);
        min_s = fminf(min_s, s);
        max_s = fmaxf(max_s, s);
    }
    
    // 4-bit quantization for scales
    float scale_2nd = (max_s - min_s) / 15.0f;
    if (scale_2nd < 1e-10f) scale_2nd = 1e-10f;
    int zero_2nd = (int)roundf(-min_s / scale_2nd);
    
    scales_2nd[group_2nd_idx] = float_to_half(scale_2nd);
    zeros_2nd[group_2nd_idx] = (int8_t)zero_2nd;
    
    // Quantize each 1st order scale
    for (int i = start; i < end; i++) {
        float s = hto_float(scales_1st[i]);
        int q = (int)roundf((s - min_s) / scale_2nd);
        q = max(0, min(15, q));
        scales_1st_quant[i] = (uint8_t)q;
    }
}

// Sparse Outlier Extraction
__global__ void count_outliers_kernel(
    const int* __restrict__ outlier_mask,
    int* __restrict__ row_counts,
    int M, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    int count = 0;
    for (int col = 0; col < K; col++) {
        if (outlier_mask[row * K + col]) {
            count++;
        }
    }
    row_counts[row] = count;
}

__global__ void extract_outliers_kernel(
    const half* __restrict__ weights,
    const int* __restrict__ outlier_mask,
    const int* __restrict__ row_ptrs,
    half* __restrict__ values,
    int* __restrict__ col_indices,
    int M, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    int write_idx = row_ptrs[row];
    
    for (int col = 0; col < K; col++) {
        if (outlier_mask[row * K + col]) {
            values[write_idx] = weights[row * K + col];
            col_indices[write_idx] = col;
            write_idx++;
        }
    }
}

// Host API Implementation
cudaError_t compute_group_sensitivity(
    const half* weights,
    const float* hessian_inv,
    float* sensitivity,
    int M, int K,
    cudaStream_t stream
) {
    int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    int block_size = 256;
    int grid_size = (num_groups + block_size - 1) / block_size;
    
    compute_sensitivity_kernel<<<grid_size, block_size, 0, stream>>>(
        weights, hessian_inv, sensitivity, M, K, num_groups
    );
    
    return cudaGetLastError();
}

cudaError_t allocate_quantized_tensor(QuantizedTensor* tensor, int M, int K) {
    tensor->M = M;
    tensor->K = K;
    
    int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    tensor->weights.num_groups = num_groups;
    tensor->weights.num_rows = M;
    tensor->weights.num_cols = K;
    
    // Allocate weight storage
    cudaMalloc(&tensor->weights.weights_2bit, M * num_groups * sizeof(uint32_t));
    cudaMalloc(&tensor->weights.weights_4bit, M * num_groups * 2 * sizeof(uint32_t));
    
    // Allocate scales and zeros
    cudaMalloc(&tensor->weights.scales_1st, M * num_groups * sizeof(half));
    cudaMalloc(&tensor->weights.zeros_1st, M * num_groups * sizeof(int8_t));
    
    int num_groups_2nd = (M * num_groups + GROUP_SIZE_2ND - 1) / GROUP_SIZE_2ND;
    cudaMalloc(&tensor->weights.scales_2nd, num_groups_2nd * sizeof(half));
    cudaMalloc(&tensor->weights.zeros_2nd, num_groups_2nd * sizeof(int8_t));
    
    cudaMalloc(&tensor->weights.group_precision, num_groups * sizeof(uint8_t));
    
    return cudaGetLastError();
}

cudaError_t free_quantized_tensor(QuantizedTensor* tensor) {
    cudaFree(tensor->weights.weights_2bit);
    cudaFree(tensor->weights.weights_4bit);
    cudaFree(tensor->weights.scales_1st);
    cudaFree(tensor->weights.zeros_1st);
    cudaFree(tensor->weights.scales_2nd);
    cudaFree(tensor->weights.zeros_2nd);
    cudaFree(tensor->weights.group_precision);
    
    if (tensor->outliers.values) cudaFree(tensor->outliers.values);
    if (tensor->outliers.col_indices) cudaFree(tensor->outliers.col_indices);
    if (tensor->outliers.row_ptrs) cudaFree(tensor->outliers.row_ptrs);
    
    return cudaGetLastError();
}



__global__ void dequantize_weights_kernel(
    const uint32_t* __restrict__ packed_2bit,
    const uint32_t* __restrict__ packed_4bit,
    const half* __restrict__ scales_1st,
    const int8_t* __restrict__ zeros_1st,
    const uint8_t* __restrict__ group_precision,
    half* __restrict__ output,
    int M, int K, int num_groups
) {
    int row = blockIdx.y;
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= num_groups || row >= M) return;

    int col_start = group_idx * GROUP_SIZE_1ST;
    int col_end = min(col_start + GROUP_SIZE_1ST, K);

    int scale_idx = row * num_groups + group_idx;
    float scale = hto_float(scales_1st[scale_idx]);
    int zero = zeros_1st[scale_idx];

    uint8_t precision = group_precision[group_idx];

    if (precision == 0) {
        uint32_t packed = packed_2bit[row * num_groups + group_idx];
        uint8_t vals[16];
        unpack_2bit(packed, vals);

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (col_start + i < K) {
                float w = scale * ((float)vals[i] - zero);
                output[row * K + col_start + i] = float_to_half(w);
            }
        }
    } else {
        // 4-bit: 2 uint32 words per group (first 8 + second 8 values)
        uint32_t packed_lo = packed_4bit[row * num_groups * 2 + group_idx * 2];
        uint32_t packed_hi = packed_4bit[row * num_groups * 2 + group_idx * 2 + 1];
        uint8_t vals_lo[8], vals_hi[8];
        unpack_4bit(packed_lo, vals_lo);
        unpack_4bit(packed_hi, vals_hi);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (col_start + i < K) {
                float w = scale * ((float)vals_lo[i] - zero);
                output[row * K + col_start + i] = float_to_half(w);
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (col_start + 8 + i < K) {
                float w = scale * ((float)vals_hi[i] - zero);
                output[row * K + col_start + 8 + i] = float_to_half(w);
            }
        }
    }
}

// Host wrappers for Python bindings
cudaError_t compute_group_stats(
    const half* weights, float* group_min, float* group_max,
    int M, int K, cudaStream_t stream
) {
    int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    int block_size = 256;
    int grid_size = (num_groups + block_size - 1) / block_size;

    compute_group_stats_kernel<<<grid_size, block_size, 0, stream>>>(
        weights, group_min, group_max, M, K, num_groups
    );
    return cudaGetLastError();
}

cudaError_t quantize_weights(
    const half* weights, const float* group_min, const float* group_max,
    const uint8_t* group_precision,
    uint32_t* packed_2bit, uint32_t* packed_4bit,
    half* scales_1st, int8_t* zeros_1st, int* outlier_mask,
    int M, int K, cudaStream_t stream
) {
    int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    dim3 grid((num_groups + 31) / 32, M);
    dim3 block(32);

    quantize_weights_kernel<<<grid, block, 0, stream>>>(
        weights, group_min, group_max, group_precision,
        packed_2bit, packed_4bit, scales_1st, zeros_1st,
        outlier_mask, M, K, num_groups
    );
    return cudaGetLastError();
}

cudaError_t quantize_scales_2nd_order(
    const half* scales_1st, half* scales_2nd, int8_t* zeros_2nd,
    uint8_t* scales_1st_quant,
    int M, int num_groups_1st, cudaStream_t stream
) {
    int total_scales = M * num_groups_1st;
    int num_groups_2nd = (total_scales + GROUP_SIZE_2ND - 1) / GROUP_SIZE_2ND;
    int block_size = 256;
    int grid_size = (num_groups_2nd + block_size - 1) / block_size;

    quantize_scales_2nd_order_kernel<<<grid_size, block_size, 0, stream>>>(
        scales_1st, scales_2nd, zeros_2nd, scales_1st_quant,
        M, num_groups_1st, num_groups_2nd
    );
    return cudaGetLastError();
}

cudaError_t count_outliers(
    const int* outlier_mask, int* row_counts,
    int M, int K, cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;

    count_outliers_kernel<<<grid_size, block_size, 0, stream>>>(
        outlier_mask, row_counts, M, K
    );
    return cudaGetLastError();
}

cudaError_t extract_outliers(
    const half* weights, const int* outlier_mask, const int* row_ptrs,
    half* values, int* col_indices,
    int M, int K, cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;

    extract_outliers_kernel<<<grid_size, block_size, 0, stream>>>(
        weights, outlier_mask, row_ptrs, values, col_indices, M, K
    );
    return cudaGetLastError();
}

cudaError_t dequantize_weights(
    const uint32_t* packed_2bit, const uint32_t* packed_4bit,
    const half* scales_1st, const int8_t* zeros_1st,
    const uint8_t* group_precision, half* output,
    int M, int K, cudaStream_t stream
) {
    int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    dim3 grid((num_groups + 31) / 32, M);
    dim3 block(32);

    dequantize_weights_kernel<<<grid, block, 0, stream>>>(
        packed_2bit, packed_4bit, scales_1st, zeros_1st,
        group_precision, output, M, K, num_groups
    );
    return cudaGetLastError();
}

}

// Comprehensive test suite for the quantization implementation
#ifdef TEST_QUANTIZE


#include <vector>
#include <random>
#include <chrono>
#include <cmath>

// Bring constants into scope for tests
using memboost::GROUP_SIZE_1ST;
using memboost::GROUP_SIZE_2ND;
using memboost::WARP_SIZE;

void test_pack_unpack() {
    printf("Test 1: 2-bit Pack/Unpack\n");
    
    uint8_t original[16];
    for (int i = 0; i < 16; i++) {
        original[i] = i % 4; // 2-bit values
    }
    
    // Pack on CPU (simulating device function)
    uint32_t packed = 0;
    for (int i = 0; i < 16; i++) {
        packed |= ((uint32_t)(original[i] & 0x3)) << (i * 2);
    }
    
    // Unpack
    uint8_t unpacked[16];
    for (int i = 0; i < 16; i++) {
        unpacked[i] = (packed >> (i * 2)) & 0x3;
    }
    
    bool passed = true;
    for (int i = 0; i < 16; i++) {
        if (original[i] != unpacked[i]) {
            printf("  FAIL at index %d: expected %d, got %d\n", i, original[i], unpacked[i]);
            passed = false;
        }
    }
    
    printf("  Result: %s\n\n", passed ? "PASSED" : "FAILED");
}

void test_4bit_pack_unpack() {
    printf("Test 2: 4-bit Pack/Unpack\n");
    
    uint8_t original[8];
    for (int i = 0; i < 8; i++) {
        original[i] = i * 2; // 4-bit values
    }
    
    // Pack
    uint32_t packed = 0;
    for (int i = 0; i < 8; i++) {
        packed |= ((uint32_t)(original[i] & 0xF)) << (i * 4);
    }
    
    // Unpack
    uint8_t unpacked[8];
    for (int i = 0; i < 8; i++) {
        unpacked[i] = (packed >> (i * 4)) & 0xF;
    }
    
    bool passed = true;
    for (int i = 0; i < 8; i++) {
        if (original[i] != unpacked[i]) {
            printf("  FAIL at index %d: expected %d, got %d\n", i, original[i], unpacked[i]);
            passed = false;
        }
    }
    
    printf("  Result: %s\n\n", passed ? "PASSED" : "FAILED");
}

void test_group_stats_kernel() {
    printf("Test 3: Group Statistics Kernel\n");
    
    const int M = 64;
    const int K = 64;
    const int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    
    // Create weights with known min/max per group
    std::vector<half> h_weights(M * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; i++) {
        h_weights[i] = __float2half(dist(gen));
    }
    
    // Allocate device memory
    half* d_weights;
    float *d_group_min, *d_group_max;
    cudaMalloc(&d_weights, M * K * sizeof(half));
    cudaMalloc(&d_group_min, num_groups * sizeof(float));
    cudaMalloc(&d_group_max, num_groups * sizeof(float));
    
    cudaMemcpy(d_weights, h_weights.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    
    // Run kernel
    int block_size = 256;
    int grid_size = (num_groups + block_size - 1) / block_size;
    memboost::compute_group_stats_kernel<<<grid_size, block_size>>>(
        d_weights, d_group_min, d_group_max, M, K, num_groups
    );
    cudaDeviceSynchronize();
    
    // Get results
    std::vector<float> h_group_min(num_groups), h_group_max(num_groups);
    cudaMemcpy(h_group_min.data(), d_group_min, num_groups * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_group_max.data(), d_group_max, num_groups * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify against CPU reference
    bool passed = true;
    for (int g = 0; g < num_groups; g++) {
        int col_start = g * GROUP_SIZE_1ST;
        int col_end = std::min(col_start + GROUP_SIZE_1ST, K);
        
        float cpu_min = 1e10f, cpu_max = -1e10f;
        for (int row = 0; row < M; row++) {
            for (int col = col_start; col < col_end; col++) {
                float w = __half2float(h_weights[row * K + col]);
                cpu_min = std::min(cpu_min, w);
                cpu_max = std::max(cpu_max, w);
            }
        }
        
        if (std::abs(cpu_min - h_group_min[g]) > 1e-5f || std::abs(cpu_max - h_group_max[g]) > 1e-5f) {
            printf("  FAIL group %d: CPU min/max = %.4f/%.4f, GPU = %.4f/%.4f\n",
                   g, cpu_min, cpu_max, h_group_min[g], h_group_max[g]);
            passed = false;
        }
    }
    
    printf("  Groups tested: %d\n", num_groups);
    printf("  Result: %s\n\n", passed ? "PASSED" : "FAILED");
    
    cudaFree(d_weights);
    cudaFree(d_group_min);
    cudaFree(d_group_max);
}

void test_full_quantization_pipeline() {
    printf("Test 4: Full Quantization Pipeline\n");
    
    const int M = 128;
    const int K = 128;
    const int num_groups = (K + GROUP_SIZE_1ST - 1) / GROUP_SIZE_1ST;
    
    printf("  Matrix: %d x %d, Groups: %d\n", M, K, num_groups);
    
    // Create random weights
    std::vector<half> h_weights(M * K);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    
    for (int i = 0; i < M * K; i++) {
        h_weights[i] = __float2half(dist(gen));
    }
    
    // Allocate device memory
    half* d_weights;
    float *d_group_min, *d_group_max;
    uint8_t* d_group_precision;
    uint32_t *d_packed_2bit, *d_packed_4bit;
    half* d_scales_1st;
    int8_t* d_zeros_1st;
    int* d_outlier_mask;
    
    cudaMalloc(&d_weights, M * K * sizeof(half));
    cudaMalloc(&d_group_min, num_groups * sizeof(float));
    cudaMalloc(&d_group_max, num_groups * sizeof(float));
    cudaMalloc(&d_group_precision, num_groups * sizeof(uint8_t));
    cudaMalloc(&d_packed_2bit, M * num_groups * sizeof(uint32_t));
    cudaMalloc(&d_packed_4bit, M * num_groups * 2 * sizeof(uint32_t));
    cudaMalloc(&d_scales_1st, M * num_groups * sizeof(half));
    cudaMalloc(&d_zeros_1st, M * num_groups * sizeof(int8_t));
    cudaMalloc(&d_outlier_mask, M * K * sizeof(int));
    
    cudaMemcpy(d_weights, h_weights.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_outlier_mask, 0, M * K * sizeof(int));
    
    // Set all groups to 2-bit precision
    std::vector<uint8_t> h_precision(num_groups, 0);
    cudaMemcpy(d_group_precision, h_precision.data(), num_groups * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Step 1: Compute group statistics
    int block_size = 256;
    int grid_size = (num_groups + block_size - 1) / block_size;
    memboost::compute_group_stats_kernel<<<grid_size, block_size>>>(
        d_weights, d_group_min, d_group_max, M, K, num_groups
    );
    cudaDeviceSynchronize();
    
    // Step 2: Quantize weights
    dim3 quant_grid((num_groups + 31) / 32, M);
    dim3 quant_block(32);
    memboost::quantize_weights_kernel<<<quant_grid, quant_block>>>(
        d_weights, d_group_min, d_group_max, d_group_precision,
        d_packed_2bit, d_packed_4bit, d_scales_1st, d_zeros_1st,
        d_outlier_mask, M, K, num_groups
    );
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA Error: %s\n", cudaGetErrorString(err));
        printf("  Result: FAILED\n\n");
        return;
    }
    
    // Count outliers
    std::vector<int> h_outlier_mask(M * K);
    cudaMemcpy(h_outlier_mask.data(), d_outlier_mask, M * K * sizeof(int), cudaMemcpyDeviceToHost);
    int outlier_count = 0;
    for (int i = 0; i < M * K; i++) {
        if (h_outlier_mask[i]) outlier_count++;
    }
    
    printf("  Outliers detected: %d (%.2f%%)\n", outlier_count, 100.0f * outlier_count / (M * K));
    printf("  Result: PASSED\n\n");
    
    // Cleanup
    cudaFree(d_weights);
    cudaFree(d_group_min);
    cudaFree(d_group_max);
    cudaFree(d_group_precision);
    cudaFree(d_packed_2bit);
    cudaFree(d_packed_4bit);
    cudaFree(d_scales_1st);
    cudaFree(d_zeros_1st);
    cudaFree(d_outlier_mask);
}

int main() {
    printf("Testing the 2-bit Quantization Cuda Implementation\n\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    test_pack_unpack();
    test_4bit_pack_unpack();
    test_group_stats_kernel();
    test_full_quantization_pipeline();

    printf("\nDone\n");
    return 0;
}

#endif