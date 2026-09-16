// Standalone exhaustive GPU check of the optimized codec against the PR3 codec.
// nvcc -O3 -std=c++20 -arch=sm_90a -Icsrc -Icsrc/cutlass/include \
//   benchmark/validate_sm90_e2m1.cu -o /workspace/validate_sm90_e2m1
#include <cstdio>
#include "kernels/sm90/decode/sparse/components/dequant.h"

using namespace sm90::decode::sparse;

__global__ void validate(unsigned int* mismatches) {
    // Four nibbles cover every pair/byte position, including negative zero.
    // Repeat them in both halves and cover every finite non-negative scale.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536u * 127u) return;
    const unsigned int codes = i & 0xffffu;
    const unsigned int packed = codes | (codes << 16);
    __nv_fp8_e4m3 scale_fp8;
    scale_fp8.__x = i >> 16;
    const __nv_bfloat162 scale = __bfloat162bfloat162(
        __float2bfloat16(static_cast<float>(scale_fp8))
    );
    const bf16x8 got = cvt_e2m1x8_bf16x8(packed, scale);
    const uint32_t* got_words = reinterpret_cast<const uint32_t*>(&got);
    bool mismatch = false;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const uint32_t bits = e2m1_to_bf16_bits((packed >> (8 * pair)) & 0xfu) |
            (uint32_t(e2m1_to_bf16_bits((packed >> (8 * pair + 4)) & 0xfu)) << 16);
        const __nv_bfloat162 expected = __hmul2(
            *reinterpret_cast<const __nv_bfloat162*>(&bits), scale
        );
        mismatch |= got_words[pair] != *reinterpret_cast<const uint32_t*>(&expected);
    }
    if (mismatch) atomicAdd(mismatches, 1u);
}

int main() {
    unsigned int* mismatches = nullptr;
    if (cudaMallocManaged(&mismatches, sizeof(*mismatches)) != cudaSuccess) return 2;
    *mismatches = 0;
    validate<<<(65536u * 127u + 255) / 256, 256>>>(mismatches);
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        cudaFree(mismatches);
        return 2;
    }
    std::printf("E2M1 GPU exhaustive check: %u patterns, %u mismatches\n",
                65536u * 127u, *mismatches);
    const int result = *mismatches == 0 ? 0 : 1;
    cudaFree(mismatches);
    return result;
}
