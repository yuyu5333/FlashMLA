/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Keep a self-contained fp8 decode params struct so this extension can stay
// compatible when upstream refactors csrc/params.h.
struct DecodingParams_fp8 {
    using index_t = int64_t;

    int b;              // batch size
    int s_q;
    int q_seq_per_hk;   // Number of q(s) per KV head
    int d, d_v;         // K/V dimension
    int h_q, h_k;       // Number of Q/K heads
    int num_blocks;     // Number of blocks in total
    int q_head_per_hk;  // Number of q heads per KV head
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int topk;

    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ o_ptr;
    void* __restrict__ softmax_lse_ptr;
    int* __restrict__ indices_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;
    index_t indices_batch_stride;
    index_t indices_row_stride;

    int* __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int* __restrict__ seqlens_k_ptr;

    int* __restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int* __restrict__ num_splits_ptr;

    int total_num_splits;
    void* __restrict__ softmax_lseaccum_ptr;
    void* __restrict__ oaccum_ptr;

    int h_h_k_ratio;
    float* __restrict__ descale_q_ptr = nullptr;
    float* __restrict__ descale_k_ptr = nullptr;

    // ------------------------------------------------------------------
    // [M3.c.4 Stage-1 wiring] packed-FP8 rotated-quant KV cache pointers.
    //
    // These four pointers + meta are the **device-side handles** to the
    // rotated low-precision KV cache (INT2/3/4 affine quant after a
    // dense orthogonal rotation). They are written by the host entry
    // `fwd_kvcache_mla_packed_fp8` (see dense_fp8_packed_entry.cpp) when
    // all four caller tensors are non-None. The current kernel body
    // (run_mha_fwd_splitkv_mla) **does not yet read these fields**; the
    // next fork commit will fuse INT-N unpack + R @ x + ×scale + zero
    // -> FP8 inside the KV-load inner loop, removing the host-side
    // shadow buffer entirely. Default value is nullptr / 0 so the
    // pre-existing dense_fp8 path is byte-identical to before.
    //
    // Layout convention (matches python/sglang/srt/mem_cache/
    // rotated_quant_dsv4_memory_pool.py wall-storage layout):
    //   * packed_kcache : uint8 [num_pages * page_size, row_bytes_nope]
    //                     (only the nope half; rope half kept BF16 inside
    //                      the same row, contiguous after nope bytes)
    //   * scale_kcache  : float32 [num_pages * page_size, qk_nope_head_dim]
    //                     per-element dequant scale (broadcast against
    //                     unpacked INT-N value)
    //   * R_matrix      : float32 [qk_nope_head_dim, qk_nope_head_dim]
    //                     dense orthogonal rotation applied AFTER unpack
    //                     + affine (i.e. dequant := R.t() @ (q * scale +
    //                     zero) under the calibration convention)
    //   * zero_point    : float32 [qk_nope_head_dim] per-element zero
    //
    // packed_row_bytes is the byte-stride of one (page,slot) row inside
    // packed_kcache; packed_k_batch_stride is the page-stride (i.e.
    // bytes per page, may include alignment padding).
    //
    // dim_of_bit / bitpos_in_dim are the per-config bit-packing metadata
    // arrays (length = row_bits = sum(bits[d] for d in 0..qk_nope-1)).
    // They are per-layer constants (same for every token) so they are
    // passed as dense pointers (not block_table indexed).
    //   dim_of_bit[i]     : which channel dim the i-th bit belongs to
    //   bitpos_in_dim[i]  : which bit position (0..bits[d]-1) within that dim
    // ------------------------------------------------------------------
    void*  __restrict__ packed_kcache_ptr     = nullptr;
    float* __restrict__ scale_kcache_ptr      = nullptr;
    float* __restrict__ R_matrix_ptr          = nullptr;
    float* __restrict__ zero_point_ptr        = nullptr;
    int*   __restrict__ dim_of_bit_ptr        = nullptr;
    int*   __restrict__ bitpos_in_dim_ptr     = nullptr;
    index_t             packed_k_batch_stride = 0;
    int                 packed_row_bytes      = 0;
    int                 qk_nope_head_dim      = 0;
    int                 row_bits              = 0;
};

static constexpr int TileSchedulerMetaDataSize = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(DecodingParams_fp8 &params, cudaStream_t stream);

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
