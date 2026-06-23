// FlashMLA fork (yuyu5333/FlashMLA, branch kv2bit-dev) — packed_fp8 entry.
//
// ============================================================================
// [M3.c.4 STAGE-1 WIRING ONLY]
// ============================================================================
//
// 目的：本 TU 是 fork 内 packed-FP8 (rotated low-precision KV cache) 端到端
// 通路的 **device-side wiring entry**。
//
// Stage-1 行为约定（本 commit 落地）：
//
//   1. 接收来自 Python 端的 4 个 packed tensor（packed_kcache / scale_kcache
//      / R_matrix / zero_point）。允许全部为 None，也允许全部为非 None；
//      混合（部分 None 部分非 None）一律 TORCH_CHECK 报错——避免 caller
//      误传半套 calib。
//
//   2. 把 4 个 packed tensor 的 device pointer + 必要 meta（packed_row_bytes
//      / packed_k_batch_stride / qk_nope_head_dim）真实写入
//      DecodingParams_fp8 的新字段。这一步打通 PyTorch tensor -> device
//      params 全链路，下一刀 inner-loop 改造时 kernel 端可以直接读。
//
//   3. **Kernel body 不读 packed 字段**：本 commit 仍然调用同一个
//      `run_mha_fwd_splitkv_mla<float_e4m3_t, bfloat16_t, 576>(params, stream)`
//      kernel；params struct 加新字段默认 nullptr/0，不影响旧逻辑。Caller
//      仍需把 dense FP8 shadow buffer 作为 kcache 传进来，packed tensor 此
//      时只是 "为 kernel 准备好但未读" 的 wiring。Stage-1 验收条件：4
//      tensor 全非 None 时，输出与 dense_fp8 kernel **bit-exact** 相等。
//
//   4. Stage-2（下一刀）：把 kernel 端 KV-load 路径替换为
//      packed unpack -> R @ x -> ×scale+zero -> FP8 -> MMA，shadow buffer
//      可删除；本 wiring 不需要再改。
//
// 不变量：
//   * descale_q / descale_k 仍按 dense_fp8 路径要求传（Stage-1 走 dense kernel
//     bit-exact，descale 不能为 None）。
//   * 任一 packed buffer 缺失（全 None）-> 直通 dense_fp8 路径，与 fork 端
//     fwd_kvcache_mla_fp8 byte-identical。
//   * 任一 packed buffer 部分缺失 -> TORCH_CHECK 立刻报错。
//
// dev loop 同 probe TU 范式：fork 改 -> push -> sgl-kernel cmake bump GIT_TAG
// -> 容器 rebuild -> torch op `sgl_kernel.fwd_kvcache_mla_packed_fp8` 接受 4
// 非 None tensor 不再报错，回包与 dense_fp8 bit-exact。
// ============================================================================

#include <torch/all.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <vector>

#include "_fork_banner.h"
#include "flash_mla.h"

#define PFP8_CHECK_DEVICE(x) TORCH_CHECK((x).is_cuda(), #x " must be on CUDA")
#define PFP8_CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// fwd_kvcache_mla_fp8 在同一编译单元集合（flashmla_ops MODULE）里的
// dense_fp8_python_api.cpp 中定义；这里 forward-declare 它，避免 fork
// 端为了头文件而到处加 include。
std::vector<at::Tensor>
fwd_kvcache_mla_fp8(
    at::Tensor &q,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k
);

#ifndef FLASH_MLA_DISABLE_FP8
template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(DecodingParams_fp8 &params, cudaStream_t stream);
#endif

namespace {

// Shape-/dtype-validate the four packed buffers and pull out the
// qk_nope_head_dim implied by R_matrix. All four are expected to be
// CUDA-resident, contiguous along the last dim. Stage-1 keeps these
// checks intentionally light: heavier numerical validation lives in the
// Python-side test_rotated_kv_quant_dsv4_canary.
void validate_packed_buffers(
    const at::Tensor &packed_kcache,
    const at::Tensor &scale_kcache,
    const at::Tensor &R_matrix,
    const at::Tensor &zero_point,
    int kv_num_rows,
    int *out_qk_nope_head_dim,
    int *out_packed_row_bytes
) {
    PFP8_CHECK_DEVICE(packed_kcache);
    PFP8_CHECK_DEVICE(scale_kcache);
    PFP8_CHECK_DEVICE(R_matrix);
    PFP8_CHECK_DEVICE(zero_point);

    TORCH_CHECK(packed_kcache.dtype() == at::kByte,
        "packed_kcache must be uint8");
    TORCH_CHECK(scale_kcache.dtype() == at::kFloat,
        "scale_kcache must be float32");
    TORCH_CHECK(R_matrix.dtype() == at::kFloat,
        "R_matrix must be float32");
    TORCH_CHECK(zero_point.dtype() == at::kFloat,
        "zero_point must be float32");

    TORCH_CHECK(packed_kcache.stride(-1) == 1,
        "packed_kcache must have contiguous last dim");
    PFP8_CHECK_CONTIGUOUS(scale_kcache);
    PFP8_CHECK_CONTIGUOUS(R_matrix);
    PFP8_CHECK_CONTIGUOUS(zero_point);

    TORCH_CHECK(R_matrix.dim() == 2,
        "R_matrix must be rank-2, got ", R_matrix.dim());
    const auto R0 = R_matrix.size(0);
    const auto R1 = R_matrix.size(1);
    TORCH_CHECK(R0 == R1,
        "R_matrix must be square, got [", R0, ", ", R1, "]");
    const int qk_nope = static_cast<int>(R0);
    TORCH_CHECK(qk_nope > 0 && qk_nope % 32 == 0,
        "qk_nope_head_dim must be positive multiple of 32, got ", qk_nope);

    TORCH_CHECK(zero_point.dim() == 1 && zero_point.size(0) == qk_nope,
        "zero_point must be [qk_nope_head_dim], got [",
        zero_point.sizes(), "]");

    // packed_kcache layout: [N, row_bytes_nope]. row_bytes_nope is at
    // most ceil(qk_nope * 8 / 8) = qk_nope when every dim uses 8-bit
    // calibration, and as low as ceil(qk_nope * 1 / 8) for INT1.
    TORCH_CHECK(packed_kcache.dim() == 2,
        "packed_kcache must be rank-2 [num_rows, row_bytes], got ",
        packed_kcache.dim());
    const auto pk_rows = packed_kcache.size(0);
    const auto pk_cols = packed_kcache.size(1);
    TORCH_CHECK(pk_rows == kv_num_rows,
        "packed_kcache row count ", pk_rows, " must equal kv num_rows ",
        kv_num_rows, " (= num_blocks * page_block_size)");
    TORCH_CHECK(pk_cols > 0 && pk_cols <= qk_nope,
        "packed_kcache row_bytes ", pk_cols,
        " must be in (0, qk_nope_head_dim=", qk_nope, "]");

    // scale_kcache layout: [N, qk_nope]. Each row holds a per-element
    // dequant scale (we do not yet support per-group scales).
    TORCH_CHECK(scale_kcache.dim() == 2,
        "scale_kcache must be rank-2 [num_rows, qk_nope], got ",
        scale_kcache.dim());
    TORCH_CHECK(scale_kcache.size(0) == kv_num_rows,
        "scale_kcache row count ", scale_kcache.size(0),
        " must equal kv num_rows ", kv_num_rows);
    TORCH_CHECK(scale_kcache.size(1) == qk_nope,
        "scale_kcache col count ", scale_kcache.size(1),
        " must equal qk_nope_head_dim ", qk_nope);

    *out_qk_nope_head_dim = qk_nope;
    *out_packed_row_bytes = static_cast<int>(pk_cols);
}

}  // namespace

std::vector<at::Tensor>
fwd_kvcache_mla_packed_fp8(
    at::Tensor &q,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k,
    // ---- packed_fp8 device-side wiring ----
    const std::optional<at::Tensor> &packed_kcache,
    const std::optional<at::Tensor> &scale_kcache,
    const std::optional<at::Tensor> &R_matrix,
    const std::optional<at::Tensor> &zero_point,
    const std::optional<at::Tensor> &dim_of_bit,
    const std::optional<at::Tensor> &bitpos_in_dim
) {
    // -----------------------------------------------------------------
    // Mode selection.
    //   * all-4 None     -> dense_fp8 fallback (bit-exact, fork pre-stage-1
    //                       behavior; nothing else this TU does).
    //   * all-4 non-None -> stage-1 packed-wiring path (this TU continues
    //                       below: shape checks + params wiring + dense
    //                       kernel launch; output remains bit-exact vs
    //                       dense_fp8 because kernel currently ignores the
    //                       packed fields).
    //   * mixed          -> hard fail (caller passed a half calib).
    // -----------------------------------------------------------------
    const int num_packed_present =
        (packed_kcache.has_value() ? 1 : 0) +
        (scale_kcache.has_value()  ? 1 : 0) +
        (R_matrix.has_value()      ? 1 : 0) +
        (zero_point.has_value()    ? 1 : 0) +
        (dim_of_bit.has_value()    ? 1 : 0) +
        (bitpos_in_dim.has_value() ? 1 : 0);

    if (num_packed_present == 0) {
        // Bit-exact dense_fp8 fallback. This matches the pre-stage-1
        // scaffold behaviour exactly and is the path the unit-test
        // canary asserts byte-equality against.
        return fwd_kvcache_mla_fp8(
            q, kcache, head_size_v, seqlens_k, block_table,
            softmax_scale, is_causal,
            tile_scheduler_metadata, num_splits,
            descale_q, descale_k);
    }

    TORCH_CHECK(num_packed_present == 6,
        "fwd_kvcache_mla_packed_fp8: packed-FP8 path requires either all "
        "six of (packed_kcache, scale_kcache, R_matrix, zero_point, "
        "dim_of_bit, bitpos_in_dim) to be None or all six non-None. "
        "Got non-None count=",
        num_packed_present,
        ". Banner=", flashmla_fork::kForkBanner);

#ifdef FLASH_MLA_DISABLE_FP8
    TORCH_CHECK(false,
        "FlashMLA is compiled with -DFLASH_MLA_DISABLE_FP8. Please remove "
        "this flag from your environment and re-compile FlashMLA.");
#else

    // -----------------------------------------------------------------
    // The dense_fp8 host entry's set-up logic is duplicated here in
    // full so we can install the four packed pointers into the params
    // struct without touching the upstream-shaped fwd_kvcache_mla_fp8.
    // Anything that changes upstream should be mirrored here in the
    // same shape; the diff is intentionally surgical (just the new
    // packed_* params fields).
    // -----------------------------------------------------------------
    int head_size_v_int = static_cast<int>(head_size_v);

    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 9 && dprops->minor == 0,
        "Dense/Packed FP8 MLA is only supported on SM90");

    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(kcache.dtype() == q.dtype(),
        "query and key must have the same dtype");
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32,
        "seqlens_k must have dtype int32");
    TORCH_CHECK(block_table.dtype() == torch::kInt32,
        "block_table must have dtype torch.int32");
    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32,
        "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(num_splits.dtype() == torch::kInt32,
        "num_splits must have dtype int32");

    PFP8_CHECK_DEVICE(q);
    PFP8_CHECK_DEVICE(kcache);
    PFP8_CHECK_DEVICE(seqlens_k);
    PFP8_CHECK_DEVICE(block_table);
    PFP8_CHECK_DEVICE(tile_scheduler_metadata);
    PFP8_CHECK_DEVICE(num_splits);
    if (descale_q.has_value()) PFP8_CHECK_DEVICE(descale_q.value());
    if (descale_k.has_value()) PFP8_CHECK_DEVICE(descale_k.value());

    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1,
        "kcache must have contiguous last dimension");
    PFP8_CHECK_CONTIGUOUS(seqlens_k);
    TORCH_CHECK(block_table.stride(-1) == 1,
        "block_table must have contiguous last dimension");
    PFP8_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    PFP8_CHECK_CONTIGUOUS(num_splits);

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_q = sizes[2];
    const int head_size_k = sizes[3];
    TORCH_CHECK(head_size_k == 576, "Only head_size_k == 576 is supported");
    TORCH_CHECK(head_size_v_int == 512,
        "Only head_size_v == 512 is supported");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(page_block_size == 64,
        "Currently page_block_size must be 64");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_k == 0,
        "Number of heads in key/value must divide number of heads in query");

    TORCH_CHECK(descale_q.has_value() && descale_k.has_value(),
        "descale is required when input dtype is fp8");
    auto descale_q_ = descale_q.value();
    auto descale_k_ = descale_k.value();
    PFP8_CHECK_DEVICE(descale_q_);
    PFP8_CHECK_DEVICE(descale_k_);
    TORCH_CHECK(descale_q_.stride(-1) == 1);
    TORCH_CHECK(descale_k_.stride(-1) == 1);
    TORCH_CHECK(descale_q_.dtype() == torch::kFloat);
    TORCH_CHECK(descale_k_.dtype() == torch::kFloat);

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int num_q_heads_per_hk = num_heads_q / num_heads_k;
    const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;
    const int num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk, head_size_k}).transpose(2, 3)
            .reshape({batch_size, q_seq_per_hk, num_heads, head_size_k});

    // ---- packed buffer validation (must match kcache row count) ----
    const int kv_num_rows = num_blocks * page_block_size;
    int qk_nope_head_dim = 0;
    int packed_row_bytes = 0;
    validate_packed_buffers(
        packed_kcache.value(),
        scale_kcache.value(),
        R_matrix.value(),
        zero_point.value(),
        kv_num_rows,
        &qk_nope_head_dim,
        &packed_row_bytes);
    TORCH_CHECK(qk_nope_head_dim + 64 == head_size_k,
        "qk_nope_head_dim ", qk_nope_head_dim,
        " + qk_rope_head_dim 64 must equal head_size_k ", head_size_k,
        " (DSv4 layout)");

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    at::Tensor out = torch::empty(
        {batch_size, q_seq_per_hk, num_heads, head_size_v_int},
        opts.dtype(at::kBFloat16));
    at::Tensor softmax_lse = torch::empty(
        {batch_size, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    PFP8_CHECK_CONTIGUOUS(softmax_lse);

    DecodingParams_fp8 params = {};
    params.b = batch_size;
    params.s_q = seqlen_q_ori;
    params.q_seq_per_hk = q_seq_per_hk;
    params.seqlens_k_ptr = seqlens_k.data_ptr<int>();
    params.h_q = num_heads_q;
    params.h_k = num_heads_k;
    params.num_blocks = num_blocks;
    params.q_head_per_hk = num_q_heads_per_hk;
    params.is_causal = is_causal;
    params.d = head_size_k;
    params.d_v = head_size_v_int;
    params.scale_softmax = static_cast<float>(softmax_scale);
    params.scale_softmax_log2 =
        float(static_cast<float>(softmax_scale) * M_LOG2E);
    params.topk = -1;
    params.h_h_k_ratio = 1;
    params.descale_q_ptr = reinterpret_cast<float *>(descale_q_.data_ptr());
    params.descale_k_ptr = reinterpret_cast<float *>(descale_k_.data_ptr());

    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.indices_ptr = nullptr;
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(1);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(2);
    params.o_head_stride = out.stride(-2);
    params.indices_batch_stride = 0;
    params.indices_row_stride = 0;

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;

    params.tile_scheduler_metadata_ptr =
        tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    params.num_splits_ptr = num_splits.data_ptr<int>();

    const int total_num_splits = batch_size + params.num_sm_parts;
    at::Tensor softmax_lse_accum = torch::empty(
        {total_num_splits, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    at::Tensor out_accum = torch::empty(
        {total_num_splits, num_heads, q_seq_per_hk, head_size_v_int},
        opts.dtype(at::kFloat));
    PFP8_CHECK_CONTIGUOUS(softmax_lse_accum);
    PFP8_CHECK_CONTIGUOUS(out_accum);
    params.total_num_splits = total_num_splits;
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();

    // ---- [M3.c.4 S2-S2] install packed pointers + pack meta into params ----
    // S2-S2 kernel consumes these for fused bit-unpack + affine dequant
    // + R@x + FP8 convert.
    {
        const at::Tensor &pk = packed_kcache.value();
        const at::Tensor &sk = scale_kcache.value();
        const at::Tensor &Rm = R_matrix.value();
        const at::Tensor &zp = zero_point.value();
        const at::Tensor &dob = dim_of_bit.value();
        const at::Tensor &bpd = bitpos_in_dim.value();

        // Validate dim_of_bit / bitpos_in_dim
        PFP8_CHECK_DEVICE(dob);
        PFP8_CHECK_DEVICE(bpd);
        PFP8_CHECK_CONTIGUOUS(dob);
        PFP8_CHECK_CONTIGUOUS(bpd);
        TORCH_CHECK(dob.dtype() == at::kInt, "dim_of_bit must be int32");
        TORCH_CHECK(bpd.dtype() == at::kInt, "bitpos_in_dim must be int32");
        TORCH_CHECK(dob.dim() == 1 && bpd.dim() == 1,
            "dim_of_bit and bitpos_in_dim must be rank-1");
        TORCH_CHECK(dob.size(0) == bpd.size(0),
            "dim_of_bit and bitpos_in_dim must have same length");
        const int row_bits_val = static_cast<int>(dob.size(0));
        TORCH_CHECK(row_bits_val > 0, "row_bits must be positive");

        params.packed_kcache_ptr = pk.data_ptr();
        params.scale_kcache_ptr =
            reinterpret_cast<float *>(sk.data_ptr());
        params.R_matrix_ptr =
            reinterpret_cast<float *>(Rm.data_ptr());
        params.zero_point_ptr =
            reinterpret_cast<float *>(zp.data_ptr());
        params.dim_of_bit_ptr =
            reinterpret_cast<int *>(dob.data_ptr());
        params.bitpos_in_dim_ptr =
            reinterpret_cast<int *>(bpd.data_ptr());
        // packed_kcache is [N, row_bytes]; page-stride in bytes is
        // page_block_size * row_bytes (rows of the same page are
        // contiguous in row-major layout by construction).
        params.packed_row_bytes = packed_row_bytes;
        params.packed_k_batch_stride =
            static_cast<DecodingParams_fp8::index_t>(page_block_size) *
            static_cast<DecodingParams_fp8::index_t>(packed_row_bytes);
        params.qk_nope_head_dim = qk_nope_head_dim;
        params.row_bits = row_bits_val;
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(
        params, stream);

    out = out.view({batch_size, seqlen_q_ori, num_q_heads_per_hk, num_heads_k, head_size_v_int}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_q, head_size_v_int});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk}).transpose(2, 3)
            .reshape({batch_size, num_heads_q, seqlen_q_ori});

    return {out, softmax_lse};
#endif  // FLASH_MLA_DISABLE_FP8
}
