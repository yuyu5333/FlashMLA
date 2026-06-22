// FlashMLA fork (yuyu5333/FlashMLA, branch kv2bit-dev) — packed_fp8 entry scaffold.
//
// 目的：作为 fork 内 packed-FP8 (low-precision rotated KV cache) inner-loop
// kernel 改造的稳定 entry。当前阶段（M3.c.4 stage-1）**不引入任何新数学**，
// 只把后续 fused-dequant kernel 需要的额外 4 个 buffer pointer 占位下来：
//   - packed_kcache   : uint8 packed KV bytes (after rotated affine quant)
//   - scale_kcache    : per-row dequant scale
//   - R_matrix        : random rotation matrix used by quant calibration
//   - zero_point      : per-row dequant zero offset
//
// 行为约定：
//   * 任一占位 tensor 为空（or .has_value()==false）→ fallback 直通已验证
//     的 dense_fp8 kernel `fwd_kvcache_mla_fp8`，保证 nullptr fallback 与
//     dense_fp8 路径 **bit-exact**（同一函数同一 stream）。
//   * 全部 4 个 packed buffer 同时提供 → 暂时 TORCH_CHECK 报错
//     "packed_fp8 fused-dequant path not yet implemented"，留给下一刀
//     在 flash_fwd_mla_kernel.h 内做真正的 inner-loop 替换。
//
// dev loop 同 probe TU 范式：
//   fork 改 -> push -> sgl-kernel cmake bump GIT_TAG -> 容器 rebuild ->
//   torch op `sgl_kernel.fwd_kvcache_mla_packed_fp8` 可用，且 nullptr
//   fallback 与 dense_fp8 bit-exact。

#include <torch/all.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <vector>

#include "_fork_banner.h"

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
    // ---- packed_fp8 占位字段 ----
    const std::optional<at::Tensor> &packed_kcache,
    const std::optional<at::Tensor> &scale_kcache,
    const std::optional<at::Tensor> &R_matrix,
    const std::optional<at::Tensor> &zero_point
) {
    // 当前 stage：四个占位 tensor **必须全部为 None**，走 dense_fp8
    // bit-exact fallback；任一非 None 视为下一刀 fused-dequant kernel
    // 的入口尚未连通，立即报错避免误用。
    const bool any_packed_buffer_present =
        packed_kcache.has_value() ||
        scale_kcache.has_value() ||
        R_matrix.has_value() ||
        zero_point.has_value();
    TORCH_CHECK(!any_packed_buffer_present,
        "fwd_kvcache_mla_packed_fp8: packed-FP8 fused-dequant inner-loop is "
        "not yet implemented in this fork commit; pass all of "
        "packed_kcache/scale_kcache/R_matrix/zero_point as None to use the "
        "dense_fp8 bit-exact fallback. Banner=",
        flashmla_fork::kForkBanner);

    return fwd_kvcache_mla_fp8(
        q, kcache, head_size_v, seqlens_k, block_table,
        softmax_scale, is_causal,
        tile_scheduler_metadata, num_splits,
        descale_q, descale_k);
}
