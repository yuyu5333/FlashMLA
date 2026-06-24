// FlashMLA fork (yuyu5333/FlashMLA, branch kv2bit-dev) link-time probe.
//
// 用途：FlashMLA fork 与 sgl-kernel build link 联通性的探针 header。
// 用一个 ABI 稳定的 int64 常量 + 一个 host-only 函数返回该常量，
// 给 sgl-kernel 注册成 torch op `flashmla_fork_banner`，
// import 测试可以确认：
//   (a) cmake FetchContent 真的从 yuyu5333/FlashMLA 拉到了带本 header 的 commit；
//   (b) 容器 rebuild 后 sgl-kernel/_C.so 重新 link，可以加载到 fork 端新增 op；
//   (c) 后续 packed-entry / fused-dequant 改造的 dev loop（fork commit -> push ->
//       cmake bump GIT_TAG -> 容器 rebuild -> 新 op 可用）是端到端通的。
//
// 这是一个零数值风险、零 kernel 改动的链路连通性 commit。
// 之后做 inner-loop fused dequant 时，第一阶段也会复用同样的注册路径。

#pragma once

#include <cstdint>

namespace flashmla_fork {

// bump 这个常量来探针每次 fork-side 改动是否被容器端真实重 build。
// 数值约定：YYYYMMDD（fork-side 首个 banner commit 日期）。
// 20260622 = stage-0 scaffold (nullptr fallback only).
// 20260623 = M3.c.4 stage-1 wiring: 4 packed tensors真实写入 DecodingParams_fp8，
//             kernel 端尚未读取（params struct 加新字段默认 nullptr/0）。
//             All-4 non-None 时 bit-exact == dense_fp8。
// 20260625 = M3.c.4 S2-S2: packed path real fused dequant in warp group 1.
//             - bit-unpack (per-channel variable-width, via dim_of_bit/bitpos_in_dim)
//             - affine dequant (x = codes * scale + zero_point)
//             - R @ x rotation (512x512 matrix-vector multiply per token)
//             - FP8 e4m3 convert
//             - rope BF16 -> FP8 direct copy
//             - result written to sK via dense smem staging buffer
//             Dense path unchanged (bit-exact vs stage-1).
// 20260626 = M3.c.4 S2-S2 fix: s_codes/s_x smem buffers 448 -> 512 (qk_nope).
//             Bug discovered during dense fallback verification: the fp32
//             code/x-stage smem arrays were declared at 448 elements but
//             FlashMLA k_head_size=576, rope=64, so nope=512. With qk_nope=512
//             (set by host wrapper from kv-cache shape), the prior arrays
//             would have produced out-of-bounds atomicOr writes for d>=448.
//             Dense path still bit-exact unchanged.
// 20260627 = M3.c.4 S2-S2 deadlock fix: replace __syncthreads() with
//             cutlass NamedBarrier(128, PackedKvProducer=4) inside the
//             packed-FP8 producer-only KV-load branch. The prior code used
//             __syncthreads() which deadlocks because this kernel is warp-
//             specialized: the consumer warp group (tidx<128) does not enter
//             the producer KV-load branch and would never reach the barrier.
//             Symptom: packed kernel hung with GPU 0% utilization.
//             Dense path still bit-exact unchanged.
inline constexpr int64_t kForkBanner = 20260627LL;

inline int64_t fork_banner() { return kForkBanner; }

}  // namespace flashmla_fork
