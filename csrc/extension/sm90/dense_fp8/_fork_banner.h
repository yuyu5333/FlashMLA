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
// 20260624 = M3.c.4 stage-2 INSERTION CONTRACT: 在 flash_fwd_mla_kernel.h
//             warp group 1 KV-load 路径 (else branch, tidx >= kNThreadsS)
//             gK 构造之前插入 ~70 行 contract 注释，明确 Stage-2 device-side
//             实现的输入/输出/不变量。**此 commit 零运行时变化**: kernel
//             逻辑未改，所有 dense_fp8 / packed-stage-1 调用仍走原路。
//             下一刀 = 真实现 Stage-2 fused dequant (INT-N unpack + R@x +
//             ×scale+zero -> FP8 -> sK)，那时 banner 推到 20260625+。
inline constexpr int64_t kForkBanner = 20260624LL;

inline int64_t fork_banner() { return kForkBanner; }

}  // namespace flashmla_fork
