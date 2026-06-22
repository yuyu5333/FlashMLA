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
inline constexpr int64_t kForkBanner = 20260622LL;

inline int64_t fork_banner() { return kForkBanner; }

}  // namespace flashmla_fork
