// FlashMLA fork (yuyu5333/FlashMLA, branch kv2bit-dev) link-time probe TU.
//
// 用途：作为 _fork_banner.h 的唯一翻译单元，被 sgl-kernel cmake build 链接到
// `flashmla_ops` 模块。sgl-kernel 注册 torch op `flashmla_fork_probe`
// 后，Python 端 `torch.ops.sgl_kernel.flashmla_fork_probe.default()` 调到
// 这里返回 _fork_banner.h 中的 constexpr int64 常量。
//
// 任何后续 fork 侧 kernel 改动（packed_fp8 / fused dequant inner-loop ...）都
// 复用同一条链路：
//   fork 改 -> push -> cmake bump GIT_TAG -> 容器 sgl-kernel rebuild ->
//   新 op 可用且数值正确。
//
// 本 TU 不依赖 CUDA、cute、cutlass，仅一个返回 int64 的 host 函数，
// 把 build/link 风险吃干净；任何数值 kernel 改造从下一刀开始。

#include <cstdint>

#include "_fork_banner.h"

// 注意：这里**不**放进 flashmla_fork 命名空间，便于 sgl-kernel 端通过
// 全局符号 `flashmla_fork_probe` 直接声明与调用（避免 cross-DSO C++ name
// mangling 的轻微歧义）。常量值仍来自 namespace 内 constexpr。
extern "C++" int64_t flashmla_fork_probe() {
    return flashmla_fork::kForkBanner;
}
