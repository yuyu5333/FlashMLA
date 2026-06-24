/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#pragma once

#include "cutlass/barrier.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
    SReady = 1,
    SoftmaxReady = 2,
    TransVReady = 3,
    // [M3.c.4] Producer-internal sync for packed-FP8 fused dequant.
    // Used to synchronize the 128 producer-side threads (warp group 1)
    // across the bit-unpack -> affine -> R@x -> FP8 pipeline stages.
    // MUST NOT use __syncthreads() in this region: the consumer warp
    // group (128 threads in tidx<kNThreadsS=128) does not enter this
    // branch and would deadlock.
    PackedKvProducer = 4,
};

} // flash