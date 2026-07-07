#pragma once

// [Route H step2a] fold-rotation PERF PROBE toggle.
//   When defined, the packed producer skips the per-K-block R@X wgmma
//   reconstruction and writes raw unpacked X straight to sK nope. Output
//   is intentionally salad; this only measures the full-load decode tps
//   ceiling to confirm the reconstruction chain is the 1-block/SM
//   latency-bound bottleneck. Comment out for the byte-correct step1 path.
//
// [Route H step2a RESULT] Probe measured full-load 32-req decode tps = 19.54,
//   IDENTICAL to step1 byte-correct (19.53). Skipping the entire producer R@X
//   wgmma reconstruction gave ZERO tps gain -> the "per-K-block R@X rebuild is
//   the main bottleneck" hypothesis is DISPROVEN. Bottleneck is NOT producer
//   compute (consumer WG QK/PV chain + 1-block/SM low-occupancy + barrier sync
//   dominate). Probe DISABLED; default is the byte-correct step1 #else path.
//   Keep the guarded probe branch for future A/B comparison.
// #define FMLA_FOLD_ROT_PROBE 1

// [Route H step3b] producer NULL-WORK PERF PROBE toggle.
//   When defined, the packed producer skips ALL nope reconstruction
//   (bit-unpack + affine + R@X wgmma/legacy + staging->sK). Only the
//   rope direct-copy and the buffer handshake (bar_k_avail wait /
//   bar_k_local_ready arrive / is_kv_valid write) survive; sK nope stays
//   uninitialized so the output is intentionally salad.
//
//   Rationale (step3 static analysis): step2a only removed the producer
//   wgmma MATH yet kept the byte-unpack + 14 NamedBarriers + staging->sK,
//   and step1-vs-step2a shows the producer GLOBAL-LOAD volume (2464 vs
//   224 loads/thread) also does not move tps. Neither is a controlled
//   variable experiment for the producer's FIXED STRUCTURAL cost (barrier
//   handshake + 7-dim_block serial skeleton + staging). This probe zeroes
//   that entire cost in one cut:
//     tps stays ~19.5 -> producer is NOT the bottleneck at all
//       (consumer WG QK/PV chain or wall+drop_shadow multi-pool schedule);
//     tps jumps       -> the producer's fixed skeleton IS the bottleneck.
//   Occupancy lever is physically ruled out: SmemPlan ~180KB/block, 2
//   blocks need 360KB > 228KB SM90 dyn-smem cap, and __launch_bounds__
//   pins minBlocksPerSM=1. So this bisection is the only remaining cut.
//
// [Route H step3b RESULT] Probe measured full-load 32-req decode tps =
//   19.54/19.54/19.53, IDENTICAL to step2a (19.54) and step1 (19.53).
//   Zeroing the ENTIRE producer nope reconstruction (unpack + 14 barriers
//   + staging->sK + wgmma) gave ZERO tps gain. Combined with step1 (load
//   dedup, no effect) and step2a (wgmma-skip, no effect), the producer WG
//   is FULLY EXONERATED across three independent negative experiments.
//   Bottleneck is definitively NOT the producer. Pivot: consumer WG QK/PV
//   chain (inside flashmla) vs wall+drop_shadow multi-pool schedule
//   (outside flashmla) -- isolated next by a baseline template-A native
//   FP8 canary on the identical workload. Probe DISABLED.
// #define FMLA_PRODUCER_NULL_PROBE 1

// [Route H step3k] in-kernel clock64 SEGMENT PROFILE toggle.
//   When defined, one representative thread per block accumulates clock64()
//   deltas for the packed producer + consumer critical-path segments into a
//   __device__ global counter array. Host run() throttled-prints the mean
//   cycles/block/segment to stderr. Byte-correct (only adds clock64 reads +
//   atomicAdds; the compute path is untouched). Purpose: split the 3.4x
//   use_packed inner-loop (step3j) into producer bar-wait / rope-copy / nope
//   rebuild vs consumer bar-wait / QK+softmax so we know which segment to
//   optimize next. Comment out for the byte-correct production build (adds no
//   counters).
// #define FMLA_CLK_PROFILE 1

#include "splitkv_mla.h"

#include <cstdio>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "flashmla_utils.h"
#include "components/dequant.h"
#include "components/helpers.h"
#include "config.h"
using namespace cute;

namespace sm90::decode::sparse_fp8 {

#ifdef FMLA_CLK_PROFILE
// [Route H step3k] segment cycle counters (device global).
//   Index layout:
//     0: producer bar_k_avail.wait   (empty-wait for buffer free)
//     1: producer rope-copy          (per-token scattered 268B rope read)
//     2: producer nope rebuild       (bit-unpack + affine + R@X wgmma/legacy)
//     3: consumer bar_k_local_ready.wait (empty-wait for producer)
//     4: consumer QK + softmax       (wgmma QK + scale_softmax + save)
//   [5]: number of accumulated (block) samples (only slot used for both WGs;
//        producer counts into 5, consumer into 6, so we can normalize each).
//   Slot 7 unused / padding.
//   NOTE: static (not inline) __device__ -> each instantiation TU gets its
//   own copy. Safe because the kernel and its host run() readback live in the
//   same TU per (MODEL_TYPE, NUM_HEADS) instantiation. inline __device__ is
//   rejected under whole-program mode (-rdc=false).
static __device__ unsigned long long g_fmla_clk[8];

static __forceinline__ __device__ void fmla_clk_add(int seg, unsigned long long dt) {
    // Only one representative lane per warpgroup logs, to avoid 128x inflation.
    atomicAdd(&g_fmla_clk[seg], dt);
}
#endif

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using fp8_e8m0 = __nv_fp8_e8m0;

template<
    bool UpdateO = true,
    typename Tensor0,
    typename Tensor1,
    typename Tensor2
>
__forceinline__ __device__ void scale_softmax(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup
) {
    float scale_for_olds[2];
    CUTE_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));

        float cur_max = -INFINITY;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            if (!is_kv_valid[(i&1)+(i/2)*8+(idx_in_warpgroup%4)*2])
                cur_rP(i) = -INFINITY;
            cur_max = max(cur_max, cur_rP(i));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        cur_max *= scale_softmax_log2;
        float old_max = rM[local_row_idx];
        rM[local_row_idx] = max(cur_max, old_max);
        float scale_for_old = exp2f(old_max - rM[local_row_idx]);
        scale_for_olds[local_row_idx] = scale_for_old;

        if constexpr (UpdateO) {
            Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rO); ++i) {
                cur_rO(i) *= scale_for_old;
            }
        }

        float cur_sum = 0;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            cur_rP(i) = exp2f(cur_rP(i)*scale_softmax_log2 - rM[local_row_idx]);
            cur_rS(i) = (bf16)cur_rP(i);
            cur_sum += cur_rP(i);
        }

        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
    if (idx_in_warpgroup%4 == 0)
        *(float2*)(sScale + 2*(idx_in_warpgroup/4)) = *(float2*)(scale_for_olds);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int idx_in_cluster = CLUSTER_SIZE == 1 ? 0 : head_block_idx % 2;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
    Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;
    
    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
    }
    
    // Initialize TMA barriers
    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_q.init(1);
        if constexpr (CLUSTER_SIZE == 2) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_K_BUFS; ++i) {
                plan.bar_k_local_ready[i].init(128);
                plan.bar_k_remote_ready[i].init(1);
                plan.bar_k_avail[i].init(4);
            }
        } else {
            CUTE_UNROLL
            for (int i = 0; i < NUM_K_BUFS; ++i) {
                plan.bar_k_local_ready[i].init(128);
                plan.bar_k_avail[i].init(256);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    ku::barrier_cluster_arrive_relaxed();

    int bar_phase_k = 0; // Don't use array here to prevent using local memory

    // Programmatic Dependent Launch: Wait for the previous kernel to finish
    // Don't use PDL because of compiler bugs!
    // cudaGridDependencySynchronize();
    
    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

    if (sched_meta.begin_req_idx >= params.b) return;

    if (warp_idx == 0 && elect_one_sync()) {
        Tensor gQ = flat_divide(
            tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
        )(_, _, head_block_idx, _0{});
        launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
        plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
    }

    ku::barrier_cluster_wait_acquire();

    struct MainloopArgs {
        int start_block_idx, end_block_idx;
        bool is_no_split;

        // The following fields are only valid for MODEL1
        int topk_length, extra_topk_length, num_orig_kv_blocks;
    };
    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        int total_topk_padded;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            total_topk_padded = params.topk;
        } else {
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
            args.topk_length = topk_length;
            args.extra_topk_length = extra_topk_length;
            args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
        }

        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        return args;
    };

    if (warpgroup_idx == 0) {
        // [DEBUG] disable reg_alloc to test if setmaxnreg is the illegal instruction
        // cutlass::arch::warpgroup_reg_alloc<192>();

        // [DEBUG] early return before main loop to bisect illegal instruction
        return;

        TiledMMA tiled_mma_QK = TiledMMA_QK{};
        ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
        
        float rL[2], rM[2];
        Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
        Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

        float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
        if (params.attn_sink != nullptr) {
            for (int i = 0; i < 2; ++i) {
                int head_idx = head_block_idx*BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
                rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
            }
        }

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);

            rL[0] = rL[1] = 0.0f;
            rM[0] = rM[1] = MAX_INIT_VAL;
            cute::fill(rO, 0.);

            // Wait for Q
            plan.bar_q.wait((sched_meta.begin_req_idx-batch_idx)&1);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;
                Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});

                // Wait, issue WGMMA
#ifdef FMLA_CLK_PROFILE
                unsigned long long _clk_c0 = clock64();
#endif
                plan.bar_k_local_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                if constexpr (CLUSTER_SIZE == 2) {
                    plan.bar_k_remote_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                }
#ifdef FMLA_CLK_PROFILE
                unsigned long long _clk_c1 = clock64();
                if (idx_in_warpgroup == 0) fmla_clk_add(3, _clk_c1 - _clk_c0);
#endif

                gemm<true, -1>(
                    tiled_mma_QK,
                    thr_mma_QK.partition_fragment_A(sQ),
                    thr_mma_QK.partition_fragment_B(sK),
                    rP
                );

                bar_phase_k ^= 1<<buf_idx;

                // [micro-opt B] Move sScale/sS free-barrier wait into QK wgmma
                // async window (between commit_batch and wait<0>). Barrier
                // arrive+wait is a non-matrix-pipe instruction, safe to issue
                // while QK wgmma is still in flight, so the spin-wait overlaps
                // with tensor-core compute instead of stalling after it.
                if (block_idx != args.start_block_idx)
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_free);

                cute::warpgroup_wait<0>();

                // Since in our case TOPK_BLOCK_SIZE == BLOCK_M, so we only need to do OOB checking for the last 2 blocks
                scale_softmax<false>(rP, rS, rO, params.sm_scale_div_log2, sScale, rM, rL, plan.is_kv_valid[buf_idx], block_idx, idx_in_warpgroup);

                // Store S into shared, inform warpgroup 1
                save_rPb_to_sP(rS, sS, idx_in_warpgroup);
                fence_view_async_shared();
#ifdef FMLA_CLK_PROFILE
                {
                    unsigned long long _clk_c2 = clock64();
                    if (idx_in_warpgroup == 0) {
                        fmla_clk_add(4, _clk_c2 - _clk_c1);
                        fmla_clk_add(6, 1ull);  // consumer block sample count
                    }
                }
#endif

                // Signal WG1 early — sS and sScale are ready, WG1 can start PV immediately
                // while we still have QK results in registers (no PV in WG0 anymore).
                NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);

                if constexpr (CLUSTER_SIZE == 2) {
                    plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                    plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
                } else {
                    plan.bar_k_avail[buf_idx].arrive();
                }
            }

            // Copy the next q
            if (threadIdx.x/32 == 0 && elect_one_sync()) {
                if (batch_idx != sched_meta.end_req_idx) {
                    Tensor gQ = flat_divide(
                        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx+1),
                        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
                    )(_, _, head_block_idx, _0{});
                    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
                    plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
                } else {
                    // This kernel is followed by the combine kernel, so we signal PDL here
                    cudaTriggerProgrammaticLaunchCompletion();
                }
            }

            // Synchronize L and M across warpgroups
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

            if (idx_in_warpgroup%4 == 0) {
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
                    sL[row] = rL[i];
                    sM[row] = rM[i];
                }
            }
            
            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                if (args.is_no_split) {
                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
                } else {
                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
                }
                if (idx_in_warpgroup%4 == 0) {
                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
                    plan.sOScale[row] = o_scales[i];
                }
            }

            // This is a synchronization point for warpgroup 0/1.
            // Warpgroup 0 should wait wg 1 for oBuf/oAccumBuf (overlapped with k) to be free
            // Warpgroup 1 should wait wg 0 for sL to be ready
            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i)
                rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];
            
            int start_head_idx = head_block_idx*BLOCK_M;
            int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
            // WG0 no longer does PV_lo; WG1 handles both PV_lo and PV_hi.
            // WG0 still participates in epilogue barrier and triggers TMA/bulk copy.
            NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

            if (args.is_no_split) {
                float* gSoftmaxLse = (float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + start_head_idx;

                if (threadIdx.x == 0) {
                    SM90_TMA_STORE_5D::copy(
                        &tma_params.tensor_map_o,
                        plan.u.oBuf.data(),
                        0, head_block_idx*64, 0,
                        s_q_idx, batch_idx
                    );
                    cute::tma_store_arrive();
                }

                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
                }

                cute::tma_store_wait<0>();
            } else {
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
                float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;
                float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;
                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_accum_h_q, _1{})
                ));

                if (elect_one_sync()) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int local_row = 0; local_row < BLOCK_M / (256/32); ++local_row) {
                        int row = local_row * (256/32) + (threadIdx.x / 32);
                        if (row < num_valid_seq_q) {
                            SM90_BULK_COPY_S2G::copy(&sOAccumBuf(row, _0{}), &gOAccum(row, _0{}), HEAD_DIM_V*sizeof(float));
                        }
                    }
                    cute::tma_store_arrive();
                }

                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
                }

                cute::tma_store_wait<0>();
            }
            
            sync_all_threads_in_cluster();
        }
    } else if (warpgroup_idx == 1) {
        // [DEBUG] disable reg_dealloc to test if setmaxnreg is the illegal instruction
        // cutlass::arch::warpgroup_reg_dealloc<192>();

        // [DEBUG] early return before main loop to bisect illegal instruction
        return;

        TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
        ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
        Tensor rO_lo = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});
        Tensor rO_hi = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            cute::fill(rO_lo, 0.);
            cute::fill(rO_hi, 0.);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;
                Tensor sV_lo = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});
                Tensor sV_hi = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});

                // Wait for S and sScale
                NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);

                // Scale O (both lo and hi accumulators)
                float cur_scales[2];
                *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup/4)*2);
                CUTE_UNROLL
                for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                    Tensor cur_rO_lo = flatten(rO_lo(make_coord(_, local_row_idx, _), _, _));
                    Tensor cur_rO_hi = flatten(rO_hi(make_coord(_, local_row_idx, _), _, _));
                    CUTE_UNROLL
                    for (int i = 0; i < size(cur_rO_lo); ++i) {
                        cur_rO_lo(i) *= cur_scales[local_row_idx];
                        cur_rO_hi(i) *= cur_scales[local_row_idx];
                    }
                }
                
                // Issue PV_lo: O_lo += S @ V_lo (first 256 V dims)
                gemm<false, -1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sS),
                    thr_mma_PV.partition_fragment_B(sV_lo),
                    rO_lo
                );

                // Issue PV_hi: O_hi += S @ V_hi (second 256 V dims)
                gemm<false, -1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sS),
                    thr_mma_PV.partition_fragment_B(sV_hi),
                    rO_hi
                );
                cute::warpgroup_wait<0>();
                
                if constexpr (CLUSTER_SIZE == 2) {
                    plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                    plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
                } else {
                    plan.bar_k_avail[buf_idx].arrive();
                }
                
                if (block_idx != args.end_block_idx-1)
                    NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);   // Tell WG0 that sScale and sS are available
            }

            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = get_AorC_row_idx(i, idx_in_warpgroup);
                o_scales[i] = plan.sOScale[row];
            }
                
            int start_head_idx = head_block_idx*BLOCK_M;
            int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
            // WG1 does both PV_lo and PV_hi. Write both halves to smem, then barrier.
            // WG0 triggers the final TMA/bulk copy to global.
            if (args.is_no_split) {
                // Write lo half (warpgroup_idx 0 position) and hi half (warpgroup_idx 1 position)
                CUTE_UNROLL
                for (int wg_half = 0; wg_half < 2; ++wg_half) {
                    Tensor sMyOutputBuf = local_tile(sOBuf, Shape<_64, _256>{}, make_coord(_0{}, wg_half));
                    constexpr int NUM_CHUNKS_IN_SW_ATOM = OBUF_SW/16;
                    bf16* base_output_buf_ptrs[NUM_CHUNKS_IN_SW_ATOM];
                    CUTE_UNROLL
                    for (int i = 0; i < NUM_CHUNKS_IN_SW_ATOM; ++i) {
                        base_output_buf_ptrs[i] = &sMyOutputBuf((idx_in_warpgroup/32)*16+idx_in_warpgroup%16, idx_in_warpgroup%32/16*8 + i*16);
                    }
                    auto& rO_cur = (wg_half == 0) ? rO_lo : rO_hi;
                    CUTE_UNROLL
                    for (int idx = 0; idx < (HEAD_DIM_V/2)/16; idx += 1) {
                        using bf16x2 = __nv_bfloat162;
                        bf16x2 a01 = __float22bfloat162_rn(float2{rO_cur(idx*8+0)*o_scales[0], rO_cur(idx*8+1)*o_scales[0]});
                        bf16x2 a23 = __float22bfloat162_rn(float2{rO_cur(idx*8+2)*o_scales[1], rO_cur(idx*8+3)*o_scales[1]});
                        bf16x2 a45 = __float22bfloat162_rn(float2{rO_cur(idx*8+4)*o_scales[0], rO_cur(idx*8+5)*o_scales[0]});
                        bf16x2 a67 = __float22bfloat162_rn(float2{rO_cur(idx*8+6)*o_scales[1], rO_cur(idx*8+7)*o_scales[1]});
                        SM90_U32x4_STSM_N::copy(
                            *reinterpret_cast<uint32_t*>(&a01),
                            *reinterpret_cast<uint32_t*>(&a23),
                            *reinterpret_cast<uint32_t*>(&a45),
                            *reinterpret_cast<uint32_t*>(&a67),
                            *reinterpret_cast<uint128_t*>(base_output_buf_ptrs[idx%4] + (idx/4*4)*16*64)
                        );
                    }
                }

                cutlass::arch::fence_view_async_shared();
                NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

                cute::tma_store_wait<0>();
            } else {
                // Split mode: write both lo and hi halves to sOAccumBuf
                CUTE_UNROLL
                for (int wg_half = 0; wg_half < 2; ++wg_half) {
                    auto& rO_cur = (wg_half == 0) ? rO_lo : rO_hi;
                    CUTLASS_PRAGMA_UNROLL
                    for (int idx = 0; idx < size(rO_cur); idx += 2) {
                        int row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%32/4) + (idx%4 >= 2 ? 8 : 0);
                        int col = wg_half*256 + (idx_in_warpgroup%4)*2 + idx/4*8;
                        *(float2*)(&(sOAccumBuf(row, col))) = float2 {
                            rO_cur(idx) * o_scales[idx%4>=2],
                            rO_cur(idx+1) * o_scales[idx%4>=2],
                        };
                    }
                }
                cutlass::arch::fence_view_async_shared();

                NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

                cute::tma_store_wait<0>();
            }

            sync_all_threads_in_cluster();
        }
    } else {
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<152>();

        static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);
        static constexpr int NUM_TOKENS_PER_THREAD = CLUSTER_SIZE == 1 ? 2 : 1;
        static constexpr int NUM_TOKENS_PER_ROUND = 32; // If head is 128, each CTA is responsible for dequantizing 32 tokens (1 rounds); if head is 64, each CTA is responsible for dequantizing 64 tokens (2 rounds)
        int warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
        int lane_idx = idx_in_warpgroup % 32;
        int my_token_idx_base = warp_idx*8 + lane_idx%8;
        
        CUTE_NO_UNROLL
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            int* gIndices = params.indices + batch_idx*params.stride_indices_b + s_q_idx*params.stride_indices_s_q; // (topk) : (1)
            int* gExtraIndices = params.extra_indices + batch_idx*params.stride_extra_indices_b + s_q_idx*params.stride_extra_indices_s_q; // (extra_topk) : (1)
            
            int nxt_token_indexs[NUM_TOKENS_PER_THREAD];
            CUTE_UNROLL
            for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                if (MODEL_TYPE == ModelType::V32 || args.start_block_idx < args.num_orig_kv_blocks)
                    nxt_token_indexs[round] = __ldg(gIndices + args.start_block_idx*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + round*NUM_TOKENS_PER_ROUND + my_token_idx_base);
            }

            struct IsOrigBlock {};
            struct IsExtraBlock {};

            struct IsFirstExtraBlock {};
            struct IsNotFirstExtraBlock {};
            auto process_one_block = [&](int block_idx, auto is_extra_block_t, auto is_first_extra_block_t) {
                static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                static constexpr bool IS_FIRST_EXTRA_BLOCK = std::is_same_v<decltype(is_first_extra_block_t), IsFirstExtraBlock>;
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;

                int* indices_base;
                int page_block_size;
                int64_t k_block_stride, k_row_stride;
                fp8* k_ptr;
                if constexpr (!IS_EXTRA_BLOCK) {
                    indices_base = gIndices + (block_idx)*TOPK_BLOCK_SIZE;
                    page_block_size = params.page_block_size;
                    k_block_stride = params.stride_kv_block;
                    k_row_stride = params.stride_kv_row;
                    k_ptr = (fp8*)params.kv;
                } else {
                    indices_base = gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE;
                    page_block_size = params.extra_page_block_size;
                    k_block_stride = params.stride_extra_kv_block;
                    k_row_stride = params.stride_extra_kv_row;
                    k_ptr = (fp8*)params.extra_kv;
                }
                [[maybe_unused]] int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
                [[maybe_unused]] int rel_block_idx = IS_EXTRA_BLOCK ? (block_idx - args.num_orig_kv_blocks) : block_idx;
                transac_bar_t* peer_bar_k_remote_ready = get_peer_addr(&(plan.bar_k_remote_ready[buf_idx]));

                // [M3.c.4 Stage-2] Packed-FP8 fused-dequant path.
                // When packed_kcache_ptr is set, we read packed INT-N rows,
                // bit-unpack + affine + R@x on the fly, and write BF16 to sK.
                // Extra KV blocks always use the dense path.
                // [DEBUG] force disable packed path to bisect illegal instruction
                const bool use_packed = false;

                if (use_packed) {
                    // ---- Packed FP8 K-load path (S2-S2 fused dequant) ----
                    //
                    // Process in 7 dim-blocks (448 / 64 = 7).
                    // Per block (64 dims):
                    //   1. compute dequant for all 64 tokens -> staging (8KB smem in union)
                    //   2. each thread reads its own token's 64 dims into registers
                    //   3. named barrier sync (staging no longer needed)
                    //   4. each thread writes regs to GMMA-layout sK
                    // This avoids smem overwrite since staging is fully read
                    // before any sK writes happen.

                    const int qk_nope = params.qk_nope_head_dim;
                    const int row_bits = params.row_bits;
                    const int packed_row_bytes = params.packed_row_bytes;
                    const int nope_bytes = packed_row_bytes - 128;  // rope = 64 bf16 = 128 bytes

                    const uint8_t* pk_base = reinterpret_cast<const uint8_t*>(params.packed_kcache_ptr);
                    const float* sk_base = params.scale_kcache_ptr;
                    const float* R_base = params.R_matrix_ptr;
                    const float* zp_base = params.zero_point_ptr;
                    const int* dob_base = params.dim_of_bit_ptr;
                    const int* bpd_base = params.bitpos_in_dim_ptr;
                    const int64_t pk_block_stride = params.packed_kv_block_stride;

                    bf16* staging = plan.packed_nope_staging;

                    // Wait for the nope buffer to be available
#ifdef FMLA_CLK_PROFILE
                    unsigned long long _clk_p0 = clock64();
#endif
                    plan.bar_k_avail[buf_idx].wait((bar_phase_k>>buf_idx&1)^1);
#ifdef FMLA_CLK_PROFILE
                    unsigned long long _clk_p1 = clock64();
                    if (idx_in_warpgroup == 0) fmla_clk_add(0, _clk_p1 - _clk_p0);
#endif

                    if (CLUSTER_SIZE == 2 && idx_in_warpgroup == 0) {
                        plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx((TOPK_BLOCK_SIZE/2)*(HEAD_DIM_NOPE+HEAD_DIM_ROPE)*sizeof(bf16));
                    }

                    // ---- First, copy rope half directly (no staging needed) ----
                    CUTE_UNROLL
                    for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                        int my_token_idx = my_token_idx_base + round*NUM_TOKENS_PER_ROUND;
                        bf16* sK_rope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*8)*TOPK_BLOCK_SIZE;
                        bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

                        const int token_idx_abs = idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx;
                        const int token_index = __ldg(indices_base + token_idx_abs);

                        if (token_index != -1) {
                            const int block_index = (int)((uint32_t)token_index / (uint32_t)page_block_size);
                            const int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
                            const uint8_t* pk_row = pk_base
                                + block_index * pk_block_stride
                                + rel_idx_in_block * packed_row_bytes;
                            const bf16* rope_bf16 = reinterpret_cast<const bf16*>(pk_row + nope_bytes);

                            CUTE_UNROLL
                            for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE/32; dim_idx += 1) {
                                bf16x8 val = *reinterpret_cast<const bf16x8*>(&rope_bf16[(lane_idx/8)*8 + dim_idx*32]);
                                int smem_offset = (HEAD_DIM_NOPE + dim_idx*32) * TOPK_BLOCK_SIZE;
                                *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&val;
                                if constexpr (CLUSTER_SIZE == 2) {
                                    st_async_128b(sK_rope_peer_base + smem_offset, val, peer_bar_k_remote_ready);
                                }
                            }
                        } else {
                            CUTE_UNROLL
                            for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE/32; dim_idx += 1) {
                                bf16x8 val;
                                *(uint128_t*)&val = uint128_t();
                                int smem_offset = (HEAD_DIM_NOPE + dim_idx*32) * TOPK_BLOCK_SIZE;
                                *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&val;
                                if constexpr (CLUSTER_SIZE == 2) {
                                    st_async_128b(sK_rope_peer_base + smem_offset, val, peer_bar_k_remote_ready);
                                }
                            }
                        }
                    }
#ifdef FMLA_CLK_PROFILE
                    unsigned long long _clk_p2 = clock64();
                    if (idx_in_warpgroup == 0) fmla_clk_add(1, _clk_p2 - _clk_p1);
#endif

                    // ==========================================================
                    // [M3.c.4 Stage-5 Route G step 4+5] wgmma R@X uniform-bit
                    // path (MODEL1 + CLUSTER_SIZE==1 + bu > 0 only).
                    //
                    // Structural rewrite that replaces the per-token 4-barrier
                    // storm of the legacy inner loop with a cooperative
                    // 128-thread fill + tensor-core reduction:
                    //
                    //   for dim_block in 0..HEAD_DIM_NOPE/64:            (7)
                    //     rC[64,64] = 0                                 (fp32)
                    //     for kt in 0..qk_nope/64:                        (7)
                    //       128 threads cooperatively fill:
                    //         sX_tile[t=0..63, d=0..63] bf16              // unpack + affine
                    //         sR_tile[j=0..63, d=0..63] bf16              // R[dim_base+j, kt*64+d]
                    //       fence + NamedBarrier(128)
                    //       wgmma MMA_64x64x16_F32BF16BF16_SS<K,K>:
                    //         rC += sX_tile @ sR_tile^T                  // 4 issues of k16 per tile
                    //       warpgroup_wait<0>
                    //       NamedBarrier(128)                             // release sX/sR for kt+1
                    //     scatter bf16(rC) -> staging via partition_C
                    //     NamedBarrier(128)
                    //     staging -> sK[dim_block tile] via 128-bit stores  (reused legacy path)
                    //     NamedBarrier(128)
                    //
                    // Barrier count per TOPK_BLOCK: 7 * (7*2 + 2) = 112,
                    //   vs legacy uniform 896 (~8x), vs var-bit 1792 (~16x).
                    // R@X FLOPs stay identical but come from tensor cores
                    //   (wgmma m64n64k16) instead of 128 lanes x 224 FMA.
                    //
                    // Preconditions: sX_tile aliases packed_nope_staging (8KB
                    // reused as wgmma A during issue, then overwritten with
                    // bf16(rC) before staging->sK copy). sR_tile is a
                    // dedicated 8KB smem region (packed_r_tile).
                    // ==========================================================
                    // Bit-uniform parameters hoisted here so they are in scope
                    // for BOTH the wgmma_uniform_supported path below AND the
                    // legacy fallback block that follows.
#ifndef FMLA_PRODUCER_NULL_PROBE
                    const int bu = params.bit_uniform;
                    const int u_groups = params.uniform_num_groups;
                    const int u_hdr_bytes = params.uniform_header_bytes;
                    const int u_group_size = params.uniform_group_size;
                    const float u_step_denom = (bu > 0) ? float((1 << bu) - 1) : 1.0f;

                    // [DEBUG] force-disable wgmma uniform path to bisect illegal instruction
                    constexpr bool wgmma_uniform_supported = false;

                    if constexpr (wgmma_uniform_supported) {
                        if (bu > 0) {
                        // [step3k smem-fit revert] Producer R@X wgmma loop,
                        //   single-buffer, dim-block-by-1. Uses only
                        //   packed_nope_staging (aliased sX + staging) and
                        //   packed_r_tile (sR) -- NO alt tiles.
                        //
                        // Why: packed_x_alt_tile + packed_r_alt_tile (+16 KB)
                        //   were REMOVED from SharedMemoryPlan to fit under
                        //   the H20 SM90 opt-in dyn-smem cap of 232448 B
                        //   (MODEL1 plan was 241664 B -> cudaFuncSetAttribute
                        //   invalid argument, kernel never launched). The
                        //   stale Jun-30 binary masked this.
                        //
                        // Route H step3b (producer-null probe) PROVED the
                        //   producer is NOT the decode bottleneck (zeroing
                        //   the ENTIRE nope rebuild gave 0 tps change at
                        //   19.53/19.54 tps). So the speculative sX-double-
                        //   buffer + dim-block-by-2 pipeline those alt tiles
                        //   enabled (Route G step8 + Route H step1) is dead
                        //   weight -- it only traded smem for cycles the
                        //   consumer WG was already waiting on.
                        //
                        // Byte-correctness: same R@X = X @ R.T math, same
                        //   tensor shape, same staging->sK copy layout. Only
                        //   the in-loop schedule changes (1-at-a-time instead
                        //   of 2-at-a-time, no fill-wgmma overlap).
                        Tensor sX_tile = make_tensor(
                            make_smem_ptr(reinterpret_cast<bf16*>(plan.packed_nope_staging)),
                            SmemLayoutXTile{}
                        );
                        Tensor sR_tile = make_tensor(
                            make_smem_ptr(plan.packed_r_tile.data()),
                            SmemLayoutKTile{}
                        );
                        Tensor sStaging = make_tensor(
                            make_smem_ptr(reinterpret_cast<bf16*>(plan.packed_nope_staging)),
                            Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, _1>>{}
                        );

                        TiledMMA tiled_mma_wg = TiledMMA_QK{};
                        ThrMMA thr_mma_wg = tiled_mma_wg.get_slice(idx_in_warpgroup);

                        const int k_tiles = qk_nope / 64;

                        auto fill_sX_tile = [&](int k_base) {
                            CUTE_UNROLL
                            for (int e = 0; e < 32; ++e) {
                                const int lin = e * 128 + idx_in_warpgroup;
                                const int t = lin >> 6;
                                const int d = lin & 63;
                                const int d_global = k_base + d;

                                const int token_index = __ldg(indices_base + t);
                                bool out_of_range = false;
                                if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                                    if (rel_block_idx * TOPK_BLOCK_SIZE + t >= topk_length) {
                                        out_of_range = true;
                                    }
                                }
                                const bool invalid = (token_index == -1) || out_of_range;

                                float x_val = 0.0f;
                                if (!invalid) {
                                    const int block_index = (int)((uint32_t)token_index / (uint32_t)page_block_size);
                                    const int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
                                    const uint8_t* pk_row = pk_base
                                        + block_index * pk_block_stride
                                        + rel_idx_in_block * packed_row_bytes;

                                    const int bit_off_global = d_global * bu;
                                    const int byte_off = bit_off_global >> 3;
                                    const int shift = bit_off_global & 7;
                                    uint32_t word = (uint32_t)pk_row[byte_off];
                                    word |= ((uint32_t)pk_row[byte_off + 1]) << 8;
                                    if (bu > 8) {
                                        word |= ((uint32_t)pk_row[byte_off + 2]) << 16;
                                    }
                                    const uint32_t mask = (1u << bu) - 1u;
                                    const int code = (int)((word >> shift) & mask);

                                    const int g = d_global / u_group_size;
                                    const __half* hdr_h = reinterpret_cast<const __half*>(
                                        pk_row + nope_bytes - u_hdr_bytes + g * 4
                                    );
                                    const float fmin = __half2float(hdr_h[0]);
                                    const float frange = __half2float(hdr_h[1]);
                                    const float fstep = frange * (1.0f / u_step_denom);
                                    x_val = fmaf((float)code, fstep, fmin);
                                }
                                sX_tile(t, d) = bf16(x_val);
                            }
                        };

                        auto fill_sR_tile = [&](int dim_base, int k_base) {
                            CUTE_UNROLL
                            for (int e = 0; e < 32; ++e) {
                                const int lin = e * 128 + idx_in_warpgroup;
                                const int j = lin >> 6;
                                const int d = lin & 63;
                                const int j_global = dim_base + j;
                                const int d_global = k_base + d;
                                const float r_val = __ldg(
                                    R_base + (int64_t)j_global * (int64_t)qk_nope + (int64_t)d_global
                                );
                                sR_tile(j, d) = bf16(r_val);
                            }
                        };

                        auto scatter_rC_to_sK = [&](auto &rC_frag, int dim_base) {
                            Tensor tC_sStaging = thr_mma_wg.partition_C(sStaging);
                            CUTE_UNROLL
                            for (int i = 0; i < size(rC_frag); ++i) {
                                tC_sStaging(i) = bf16(rC_frag(i));
                            }
                            cutlass::arch::fence_view_async_shared();
                            NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                            CUTE_UNROLL
                            for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                                int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
                                const int abs_token = idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx;
                                const int dim_in_block = (lane_idx / 8) * 16;

                                bf16x8 val_lo = *reinterpret_cast<bf16x8*>(
                                    plan.packed_nope_staging + abs_token * 64 + dim_in_block + 0
                                );
                                bf16x8 val_hi = *reinterpret_cast<bf16x8*>(
                                    plan.packed_nope_staging + abs_token * 64 + dim_in_block + 8
                                );

                                bf16* sK_nope_base = plan.u.k[buf_idx].data()
                                    + abs_token * 8 + ((lane_idx / 8) * 16) * TOPK_BLOCK_SIZE;

                                int smem_offset_lo = (dim_base + 0) * TOPK_BLOCK_SIZE;
                                int smem_offset_hi = (dim_base + 8) * TOPK_BLOCK_SIZE;
                                *(__int128_t*)(sK_nope_base + smem_offset_lo) = *(__int128_t*)&val_lo;
                                *(__int128_t*)(sK_nope_base + smem_offset_hi) = *(__int128_t*)&val_hi;
                            }

                            NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                        };

                        CUTE_UNROLL
                        for (int dim_block = 0; dim_block < HEAD_DIM_NOPE / 64; ++dim_block) {
                            const int dim_base = dim_block * 64;

                            Tensor rC = partition_fragment_C(
                                tiled_mma_wg, Shape<Int<64>, Int<64>>{}
                            );
                            clear(rC);

                            for (int kt = 0; kt < k_tiles; ++kt) {
                                const int k_base = kt * 64;
                                fill_sX_tile(k_base);
                                fill_sR_tile(dim_base, k_base);
                                cutlass::arch::fence_view_async_shared();
                                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                                gemm<false, -1>(
                                    tiled_mma_wg,
                                    thr_mma_wg.partition_fragment_A(sX_tile),
                                    thr_mma_wg.partition_fragment_B(sR_tile),
                                    rC
                                );
                                cute::warpgroup_wait<0>();
                                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                            }

                            scatter_rC_to_sK(rC, dim_base);
                        }

                        cutlass::arch::fence_view_async_shared();
                        // Fall through to shared bar_k_local_ready arrive +
                        // is_kv_valid write below (outside the packed branch).
                        }  // end if (bu > 0) inside wgmma_uniform_supported
                    }

                    if (!wgmma_uniform_supported || bu == 0) {

                    // [M3.c.4 Stage-5 Bug-3 fix] Per-token full unpack + affine
                    // + R@x dequant, with **unified barrier sequence** for both
                    // valid and invalid tokens.
                    //
                    // Calibration convention (build_rotated_kv_calib.py +
                    // rotated_quant_dsv4_kernels.py):
                    //   store:   K_rot = nope @ R; codes = round((K_rot - zero) / scale)
                    //   load:    nope  = (codes * scale + zero) @ R.t()
                    // With R row-major in memory, the inverse rotation produces
                    //   result[j] = sum_d R[j, d] * x[d]
                    // where x[d] = codes[d] * scale[d] + zero[d]. This mirrors
                    // dense_fp8 fork's flash_fwd_mla_kernel.h prologue + prefetch.
                    //
                    // Why unified barriers: the previous revision had the
                    // invalid path skip all 4 NamedBarriers in the per-token
                    // loop while the valid path did them. Although `invalid`
                    // is uniform across the producer warpgroup's 128 threads
                    // per token, mixing barrier-bearing and barrier-free
                    // iterations of the SAME loop creates a fragile contract
                    // with the consumer warpgroups' wait on
                    // bar_k_local_ready[buf_idx] (arrived after the dim_block
                    // outer loop). Forcing both branches through the exact
                    // same 4-barrier sequence makes the producer's smem
                    // ordering provably consistent with the dense fork's
                    // gold reference (which has no invalid branching at all).
                    //
                    // s_codes / s_x are token-scoped scratchpads shared by the
                    // 128 producer-WG threads. qk_nope <= 512 (V32: 512, MODEL1:
                    // 448); we size to 576 to stay above HEAD_DIM_K.
                    __shared__ int s_codes[576];
                    __shared__ float s_x[576];
                    // [Stage-5 Route G step6.4] revert step6.3 header smem cache:
                    // the extra NamedBarrier::sync(128,...) needed to publish
                    // s_hdr across the 128 producer threads costs ~100 cyc/token
                    // but empirically dropped 32-req steady-state gen tps from
                    // 12.19 -> 6.76 (measured on fp8-dsv4 canary 09:05:35 UTC).
                    // Producer WG has only 128 threads and is Q/K-bound, not
                    // header-LDG bound, so per-thread ldg header (already
                    // L2-hot after step6.1 warm-up) is cheaper than a
                    // whole-warp synchronization. Route to reclaim tps.

                    // [Stage-5 Route G step 5] uniform-bit fast path.
                    //
                    // Selected when params.bit_uniform > 0. Each nope dim
                    // uses `bu` contiguous bits, so a single thread can
                    // locate its own dim's code via byte shift + mask
                    // (no atomicOr scatter). Per-token affine lives in a
                    // 28 B (for 7 groups) header right after the code
                    // bytes and right before the rope BF16 tail:
                    //     [code_bytes][28 B header][128 B rope]
                    // header[g] = (fp16 min, fp16 range), 4 B per group.
                    // s_x[d] = code * (range / ((1<<bu)-1)) + min.
                    //
                    // Barrier count per token drops from 4 to 2 (one
                    // after we fill s_x, one after R@x staging write
                    // before reusing s_x for the next t).
                    // (bu/u_groups/u_hdr_bytes/u_group_size/u_step_denom
                    // are declared above, before the wgmma_uniform_supported
                    // branch, so they are in scope here.)

                    CUTE_UNROLL
                    for (int dim_block = 0; dim_block < HEAD_DIM_NOPE / 64; ++dim_block) {
                        const int dim_base = dim_block * 64;

                        // ---- Step 1: per-token unpack + affine + R@x ----
                        // Variable-width bit layout described by row_bits global bit slots:
                        //   bit i lives at byte (i/8), bit (i%8) of a packed row, and
                        //   contributes value (1 << bitpos_in_dim[i]) to dim_of_bit[i].
                        //
                        // For each token t we (always 4 NamedBarriers, both paths):
                        //   (a) s_codes[d] = sum_{i : dim_of_bit[i] == d} bit(i) << bitpos_in_dim[i]
                        //   (b) s_x[d]     = s_codes[d] * scale[d] + zero[d]
                        //   (c) staging[t, d_in_block] = sum_d R[(dim_base+d_in_block), d] * s_x[d]
                        for (int t = 0; t < TOPK_BLOCK_SIZE; ++t) {
                            int token_index = __ldg(indices_base + t);
                            bool out_of_range = false;
                            if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                                if (rel_block_idx*TOPK_BLOCK_SIZE + t >= topk_length) {
                                    out_of_range = true;
                                }
                            }
                            const bool invalid = (token_index == -1) || out_of_range;

                            // Compute pk_row pointer up-front (only used when valid).
                            const uint8_t* pk_row = nullptr;
                            if (!invalid) {
                                const int block_index = (int)((uint32_t)token_index / (uint32_t)page_block_size);
                                const int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
                                pk_row = pk_base
                                    + block_index * pk_block_stride
                                    + rel_idx_in_block * packed_row_bytes;
                            }

                            if (bu > 0) {
                                // ---- Uniform-bit fast path (Stage-5 Route G step6.4). ----
                                // Same as step6.1: per-thread ldg header (L2-hot
                                // after 7-group warm-up on the first few tokens)
                                // + 1 FMA per dim. No smem cache (step6.3 attempt
                                // regressed 12.19 -> 6.76 tps due to the extra
                                // NamedBarrier::sync(128,...) needed to publish
                                // s_hdr across the 128 producer-WG threads).
                                const uint8_t* hdr_base = invalid
                                    ? nullptr
                                    : (pk_row + nope_bytes - u_hdr_bytes);

                                if (!invalid) {
                                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                                        const int bit_off_global = d * bu;
                                        const int byte_off = bit_off_global >> 3;
                                        const int shift = bit_off_global & 7;
                                        uint32_t word = (uint32_t)pk_row[byte_off];
                                        word |= ((uint32_t)pk_row[byte_off + 1]) << 8;
                                        if (bu > 8) {
                                            word |= ((uint32_t)pk_row[byte_off + 2]) << 16;
                                        }
                                        const uint32_t mask = (1u << bu) - 1u;
                                        const int code = (int)((word >> shift) & mask);
                                        const int g = d / u_group_size;
                                        const __half* hdr_h =
                                            reinterpret_cast<const __half*>(hdr_base + g * 4);
                                        const float fmin   = __half2float(hdr_h[0]);
                                        const float frange = __half2float(hdr_h[1]);
                                        const float fstep  = frange * (1.0f / u_step_denom);
                                        s_x[d] = fmaf((float)code, fstep, fmin);
                                    }
                                } else {
                                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                                        s_x[d] = 0.0f;
                                    }
                                }
                                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                            } else {
                                // ---- Legacy variable-bit path (path-disjoint). ----
                                // (a-1) init s_codes (always)
                                for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                                    s_codes[d] = 0;
                                }
                                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                                // (a-2) atomicOr each bit slot into its dim (skip for invalid;
                                // codes remain 0 from init).
                                if (!invalid) {
                                    for (int bit_idx = idx_in_warpgroup; bit_idx < row_bits; bit_idx += 128) {
                                        const int d = __ldg(dob_base + bit_idx);
                                        const int bpos = __ldg(bpd_base + bit_idx);
                                        const int byte_off = bit_idx >> 3;
                                        const int bit_off = bit_idx & 7;
                                        const int bit_v = (pk_row[byte_off] >> bit_off) & 1;
                                        if (bit_v) {
                                            atomicOr(&s_codes[d], 1 << bpos);
                                        }
                                    }
                                }
                                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                                // (b) affine dequant: s_x[d] = codes*scale + zero  for valid
                                //                     s_x[d] = 0                    for invalid
                                // sk_base / zp_base are per-dim length-qk_nope.
                                if (!invalid) {
                                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                                        s_x[d] = (float)s_codes[d] * sk_base[d] + zp_base[d];
                                    }
                                } else {
                                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                                        s_x[d] = 0.0f;
                                    }
                                }
                                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                            }

                            // (c) R @ s_x for this dim_block's 64 outputs.
                            // [Stage-5 Route G step6.1] 双 lane 协作 + float4 向量化.
                            //   - 128 lanes 全部激活: pair (l, l^1) 同 warp，
                            //     每 pair 计算 1 个 output j = l/2 (lane 0/1
                            //     -> j=0, lane 2/3 -> j=1, ..., lane 62/63 -> j=31;
                            //     lane 64/65 -> j=32, ..., lane 126/127 -> j=63)。
                            //   - 每 lane 累加 half dims (224)，从 half-offset
                            //     开始，用 float4 一次 load 4 个 (R, s_x) 做 4× FMA。
                            //   - qk_nope=448 -> 224/4 = 56 iter/lane。
                            //   - 用 __shfl_xor_sync(mask, sum, 1) 在 pair 内合并。
                            //   - 相比旧 kernel (64 lane 各 448 标量 MADD)：
                            //     lane 利用率 64->128 (×2)，
                            //     每 lane MADD 数 448->56*4=224 (× 0.5 计算量)，
                            //     LDG/LDS 从 float 变 float4 (×4 带宽利用)。
                            //     综合 ~4× 加速 R@x 阶段。
                            //   - 对 s_x 语义无假设：invalid token s_x 全 0 时
                            //     sum 天然为 0，barrier 序列与 legacy 完全一致。
                            {
                                const int lane = idx_in_warpgroup;
                                const int pair_id = lane >> 1;          // 0..63
                                const int half = lane & 1;              // 0 or 1
                                const int j = dim_base + pair_id;
                                const float* R_row = R_base + (int64_t)j * (int64_t)qk_nope;
                                const int qk_half = qk_nope >> 1;       // 224
                                const int d_start = half * qk_half;
                                float sum = 0.0f;
                                #pragma unroll 1
                                for (int d = 0; d < qk_half; d += 4) {
                                    const int gd = d_start + d;
                                    const float4 r4 = *reinterpret_cast<const float4*>(R_row + gd);
                                    const float4 x4 = *reinterpret_cast<const float4*>(&s_x[gd]);
                                    sum += r4.x * x4.x;
                                    sum += r4.y * x4.y;
                                    sum += r4.z * x4.z;
                                    sum += r4.w * x4.w;
                                }
                                // Merge lane pair (l, l^1) within warp.
                                sum += __shfl_xor_sync(0xffffffff, sum, 1);
                                if (half == 0) {
                                    staging[t * 64 + pair_id] = bf16(sum);
                                }
                            }
                            // Sync before reusing s_codes/s_x for the next token.
                            NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                        }
                        // [Stage-5 Route G step7-cleanup] Removed a redundant
                        // NamedBarrier here. The barrier inside the for-t loop
                        // (post-R@x, pre next-iter s_x rewrite) already fires
                        // on the t=63 iteration and synchronizes all 128
                        // producer threads. All 64 staging[] entries are
                        // guaranteed visible + globally consistent at loop
                        // exit, so the extra sync before staging->sK read
                        // was pure overhead (~50-100 cyc * 7 dim_blocks =
                        // ~350-700 cyc per TOPK_BLOCK saved).

                        // ---- Step 2 + 3: per round, read staging to regs, write sK ----
                        CUTE_UNROLL
                        for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                            int my_token_idx = my_token_idx_base + round*NUM_TOKENS_PER_ROUND;
                            const int abs_token = idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx;
                            const int dim_in_block = (lane_idx / 8) * 16;

                            // Read this thread's 16 dims (= 2 x bf16x8) into registers
                            bf16x8 val_lo = *reinterpret_cast<bf16x8*>(&staging[abs_token * 64 + dim_in_block + 0]);
                            bf16x8 val_hi = *reinterpret_cast<bf16x8*>(&staging[abs_token * 64 + dim_in_block + 8]);

                            // Write registers to GMMA-layout sK
                            bf16* sK_nope_base = plan.u.k[buf_idx].data() + abs_token*8 + ((lane_idx/8)*16)*TOPK_BLOCK_SIZE;
                            bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);

                            int smem_offset_lo = (dim_base + 0) * TOPK_BLOCK_SIZE;
                            int smem_offset_hi = (dim_base + 8) * TOPK_BLOCK_SIZE;
                            *(__int128_t*)(sK_nope_base + smem_offset_lo) = *(__int128_t*)&val_lo;
                            *(__int128_t*)(sK_nope_base + smem_offset_hi) = *(__int128_t*)&val_hi;
                            if constexpr (CLUSTER_SIZE == 2) {
                                st_async_128b(sK_nope_peer_base + smem_offset_lo, val_lo, peer_bar_k_remote_ready);
                                st_async_128b(sK_nope_peer_base + smem_offset_hi, val_hi, peer_bar_k_remote_ready);
                            }
                        }

                        // All threads done reading staging; safe to refill next dim-block
                        NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                    }

                    fence_view_async_shared();
                    }  // end if (!wgmma_uniform_supported || bu == 0) legacy path
#else
                    // [Route H step3b] producer null-work: nope reconstruction
                    // skipped entirely; only rope-copy above + handshake below
                    // survive. Keep the async-proxy fence so consumer wgmma
                    // reads of sK are ordered after the rope stores.
                    fence_view_async_shared();
#endif  // FMLA_PRODUCER_NULL_PROBE (skips all nope reconstruction)
#ifdef FMLA_CLK_PROFILE
                    unsigned long long _clk_p3 = clock64();
                    if (idx_in_warpgroup == 0) {
                        fmla_clk_add(2, _clk_p3 - _clk_p2);
                        fmla_clk_add(5, 1ull);  // producer block sample count
                    }
#endif
                } else {
                    // ---- Original dense FP8 K-load path ----
                CUTE_UNROLL
                for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                    int my_token_idx = my_token_idx_base + round*NUM_TOKENS_PER_ROUND;
                    bf16* sK_nope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*16)*TOPK_BLOCK_SIZE;
                    bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);

                    // Get prefetched token index
                    int token_index;
                    if constexpr (!IS_EXTRA_BLOCK) {
                        token_index = nxt_token_indexs[round];
                        if (block_idx+1 != (MODEL_TYPE == ModelType::V32 ? args.end_block_idx : args.num_orig_kv_blocks))
                            nxt_token_indexs[round] = __ldg(gIndices + (block_idx+1)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                    } else {
                        if constexpr (IS_FIRST_EXTRA_BLOCK) {
                            token_index = __ldg(gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                        } else {
                            token_index = nxt_token_indexs[round];
                        }
                        if (block_idx+1 != args.end_block_idx)
                            nxt_token_indexs[round] = __ldg(gExtraIndices + (block_idx+1-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                    }
                    
                    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                        // For MODEL1, we need to check whether the token_index is within topk_length
                        if (rel_block_idx*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx >= topk_length) {
                            token_index = -1;   // To prevent IMA when we have invalid (e.g. INT_MAX) topk indexes outside topk_length
                        }
                    }

                    int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);   // Use uint32_t division and mod to improve performance
                    int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;   // NOTE When token_index is -1 (UINT_MAX), UINT_MAX%page_block_size < page_block_size, so there will be no illegal-memory-access error

                    fp8* gK_base;
                    bf16 scales[NUM_SCALES];
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        static_assert(NUM_SCALES == 4);
                        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*k_row_stride;
                        float scales_float[NUM_SCALES];
                        *(float4*)(scales_float) = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>((float*)(gK_base+HEAD_DIM_NOPE));
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; ++i) {
                            scales[i] = (bf16)scales_float[i];
                        }
                    } else {
                        static_assert(NUM_SCALES == 8);
                        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16));
                        fp8_e8m0* gK_scales_base = (fp8_e8m0*)(k_ptr + block_index*k_block_stride + page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*sizeof(bf16)) + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0));
                        fp8_e8m0 scales_e8m0[NUM_SCALES];
                        *(int64_t*)scales_e8m0 = __ldg((int64_t*)gK_scales_base);
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; i += 2) {
                            *(__nv_bfloat162_raw*)(scales+i) = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(scales_e8m0+i));
                        }
                    }

                    // Wait for the nope buffer to be available
                    if (round == 0) {
                        plan.bar_k_avail[buf_idx].wait((bar_phase_k>>buf_idx&1)^1);
                    }
                    
                    if (CLUSTER_SIZE == 2 && round == 0 && idx_in_warpgroup == 0) {
                        plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx((TOPK_BLOCK_SIZE/2)*(HEAD_DIM_NOPE+HEAD_DIM_ROPE)*sizeof(bf16));
                    }

                    // Collectively copy from global memory and dequant
                    // For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
                    
                    fp8* gK_nope = gK_base + (lane_idx/8)*16;
                    if (token_index == -1) {
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; ++i)
                            scales[i] = (bf16)0.0f;
                    }
                    CUTE_UNROLL
                    for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE/64; dim_idx += 1) {
                        fp8x16 cur_fp8x16 = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_nope + dim_idx*64);   // We use EVICT_LAST here since gK_base may not be aligned to 32B (for V3.2) and the performance is the best among all cache hints (for MODEL1)
                        bf16 scale = scales[MODEL_TYPE == ModelType::V32 ? dim_idx/2 : dim_idx];
                        auto dequant_and_save_bf16x8 = [&](const fp8x8 &data, int offset) {
                            int smem_offset = (dim_idx*64 + offset) * TOPK_BLOCK_SIZE;
                            bf16x8 cur_bf16x8 = cvt_fp8x8_bf16x8(data, __bfloat162bfloat162(*(__nv_bfloat16*)(&scale)));
                            *(__int128_t*)(sK_nope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                            if constexpr (CLUSTER_SIZE == 2) {
                                st_async_128b(sK_nope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                            }
                        };
                        if (token_index == -1)
                            *(uint128_t*)(&cur_fp8x16) = uint128_t();
                        dequant_and_save_bf16x8(cur_fp8x16.lo, 0);
                        dequant_and_save_bf16x8(cur_fp8x16.hi, 8);
                    }

                    bf16* gK_rope;
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE+NUM_SCALES*sizeof(float)) + (lane_idx/8)*8;
                    } else {
                        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE) + (lane_idx/8)*8;
                    }
                    bf16* sK_rope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*8)*TOPK_BLOCK_SIZE;
                    bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

                    CUTE_UNROLL
                    for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE/32; dim_idx += 1) {
                        bf16x8 cur_bf16x8 = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim_idx*32);
                        if constexpr (MODEL_TYPE == ModelType::V32) {
                            // NOTE We do not need to mask the RoPE part for V3.2 since it isn't involved in the SV gemm
                        } else {
                            if (token_index == -1)
                                *(uint128_t*)(&cur_bf16x8) = uint128_t();
                        }
                        int smem_offset = (HEAD_DIM_NOPE + dim_idx*32) * TOPK_BLOCK_SIZE;
                        *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                        if constexpr (CLUSTER_SIZE == 2) {
                            st_async_128b(sK_rope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                        }
                    }
                }

                fence_view_async_shared();
                }  // end if (use_packed) / else

                if (idx_in_warpgroup < 32) {
                    // We put this after fence_view_async_shared() since this won't be read by async proxy
                    auto is_index_valid = [&](int index, int offset_within_thread) -> bool {
                        if constexpr (MODEL_TYPE == ModelType::V32) {
                            return index != -1;
                        } else {
                            return index != -1 && rel_block_idx*TOPK_BLOCK_SIZE + lane_idx*2 + offset_within_thread < topk_length;
                        }
                    };
                    int2 indices = __ldg((int2*)(indices_base + lane_idx*2));
                    *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx*2]) = {
                        is_index_valid(indices.x, 0),
                        is_index_valid(indices.y, 1)
                    };
                }

                // Signal the barrier
                plan.bar_k_local_ready[buf_idx].arrive();
                bar_phase_k ^= 1 << buf_idx;
            };

            if constexpr (MODEL_TYPE == ModelType::V32) {
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
                }
            } else {
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
                }

                if (args.num_orig_kv_blocks < args.end_block_idx) {
                    process_one_block(max(args.start_block_idx, args.num_orig_kv_blocks), IsExtraBlock{}, IsFirstExtraBlock{});
                }
                CUTE_NO_UNROLL
                for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks)+1; block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsExtraBlock{}, IsNotFirstExtraBlock{});
                }
            }

            sync_all_threads_in_cluster();
        }
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif

}

template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.h_q % BLOCK_M == 0);
    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
        constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8;
        // [M3.c.4 Stage-5] When use_packed=true (packed_kcache_ptr non-null),
        // kv tensor carries packed-FP8 bytes_per_token (e.g. 268 for b=2.5)
        // and the kernel ignores stride_kv_row in favor of packed_row_bytes.
        // Native FP8 path (packed_kcache_ptr == nullptr) still enforces 584.
        if (params.packed_kcache_ptr == nullptr) {
            KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
            if (params.extra_kv != nullptr) {
                KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN, "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
            }
        }
    } else {
        KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
        KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
        KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
    }

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
            )
        ),
        SmemLayoutQ{}
    );
    
    CUtensorMap tensor_map_o;
    {
        // Here we manually construct TMA descriptor to store O, in order to leverage 5D TMA
        uint64_t size[5] = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V/OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
        uint64_t stride[4] = {params.stride_o_h_q*sizeof(bf16), OBUF_SW*sizeof(bf16), params.stride_o_s_q*sizeof(bf16), params.stride_o_b*sizeof(bf16)};
        uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V/OBUF_SW, 1, 1};
        uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_o,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            5,
            params.out,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            OBUF_SW == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
                OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
                OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B :
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_Q, tma_Q,
        tensor_map_o
    };
    auto mla_kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // NOTE Don't use PDL because of potential compiler bugs!
    // cudaLaunchAttribute mla_kernel_attributes[1];
    // mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    // mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    // cudaLaunchConfig_t mla_kernel_config = {
    //     dim3(num_m_block, params.h_k, params.num_sm_parts),
    //     dim3(NUM_THREADS, 1, 1),
    //     smem_size,
    //     stream,
    //     mla_kernel_attributes,
    //     1
    // };
    // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
    cutlass::ClusterLaunchParams launch_params = {
        dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)mla_kernel, params, tma_params
    );
    KU_CHECK_KERNEL_LAUNCH();

    // [DEBUG] Force synchronous error check to pinpoint illegal instruction.
    {
        cudaError_t _e = cudaStreamSynchronize(params.stream);
        if (_e != cudaSuccess) {
            fprintf(stderr, "[DEBUG step3k] mla_kernel sync error: %s at %s:%d\n",
                cudaGetErrorString(_e), __FILE__, __LINE__);
            fflush(stderr);
        }
    }

#ifdef FMLA_CLK_PROFILE
    // [Route H step3k] throttled readback of the segment cycle counters.
    //   Print mean cycles/block/segment every N launches, then zero the
    //   accumulators. N chosen so decode-loop noise averages out while
    //   staying human-readable in the server log.
    {
        static thread_local unsigned long long _fmla_launch_ctr = 0;
        constexpr unsigned long long PRINT_EVERY = 500ull;
        if ((++_fmla_launch_ctr % PRINT_EVERY) == 0) {
            unsigned long long h[8] = {0};
            cudaMemcpyFromSymbol(h, g_fmla_clk, sizeof(h));
            unsigned long long np = h[5] ? h[5] : 1ull;  // producer samples
            unsigned long long nc = h[6] ? h[6] : 1ull;  // consumer samples
            double p0 = (double)h[0] / np, p1 = (double)h[1] / np, p2 = (double)h[2] / np;
            double c3 = (double)h[3] / nc, c4 = (double)h[4] / nc;
            fprintf(stderr,
                "[FMLA_CLK step3k] launch#%llu np=%llu nc=%llu | "
                "PROD bar_avail=%.0f rope=%.0f nope_rebuild=%.0f | "
                "CONS bar_ready=%.0f QK_softmax=%.0f (cyc/block)\n",
                _fmla_launch_ctr, np, nc, p0, p1, p2, c3, c4);
            fflush(stderr);
            unsigned long long z[8] = {0};
            cudaMemcpyToSymbol(g_fmla_clk, z, sizeof(z));
        }
    }
#endif
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}
