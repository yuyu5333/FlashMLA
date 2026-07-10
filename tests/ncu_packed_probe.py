"""Standalone single-GPU microbench that drives the packed FP8 sparse-decode
FlashMLA kernel (use_packed=true branch) so ncu / nsys can attach and report
the definitive stall-reason / memory-throughput / occupancy breakdown.

Why: clock64 segment profiling localized WHERE the time goes (producer
nope_rebuild ~477K cyc/block, fill_sX 231K = per-token scattered global read),
but never answered WHY it stalls (memory-latency-bound vs occupancy-bound vs
barrier-bound). Those imply opposite levers (byte-reduction/vectorize vs raise
occupancy). This target lets ncu settle it with WarpState + Memory Workload.

Approach: reuse FlashMLA's own validated testcase generator to build a legal
MODEL1_FP8Sparse decode workload (q / k_cache / indices / sched_meta). Then
build a byte-correct packed swa cache (rotate+affine+bitpack) whose row layout
matches indices_in_kvcache exactly, and inject the 6 packed kwargs so the
kernel runs the use_packed inner loop (no IMA).

Usage (inside fp8-dsv4, cwd /workspace/FlashMLA):
    CUDA_VISIBLE_DEVICES=7 python3 tests/ncu_packed_probe.py         # sanity
    CUDA_VISIBLE_DEVICES=7 ncu --set full -k regex:splitkv.*sparse -c 3 \
        python3 tests/ncu_packed_probe.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ for lib,quant
sys.path.insert(0, "/workspace/sglang-bytedance/python")        # sglang fork

import flash_mla
import lib
from lib import RawTestParamForDecode as RawTestParam

from sglang.srt.mem_cache.rotated_quant_dsv4_memory_pool import (
    build_synthetic_dsv4_calibration,
)
from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
    packed_bytes_per_token,
    rotated_store_to_packed,
    _get_cached_cfg_gpu,
)

dev = torch.device("cuda:0")
# The FlashMLA testcase generator (lib.generate_testcase_for_decode ->
# _randperm_batch) allocates helper tensors on the *default* device. Match the
# validated tests (test_flash_mla_sparse_decoding.py:240) so cuda tensors and
# cpu helpers don't collide.
torch.set_default_device(dev)
torch.cuda.set_device(dev)
# MODEL1_FP8Sparse layout hardcodes 2-byte (bf16) rope; the generator uses the
# default dtype, so it must be bf16 (matching the validated test main()).
torch.set_default_dtype(torch.bfloat16)

# ---- production-like MODEL1 swa decode workload ----
B = int(os.environ.get("PROBE_B", "32"))
H_Q = 128
S_KV = int(os.environ.get("PROBE_SKV", "4096"))
TOPK = int(os.environ.get("PROBE_TOPK", "512"))
BLOCK_SIZE = int(os.environ.get("PROBE_BLK", "64"))
ITERS = int(os.environ.get("PROBE_ITERS", "50"))
QK_NOPE = 448
D_QK = 512
BIT_UNIFORM = int(os.environ.get("PROBE_BIT_UNIFORM", "3"))

p = RawTestParam(
    b=B, h_q=H_Q, s_q=1, h_kv=1, s_kv=S_KV, is_varlen=False, topk=TOPK,
    have_topk_length=False, enable_attn_sink=True,
    block_size=BLOCK_SIZE, d_qk=D_QK, d_v=512,
    check_correctness=False, num_runs=0, seed=0,
).to_test_param()

t = lib.generate_testcase_for_decode(p)
sched_meta = flash_mla.get_mla_metadata()[0]

kv_scope = t.kv_scope
k_cache = kv_scope.get_kvcache_for_flash_mla()   # MODEL1_FP8Sparse quantized
indices = kv_scope.indices_in_kvcache            # [b, s_q, topk] int32
# dequantized bf16 kv (rotate source): kv_scope.blocked_k after quant_and_dequant_
# is the dequantized bf16 [num_blocks, block_size, h_kv, d_qk].
kv_bf16 = kv_scope.blocked_k.view(-1, D_QK).contiguous()   # [num_rows, 512]
num_rows = kv_bf16.shape[0]
print(f"[probe] b={B} topk={TOPK} block={BLOCK_SIZE} num_rows={num_rows} "
      f"k_cache.shape={tuple(k_cache.shape)}")

# ---- build byte-correct packed cache, row layout == indices_in_kvcache ----
cfg = build_synthetic_dsv4_calibration(1, QK_NOPE)[0]
row_bytes_nope = cfg.row_bytes
bpt = packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
print(f"[probe] bit_uniform={cfg.bit_uniform} row_bytes_nope={row_bytes_nope} "
      f"packed_bpt={bpt}  (native FP8 = 584)")

# indices_in_kvcache = block_index * block_size + offset. That equals the flat
# row into k_cache.view(-1, d_qk) i.e. our num_rows. Use page_size=block_size so
# the packed store's page*page_size+slot maps identically.
PAGE_SIZE = BLOCK_SIZE
num_pages = (num_rows + PAGE_SIZE - 1) // PAGE_SIZE
packed_bytes_per_page = bpt * PAGE_SIZE
packed_cache = torch.zeros(num_pages, packed_bytes_per_page,
                           dtype=torch.uint8, device=dev)

loc = torch.arange(num_rows, dtype=torch.int32, device=dev)
CHUNK = 16384
# replace NaN (unused slots) with 0 so pack does not propagate NaN
kv_src = torch.nan_to_num(kv_bf16.to(torch.bfloat16), nan=0.0)
for s in range(0, num_rows, CHUNK):
    e = min(s + CHUNK, num_rows)
    rotated_store_to_packed(
        kv_src[s:e], packed_cache, loc[s:e], page_size=PAGE_SIZE, cfg=cfg,
    )
torch.cuda.synchronize()
packed_rows = packed_cache.view(num_pages * PAGE_SIZE, bpt)

cfg_gpu = _get_cached_cfg_gpu(cfg, dev)
_bu = int(cfg.bit_uniform)
packed_kwargs = {
    "packed_kcache": packed_rows,
    "scale_kcache": cfg_gpu["scale"],
    "R_matrix": cfg_gpu["R_bf16"] if _bu > 0 else cfg_gpu["R"],
    "zero_point": cfg_gpu["zero"],
    "dim_of_bit": cfg_gpu["dim_of_bit"],
    "bitpos_in_dim": cfg_gpu["bitpos_in_dim"],
    "bit_uniform": _bu,
}


def one_call():
    return flash_mla.flash_mla_with_kvcache(
        q=t.q,
        k_cache=k_cache,
        head_dim_v=p.d_v,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=sched_meta,
        softmax_scale=t.sm_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=kv_scope.topk_length,
        attn_sink=t.attn_sink,
        **packed_kwargs,
    )[0]


out = one_call()
torch.cuda.synchronize()
print(f"[probe] out.shape={tuple(out.shape)} "
      f"finite={torch.isfinite(out).all().item()}")

for _ in range(ITERS):
    one_call()
torch.cuda.synchronize()
print("[probe] done")
