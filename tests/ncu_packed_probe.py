"""Standalone single-GPU microbench that drives the packed FP8 sparse-decode
FlashMLA kernel (use_packed=true branch) so ncu / nsys can attach and report
the definitive stall-reason / memory-throughput / occupancy breakdown.

Why: clock64 segment profiling localized WHERE the time goes (producer
nope_rebuild ~477K cyc/block, fill_sX 231K = per-token scattered global read),
but never answered WHY it stalls (memory-latency-bound vs occupancy-bound vs
barrier-bound). Those imply opposite levers (byte-reduction/vectorize vs raise
occupancy). This target lets ncu settle it with WarpState + Memory Workload.

Usage (inside fp8-dsv4, cwd /workspace/FlashMLA):
    python3 tests/ncu_packed_probe.py                 # sanity run
    ncu --set full -k regex:splitkv_mla_fp8_sparse -c 1 \
        python3 tests/ncu_packed_probe.py

It builds a byte-correct packed swa cache via sglang's rotated_store_to_packed
so the kernel's use_packed inner loop reads valid memory (no IMA).
"""
import os
import sys

import torch

# sglang fork on PYTHONPATH so we reuse the exact pack path.
sys.path.insert(0, "/workspace/sglang-bytedance/python")

import flash_mla
from sglang.srt.mem_cache.rotated_quant_dsv4_memory_pool import (
    build_synthetic_dsv4_calibration,
)
from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
    packed_bytes_per_token,
    rotated_store_to_packed,
    _get_cached_cfg_gpu,
)

torch.manual_seed(0)
dev = torch.device("cuda:0")

# ---- production MODEL1 swa shapes ----
H_Q = 128            # MODEL1 num_heads_q
D_QK = 512           # nope(448) + rope(64)
QK_NOPE = 448
QK_ROPE = 64
D_V = 512
PAGE_SIZE = 128      # swa page_size in the packed pool view row layout
BLOCK_SIZE = 128     # k_cache page_block_size for sparse swa
BIT_UNIFORM = int(os.environ.get("PROBE_BIT_UNIFORM", "3"))

# Batch / topk chosen to saturate the producer like the 32-req decode workload.
B = int(os.environ.get("PROBE_B", "32"))
S_Q = 1
TOPK = int(os.environ.get("PROBE_TOPK", "512"))   # aligned to 64
S_KV = int(os.environ.get("PROBE_SKV", "4096"))
ITERS = int(os.environ.get("PROBE_ITERS", "50"))

assert TOPK % 64 == 0

cfg = build_synthetic_dsv4_calibration(1, QK_NOPE)[0]
# cfg.row_bytes / row_bits are set by RotatedQuantizerConfig from bits.
row_bytes_nope = cfg.row_bytes
bpt = packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
print(f"[probe] bit_uniform={cfg.bit_uniform} row_bytes_nope={row_bytes_nope} "
      f"packed_bpt={bpt}  (native FP8 = 584)")

# ---- allocate a packed paged cache large enough for S_KV tokens ----
num_pages = (B * S_KV + PAGE_SIZE - 1) // PAGE_SIZE + 4
packed_bytes_per_page = bpt * PAGE_SIZE
packed_cache = torch.zeros(num_pages, packed_bytes_per_page,
                           dtype=torch.uint8, device=dev)

# ---- fill the packed cache with byte-correct data (rotate+affine+bitpack) ----
num_rows = num_pages * PAGE_SIZE
# write in chunks to avoid a giant [num_rows,512] bf16 temp
CHUNK = 8192
loc = torch.arange(num_rows, dtype=torch.int32, device=dev)
for s in range(0, num_rows, CHUNK):
    e = min(s + CHUNK, num_rows)
    inp = torch.randn(e - s, D_QK, dtype=torch.bfloat16, device=dev) * 0.1
    rotated_store_to_packed(
        inp, packed_cache, loc[s:e], page_size=PAGE_SIZE, cfg=cfg,
    )
torch.cuda.synchronize()

packed_rows = packed_cache.view(num_rows, bpt)

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

# ---- k_cache is the shadow FP8 layout the kernel formally requires; the
# use_packed branch reads packed_kcache instead, but the arg must be a valid
# tensor of the sparse fp8 layout (num_blocks, block_size, h_k=1, d_qk). ----
# sparse fp8 needs 656B for d_qk=576; here d_qk=512 -> 576B/token layout.
# Build it as bytes: (num_blocks, block_size, 1, D_QK) fp8-ish via uint8 view.
num_blocks = num_pages
k_cache = torch.zeros(num_blocks, BLOCK_SIZE, 1, D_QK,
                      dtype=torch.float8_e4m3fn, device=dev)

# ---- q ----
q = torch.randn(B, S_Q, H_Q, D_QK, dtype=torch.bfloat16, device=dev) * 0.1

# ---- valid swa indices: point every query at TOPK real rows ----
# indices_in_kvcache[i][j][k] = page_block_index * block_size + offset
max_valid = num_rows - 1
idx = torch.randint(0, max_valid, (B, S_Q, TOPK), dtype=torch.int32, device=dev)
topk_length = torch.full((B,), TOPK, dtype=torch.int32, device=dev)

attn_sink = torch.zeros(H_Q, dtype=torch.float32, device=dev)

sched_meta = flash_mla.get_mla_metadata()[0]

sm_scale = D_QK ** -0.5


def one_call():
    return flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        head_dim_v=D_V,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=sched_meta,
        softmax_scale=sm_scale,
        is_fp8_kvcache=True,
        indices=idx,
        topk_length=topk_length,
        attn_sink=attn_sink,
        **packed_kwargs,
    )[0]


# warmup + sanity
out = one_call()
torch.cuda.synchronize()
print(f"[probe] out.shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")

for _ in range(ITERS):
    one_call()
torch.cuda.synchronize()
print("[probe] done")
