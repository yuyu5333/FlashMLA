"""Long-tile, empty, invalid-index and graph replay validation for the loader."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import flash_mla
import quant
from test_sm90_mixed_main_kv import MAIN_LAYOUT, _reference


@torch.inference_mode()
def main():
    checks = []
    for heads in [64, 128]:
        for p in [128, 256]:
            for topk in [64, 512, 2048]:
                torch.manual_seed(heads + p + topk)
                b, s_q, n = 3, 2, p * 8
                q = (torch.randn(b, s_q, heads, 512, device="cuda") / 10).bfloat16()
                main_kv = (torch.randn(8, p, 512, device="cuda") / 10).bfloat16()
                swa = (torch.randn(n // 64, 64, 1, 512, device="cuda") / 10).bfloat16()
                packed = quant.quantize_dsv41_main_kv(main_kv)
                swa_cache = quant.quantize_k_cache(swa, quant.KVCacheLayout.V4_FP8Sparse)
                main_ref = quant.dequantize_dsv41_main_kv(packed, p)
                swa_ref = quant.dequantize_k_cache(swa_cache, quant.KVCacheLayout.V4_FP8Sparse).squeeze(2)
                indices = [torch.randint(n, (b, s_q, topk), device="cuda", dtype=torch.int32) for _ in range(2)]
                for x in indices:
                    x[..., 0] = -1
                    x[..., 1] = n
                    x[..., 2] = torch.iinfo(torch.int32).max
                    x[..., 4] = x[..., 3]
                    x[2] = -1
                # Empty Main, partial tile, full tile, and entirely masked request.
                swa_len = torch.tensor([topk, topk - 13, 0], device="cuda", dtype=torch.int32)
                main_len = torch.tensor([0, topk - 7, topk], device="cuda", dtype=torch.int32)
                sink = torch.randn(heads, device="cuda")
                meta, _ = flash_mla.get_mla_metadata()

                def run():
                    return flash_mla.flash_mla_with_mixed_kvcache(
                        q, swa_cache, indices[0], packed, indices[1],
                        swa_layout="V4", main_layout=MAIN_LAYOUT,
                        main_page_slots=p, main_page_bytes=p * 384,
                        head_dim_v=512, tile_scheduler_metadata=meta,
                        softmax_scale=512**-0.5, attn_sink=sink,
                        swa_topk_length=swa_len, main_topk_length=main_len,
                    )

                def verify(out, lse):
                    expected = _reference(
                        q, swa_ref, indices[0], swa_len, main_ref, indices[1],
                        main_len, sink, 512**-0.5,
                    )
                    torch.testing.assert_close(out, expected[0], atol=1e-3, rtol=2.01 / 128)
                    torch.testing.assert_close(lse, expected[1], atol=1e-6, rtol=8.01 / 65536)

                verify(*run())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out, lse = run()
                # Same allocations, different input contents: detect stale replay data.
                q.neg_()
                indices[1][1].copy_(indices[1][1].roll(5, -1))
                graph.replay()
                torch.cuda.synchronize()
                verify(out, lse)
                checks.append({"heads": heads, "page_slots": p, "topk": topk, "graph": "passed"})
    print(json.dumps({"passed": len(checks), "cases": checks}, indent=2))


if __name__ == "__main__":
    main()
