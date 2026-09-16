"""Reproducible legacy / packed attention comparison on SM90.

Run each revision in a separate process using --impl-root. Conversion is outside
timing. Graph samples amortize Python launch gaps; these are not serving QPS/P95.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def measure(fn, *, graph, iterations, repeats):
    for _ in range(40):
        fn()
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            for _ in range(iterations):
                fn()
        run = captured.replay
        for _ in range(3):
            run()
    else:
        def run():
            for _ in range(iterations):
                fn()
    samples = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--topk", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--s-q", type=int, default=2)
    parser.add_argument("--page-slots", type=int, default=128)
    parser.add_argument("--cache-tokens", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--graph-only", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(args.impl_root), str(args.impl_root / "tests")]
    import flash_mla
    import quant
    from test_sm90_mixed_main_kv import MAIN_LAYOUT, _reference

    assert torch.cuda.get_device_capability()[0] == 9
    assert args.cache_tokens % args.page_slots == 0
    result = {
        "commit": subprocess.check_output(
            ["git", "-C", str(args.impl_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "module": str(flash_mla.__file__),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": [],
    }
    for heads in args.heads:
        for topk in args.topk:
            torch.manual_seed(20260916 + heads + topk)
            b, s_q, p = args.batch, args.s_q, args.page_slots
            q = (torch.randn(b, s_q, heads, 512, device="cuda") / 10).bfloat16()
            swa = (torch.randn(args.cache_tokens // 64, 64, 1, 512, device="cuda") / 10).bfloat16()
            main_kv = (torch.randn(args.cache_tokens // p, p, 512, device="cuda") / 10).bfloat16()
            swa_cache = quant.quantize_k_cache(swa, quant.KVCacheLayout.V4_FP8Sparse)
            packed = quant.quantize_dsv41_main_kv(main_kv)
            dequant = quant.dequantize_dsv41_main_kv(packed, p)
            legacy = quant.quantize_k_cache(main_kv.unsqueeze(2), quant.KVCacheLayout.V4_FP8Sparse)
            indices = [
                torch.randint(args.cache_tokens, (b, s_q, topk), device="cuda", dtype=torch.int32)
                for _ in range(2)
            ]
            lengths = torch.full((b,), topk, device="cuda", dtype=torch.int32)
            sink = torch.randn(heads, device="cuda", dtype=torch.float32)
            direct_meta, _ = flash_mla.get_mla_metadata()
            legacy_meta, legacy_splits = flash_mla.get_mla_metadata()

            def direct():
                return flash_mla.flash_mla_with_mixed_kvcache(
                    q=q, swa_cache=swa_cache, swa_indices=indices[0],
                    main_cache_bytes=packed, main_indices=indices[1],
                    swa_layout="V4", main_layout=MAIN_LAYOUT,
                    main_page_slots=p, main_page_bytes=p * 384,
                    head_dim_v=512, tile_scheduler_metadata=direct_meta,
                    softmax_scale=512**-0.5, attn_sink=sink,
                    swa_topk_length=lengths, main_topk_length=lengths,
                )

            def legacy_fn():
                return flash_mla.flash_mla_with_kvcache(
                    q=q, k_cache=swa_cache, block_table=None, cache_seqlens=None,
                    head_dim_v=512, tile_scheduler_metadata=legacy_meta,
                    num_splits=legacy_splits, softmax_scale=512**-0.5,
                    causal=False, is_fp8_kvcache=True, indices_in_kvcache=indices[0],
                    attn_sink=sink, extra_k_cache=legacy,
                    extra_indices_in_kvcache=indices[1], topk_length=lengths,
                    extra_topk_length=lengths,
                )

            out, lse = direct()
            out_ref, lse_ref = _reference(
                q, quant.dequantize_k_cache(swa_cache, quant.KVCacheLayout.V4_FP8Sparse).squeeze(2),
                indices[0], lengths, dequant, indices[1], lengths, sink, 512**-0.5,
            )
            torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=2.01 / 128)
            torch.testing.assert_close(lse, lse_ref, atol=1e-6, rtol=8.01 / 65536)
            case = {
                "heads": heads, "topk_each": topk,
                "max_abs_error": (out.float() - out_ref.float()).abs().max().item(),
                "main_packed_bytes": packed.numel() * packed.element_size(),
                "main_legacy_bytes": legacy.numel() * legacy.element_size(),
            }
            for graph in ([True] if args.graph_only else [False, True]):
                mode = "graph" if graph else "eager"
                # Alternate measurement order across shapes.
                fns = [("legacy", legacy_fn), ("direct", direct)]
                if (heads == 128) ^ (topk > 64):
                    fns.reverse()
                for label, fn in fns:
                    case[f"{label}_{mode}"] = measure(
                        fn, graph=graph, iterations=args.iterations, repeats=args.repeats
                    )
            result["cases"].append(case)
            print(json.dumps(case), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            if args.trace:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for label, fn in [("legacy", legacy_fn), ("direct", direct)]:
                        with torch.profiler.record_function(label):
                            for _ in range(3):
                                fn()
                    torch.cuda.synchronize()
                trace = args.trace.with_name(f"{args.trace.stem}-h{heads}-k{topk}.json")
                trace.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(trace))


if __name__ == "__main__":
    main()
