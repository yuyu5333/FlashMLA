import pytest
import torch

import flash_mla
import quant


MAIN_LAYOUT = "DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1"


def _gather_scope(
    cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s_q, topk = indices.shape
    flat = cache.reshape(-1, 512)
    invalid = (indices < 0) | (indices >= flat.shape[0])
    invalid |= (
        torch.arange(topk, device=indices.device).view(1, 1, -1)
        >= topk_length.view(b, 1, 1)
    )
    safe_indices = indices.masked_fill(invalid, 0)
    gathered = flat.index_select(0, safe_indices.reshape(-1)).view(
        b, s_q, topk, 512
    )
    return gathered, invalid


def _reference(
    q: torch.Tensor,
    swa: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_length: torch.Tensor,
    main: torch.Tensor,
    main_indices: torch.Tensor,
    main_topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    swa_gathered, swa_invalid = _gather_scope(
        swa, swa_indices, swa_topk_length
    )
    main_gathered, main_invalid = _gather_scope(
        main, main_indices, main_topk_length
    )
    gathered = torch.cat((swa_gathered, main_gathered), dim=2).float()
    invalid = torch.cat((swa_invalid, main_invalid), dim=2)

    logits = torch.matmul(q.float(), gathered.transpose(-1, -2)) * sm_scale
    logits.masked_fill_(invalid.unsqueeze(2), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    weights = torch.exp(logits - lse.unsqueeze(-1))
    out = torch.matmul(weights, gathered)
    out *= (
        1.0
        / (1.0 + torch.exp(attn_sink.view(1, 1, -1) - lse))
    ).unsqueeze(-1)

    empty = torch.isneginf(lse)
    out.masked_fill_(empty.unsqueeze(-1), 0.0)
    lse.masked_fill_(empty, float("inf"))
    return out.to(torch.bfloat16), lse.transpose(1, 2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 mixed-cache kernel requires Hopper",
)
@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("page_slots", [128, 256])
def test_sm90_mixed_main_kv_matches_reference(
    num_heads: int, page_slots: int
) -> None:
    torch.manual_seed(20260916 + num_heads + page_slots)
    device = torch.device("cuda")
    b, s_q, topk = 2, 2, 64
    swa_page_slots = 64
    sm_scale = 512**-0.5

    q = (torch.randn(b, s_q, num_heads, 512, device=device) / 10).to(
        torch.bfloat16
    )
    swa = (torch.randn(2, swa_page_slots, 1, 512, device=device) / 10).to(
        torch.bfloat16
    )
    main = (torch.randn(2, page_slots, 512, device=device) / 10).to(
        torch.bfloat16
    )
    swa_cache = quant.quantize_k_cache(
        swa, quant.KVCacheLayout.V4_FP8Sparse
    )
    swa_dequant = quant.dequantize_k_cache(
        swa_cache, quant.KVCacheLayout.V4_FP8Sparse
    ).squeeze(2)
    main_cache = quant.quantize_dsv41_main_kv(main)
    main_dequant = quant.dequantize_dsv41_main_kv(
        main_cache, page_slots
    )

    swa_indices = torch.randint(
        0, swa.numel() // 512, (b, s_q, topk), dtype=torch.int32, device=device
    )
    main_indices = torch.randint(
        0, main.numel() // 512, (b, s_q, topk), dtype=torch.int32, device=device
    )
    swa_indices[..., 5] = swa_indices[..., 4]
    main_indices[..., 7] = main_indices[..., 6]
    swa_indices[..., -1] = -1
    main_indices[..., -2:] = -1
    swa_topk_length = torch.tensor([57, 64], dtype=torch.int32, device=device)
    main_topk_length = torch.tensor([61, 43], dtype=torch.int32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)

    sched_meta, _ = flash_mla.get_mla_metadata()
    out, lse = flash_mla.flash_mla_with_mixed_kvcache(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        main_cache_bytes=main_cache,
        main_indices=main_indices,
        swa_layout="V4",
        main_layout=MAIN_LAYOUT,
        main_page_slots=page_slots,
        main_page_bytes=page_slots * 384,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta,
        softmax_scale=sm_scale,
        attn_sink=attn_sink,
        swa_topk_length=swa_topk_length,
        main_topk_length=main_topk_length,
    )
    out_ref, lse_ref = _reference(
        q,
        swa_dequant,
        swa_indices,
        swa_topk_length,
        main_dequant,
        main_indices,
        main_topk_length,
        attn_sink,
        sm_scale,
    )

    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=2.01 / 128)
    torch.testing.assert_close(lse, lse_ref, atol=1e-6, rtol=8.01 / 65536)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 mixed-cache kernel requires Hopper",
)
def test_sm90_mixed_main_kv_capabilities() -> None:
    capabilities = flash_mla.get_mla_capabilities()
    assert capabilities["mixed_kvcache_api_version"] == 1
    assert capabilities["mixed_kvcache_supported"]
    assert ["V4", MAIN_LAYOUT] in capabilities["supported_mixed_layout_pairs"]
    assert capabilities["supported_num_heads"] == [64, 128]
    assert capabilities["supported_main_page_slots"] == [128, 256]
