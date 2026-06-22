"""ROCm/MI300 dsa_sparse_attention solution.

DeepSeek sparse MLA attention. Wraps AITER's `unified_attention_sparse_mla` (Triton, gfx942).

Per query token: gather the topk KV tokens given by absolute indices `sparse_indices`
(= page_idx*page_size + offset, -1 = padding), compute MLA logits
    logits = q_nope @ ckv.T + q_pe @ kpe.T   (scaled by sm_scale)
softmax, then output = softmax @ ckv.

Layout for AITER: q = [q_nope(512) | q_pe(64)] -> [T,16,576];
kv = [ckv(512) | kpe(64)] -> [num_pages, page_size, 1, 576]; value = kv[..., :512].
Each query token is its own length-1 sequence (ALL_DECODE). The kernel masks -1 directly.

We return only `output` (the evaluator accepts 1 or 2 outputs; lse is optional).
"""
import torch
from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    else:
        sm_scale = float(sm_scale)

    device = q_nope.device

    # q: [T, H, 576] = [lora | rope]
    q = torch.cat([q_nope, q_pe], dim=-1).contiguous()

    # kv: [num_pages, page_size, 1, 576] = [lora | rope]
    kv = torch.cat([ckv_cache, kpe_cache], dim=-1).unsqueeze(2).contiguous()

    sparse_indices = sparse_indices.to(torch.int32)

    # each query token is a length-1 sequence
    cu_seqlens_q = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    max_seqlen_q = 1
    seqused_k = (sparse_indices != -1).sum(dim=1).to(torch.int32)
    max_seqlen_k = int(seqused_k.max().item()) if num_tokens > 0 else 0

    # block_table is not used for indexing (the kernel maps topk_pos directly), but a
    # tensor with a valid stride(0) is required.
    block_table = torch.zeros((num_tokens, 1), dtype=torch.int32, device=device)

    output = torch.empty(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )

    unified_attention_sparse_mla(
        q,
        kv,
        output,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        sm_scale,
        sparse_indices,
        block_table,
        head_dim_ckv,  # kv_lora_rank
    )

    return (output,)
