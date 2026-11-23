# Modified by MAI-Gym contributors.

import math

import torch
from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward, flash_attn_varlen_func
from torch import Tensor


class _LearnableSoftmaxFlashAttnVarlen(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_offset,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        dropout_p,
        deterministic,
    ):
        out_vanilla, softmax_lse, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(window_size_left, window_size_right),
            deterministic=deterministic,
            return_attn_probs=True,
        )
        scale = torch.sigmoid(softmax_lse - softmax_offset.unsqueeze(1))
        scale_t = scale.t().contiguous()
        out_learn = (out_vanilla.float() * scale_t.unsqueeze(-1)).to(q.dtype)

        ctx.save_for_backward(
            q,
            k,
            v,
            out_vanilla,
            out_learn,
            softmax_lse,
            softmax_offset,
            scale_t,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.dropout_p = dropout_p
        ctx.deterministic = deterministic
        return out_learn

    @staticmethod
    def backward(ctx, d_out):
        (
            q,
            k,
            v,
            out_vanilla,
            out_learn,
            softmax_lse,
            softmax_offset,
            scale_t,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors

        d_out_modified = (d_out.float() * scale_t.unsqueeze(-1)).to(q.dtype)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        head_size_og = d_out_modified.size(2)
        d_out_padded = d_out_modified
        if head_size_og % 8 != 0:
            d_out_padded = torch.nn.functional.pad(d_out_modified, [0, 8 - head_size_og % 8])

        _wrapped_flash_attn_varlen_backward(
            d_out_padded.contiguous(),
            q,
            k,
            v,
            out_learn.contiguous(),
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size_left,
            ctx.window_size_right,
            0.0,
            None,
            ctx.deterministic,
            rng_state=None,
        )

        dq = dq[..., :head_size_og]
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]

        scale = scale_t.t()
        dot_product = (d_out.float() * out_vanilla.float()).sum(-1).t()
        sigmoid_grad = scale * (1 - scale)
        d_offset = -(dot_product * sigmoid_grad).sum(1)

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            d_offset,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def learnable_softmax_flash_attn_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_offset: Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dropout_p: float = 0.0,
    deterministic: bool = False,
) -> Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))
    return _LearnableSoftmaxFlashAttnVarlen.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_offset,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        dropout_p,
        deterministic,
    )
