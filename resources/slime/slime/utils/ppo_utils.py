# Modified by MAI-Gym contributors.

from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    if importance_ratio is not None:
        kl = importance_ratio * kl

    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        masked_kl = full_kl_response * full_mask
        token_level_rewards = -kl_coef * masked_kl
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def compute_length_adaptive_lambd(
    response_lengths,
    alpha: float,
    device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if alpha is None or alpha <= 0.0:
        raise ValueError(
            f"compute_length_adaptive_lambd requires alpha > 0; got {alpha}. "
            "Pass lambd_length_adaptive_alpha=0 at the caller to disable."
        )
    L = torch.tensor(response_lengths, dtype=torch.float32, device=device)
    lambd = 1.0 - 1.0 / (alpha * L)
    return lambd.clamp_(min=0.0, max=1.0).to(dtype=dtype)


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
    max_seq_lens=None,
    qkv_format: str = "thd",
    terminal_rewards=None,
    lambd_critic: float | None = None,
    lambd_length_adaptive_alpha: float = 0.0,
):

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for idx, (total_len, resp_len, v, r) in enumerate(
                zip(total_lengths, response_lengths, values_list, rewards_list, strict=False)
            ):
                max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
                full_v = all_gather_with_cp(v, total_len, resp_len, qkv_format, max_seq_len)
                full_r = all_gather_with_cp(r, total_len, resp_len, qkv_format, max_seq_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]
            if terminal_rewards is not None and L > 0:
                full_rewards[i, L - 1] += terminal_rewards[i]

        if (
            terminal_rewards is not None
            and mpu.get_context_parallel_rank() == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        ):
            _tr = [float(r) for r in terminal_rewards]
            _nz = sum(1 for r in _tr if r != 0.0)
            print(
                f"[ppo] terminal reward placed on {len(_tr)} samples: {_nz} nonzero, "
                f"mean={sum(_tr) / max(len(_tr), 1):.4f}, max_len={max_len}",
                flush=True,
            )

        lambd_per_sample = None
        if lambd_length_adaptive_alpha and lambd_length_adaptive_alpha > 0.0:
            lambd_per_sample = compute_length_adaptive_lambd(
                response_lengths,
                lambd_length_adaptive_alpha,
                device=device,
                dtype=dtype,
            )

        if not chunked or lambd_per_sample is not None:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
                lambd_per_sample=lambd_per_sample,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        if lambd_critic is not None and lambd_critic != lambd:
            if chunked:
                _, full_returns = chunked_gae(
                    rewards=full_rewards,
                    values=full_values,
                    gamma=gamma,
                    lambd=lambd_critic,
                )
            else:
                _, full_returns = vanilla_gae(
                    rewards=full_rewards,
                    values=full_values,
                    gamma=gamma,
                    lambd=lambd_critic,
                )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for idx, (total_len, resp_len, adv_row, ret_row) in enumerate(
                zip(
                    total_lengths,
                    response_lengths,
                    full_advantages,
                    full_returns,
                    strict=False,
                )
            ):
                max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
                adv_full = adv_row
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len, qkv_format, max_seq_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len, qkv_format, max_seq_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    lambd_per_sample: torch.Tensor | None = None,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    if lambd_per_sample is not None:
        assert lambd_per_sample.shape == (B,), (
            f"lambd_per_sample must be [B={B}], got {tuple(lambd_per_sample.shape)}"
        )
        w_vec = (gamma * lambd_per_sample).to(device=device, dtype=dtype)
    else:
        w_vec = None

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        if w_vec is None:
            lastgaelam = delta + gamma * lambd * lastgaelam
        else:
            lastgaelam = delta + w_vec * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)
    full_returns = full_advantages + values
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):

    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])

    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]

    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy: bool = False, chunk_size: int = -1):
    logits = logits.contiguous()
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            logits_chunks = logits.chunk(num_chunks, dim=0)
            tokens_chunks = tokens.chunk(num_chunks, dim=0)

            if with_entropy:
                entropys = []
                for logits_chunk in logits_chunks:
                    entropy_input = logits_chunk.clone()
                    entropys.append(compute_entropy_from_logits(entropy_input, tp_group))
                entropy = torch.cat(entropys, dim=0)

            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
        else:
            if with_entropy:
                entropy_input = logits.clone()
                entropy = compute_entropy_from_logits(entropy_input, tp_group)

            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy
