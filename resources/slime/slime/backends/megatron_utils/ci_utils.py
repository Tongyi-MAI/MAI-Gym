# Modified by MAI-Gym contributors.

import logging
from collections.abc import Sequence

from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def check_mtp_only_grad(model: Sequence[DDP], step_id: int) -> None:
    non_mtp_nonzero_grads = []
    mtp_nonzero_grads = []

    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue

            grad_norm = grad.abs().max().item()
            is_mtp = ".mtp." in name

            if is_mtp:
                if grad_norm > 0:
                    mtp_nonzero_grads.append((name, grad_norm))
            else:
                if grad_norm > 0:
                    non_mtp_nonzero_grads.append((name, grad_norm))

    logger.info(
        f"[CI MTP Grad Check] Step {step_id}: "
        f"MTP params with non-zero grad: {len(mtp_nonzero_grads)}, "
        f"non-MTP params with non-zero grad: {len(non_mtp_nonzero_grads)}"
    )

    if non_mtp_nonzero_grads:
        for name, grad_norm in non_mtp_nonzero_grads[:5]:
            logger.error(f"[CI MTP Grad Check] Non-MTP param with non-zero grad: {name}, max_grad={grad_norm}")

    assert len(non_mtp_nonzero_grads) == 0, (
        f"Expected all non-MTP parameters to have zero gradients, "
        f"but found {len(non_mtp_nonzero_grads)} with non-zero gradients. "
        f"First few: {non_mtp_nonzero_grads[:5]}"
    )

    assert len(mtp_nonzero_grads) > 0, (
        "Expected MTP parameters to have non-zero gradients, but all were zero. "
        "This may indicate the MTP loss is not being computed."
    )


def check_mtp_loss(mtp_loss: float, max_mtp_loss: float = 1.0) -> None:
    assert mtp_loss < max_mtp_loss, (
        f"MTP loss {mtp_loss} exceeds maximum allowed value {max_mtp_loss}. "
        "This may indicate an issue with MTP training."
    )
