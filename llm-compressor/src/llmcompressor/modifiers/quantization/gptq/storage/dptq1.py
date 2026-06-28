from copy import copy

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from compressed_tensors.utils import update_offload_parameter
from loguru import logger

from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    GPTQ_PRECISION,
    accumulate_hessian,
    accumulate_hessian_next_reg,
    make_empty_hessian,
)
from llmcompressor.modifiers.quantization.gptq.utils import save_matrix_results
from llmcompressor.modifiers.utils import SPARSITY_THRESHOLD
from llmcompressor.modifiers.utils.kernels import apply_conv
from llmcompressor.observers.base import Observer
from llmcompressor.pytorch.utils.helpers import tensor_sparsity

__all__ = [
    "make_empty_hessian",
    "accumulate_hessian",
    "accumulate_hessian_next_reg",
    "build_downstream_q",
    "apply_dptq_refinement",
    "quantize_weight",
]


def _as_weight_matrix(module: torch.nn.Module | None) -> torch.Tensor | None:
    if module is None or not hasattr(module, "weight") or module.weight is None:
        return None
    if module.weight.is_meta:
        return None

    weight = module.weight.detach()
    match module:
        case torch.nn.Conv2d():
            weight = weight.flatten(1)
        case transformers.Conv1D():
            weight = weight.t()

    return weight.to(dtype=GPTQ_PRECISION)


def _lambda_value(
    lam_tensor: torch.Tensor | float | None, idx: int, device: torch.device
) -> torch.Tensor:
    if lam_tensor is None:
        return torch.zeros((), dtype=GPTQ_PRECISION, device=device)

    if torch.is_tensor(lam_tensor):
        if lam_tensor.numel() == 0:
            return torch.zeros((), dtype=GPTQ_PRECISION, device=device)
        lam = lam_tensor[min(idx, lam_tensor.numel() - 1)].detach().to(device=device)
    else:
        lam = torch.tensor(float(lam_tensor), device=device)

    # Eq. (2)-(3): lambda_t >= 0
    return lam.to(dtype=GPTQ_PRECISION).clamp_min(0)


def build_downstream_q(
    num_rows: int,
    next_modules: list[torch.nn.Module | None] | None,
    lam_tensor: torch.Tensor | float | None,
    device: torch.device,
    kernel_mode: str = "default",
) -> torch.Tensor | None:
    """
    Build Q_k from Sections 3.1 / 3.2:

        P_0 = I,  P_t = W_t ... W_1
        Q_k = I + sum_{t=1}^k lambda_t P_t^T P_t

    Q_k acts on output directions of the current layer (row space of W).
    """
    if not next_modules:
        return None

    q_k = torch.eye(num_rows, device=device, dtype=GPTQ_PRECISION)
    p_t = torch.eye(num_rows, device=device, dtype=GPTQ_PRECISION)
    used_any = False

    for idx, module_next in enumerate(next_modules):
        weight_next = _as_weight_matrix(module_next)
        if weight_next is None:
            continue
        weight_next = weight_next.to(device=device)

        # Ensure P_t = W_t ... W_1 is well-defined.
        if weight_next.shape[1] != p_t.shape[0]:
            logger.warning(
                "Skipping DPTQ downstream layer {}: incompatible shapes {} and {}",
                idx,
                tuple(weight_next.shape),
                tuple(p_t.shape),
            )
            continue

        p_t = weight_next @ p_t
        lambda_t = _lambda_value(lam_tensor, idx, device)
        if torch.all(lambda_t == 0):
            continue

        p_t_t_p_t = p_t.t() @ p_t

        # Keep optional legacy hook for experiments with custom kernels.
        if kernel_mode != "default":
            p_t_t_p_t = apply_conv(p_t_t_p_t, mode=kernel_mode)

        if p_t_t_p_t.shape != q_k.shape:
            logger.warning(
                "Skipping DPTQ downstream layer {} after projection: expected {} got {}",
                idx,
                tuple(q_k.shape),
                tuple(p_t_t_p_t.shape),
            )
            continue

        q_k = q_k + lambda_t * p_t_t_p_t
        used_any = True

    return q_k if used_any else None


def apply_dptq_refinement(
    h_x: torch.Tensor, q_k: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Build GPTQ-compatible Hessian from Section 3.2 factorization.

    Theorem 3.1 gives:
        H_k = 2 (X X^T \otimes Q_k).
    GPTQ expects an SPD input-space Hessian only. We therefore keep GPTQ's
    block loop and inject downstream sensitivity through an SPD congruence:
        H_X = L L^T,   H_X^opt = L Q_k L^T.
    """
    h_init = h_x.clone()
    if q_k is None:
        return h_x, h_init, None

    if q_k.shape != h_x.shape:
        logger.warning(
            "DPTQ Q_k shape {} is incompatible with H_X shape {}; "
            "falling back to unrefined GPTQ Hessian",
            tuple(q_k.shape),
            tuple(h_x.shape),
        )
        return h_x, h_init, q_k

    try:
        h_sym = (h_x + h_x.t()) / 2
        q_sym = (q_k + q_k.t()) / 2

        diag = torch.arange(h_sym.shape[0], device=h_sym.device)
        eps = torch.finfo(h_sym.dtype).eps
        h_jitter = eps * (torch.mean(torch.diag(h_sym)).abs() + 1.0)
        q_jitter = eps * (torch.mean(torch.diag(q_sym)).abs() + 1.0)

        h_sym = h_sym.clone()
        q_sym = q_sym.clone()
        h_sym[diag, diag] += h_jitter
        q_sym[diag, diag] += q_jitter

        chol_h = torch.linalg.cholesky(h_sym)
        h_opt = chol_h @ q_sym @ chol_h.t()
        h_opt = (h_opt + h_opt.t()) / 2

        return h_opt, h_init, q_sym
    except torch._C._LinAlgError:
        logger.warning("Failed DPTQ Hessian refinement; using unrefined GPTQ Hessian")
        return h_init, h_init, q_k


def quantize_weight(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessians_dict: dict[torch.nn.Module, torch.Tensor],
    blocksize: int = 128,
    percdamp: float = 0.01,
    next_modules: list[torch.nn.Module | None] | None = None,
    lam_tensor: torch.Tensor | float | None = None,
    kernel_mode: str = "default",
    name: str = None,
    save_dir: str = None,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    GPTQ quantization loop with DPTQ next-layer Hessian refinement.

    The GPTQ block-wise mechanics remain unchanged. The only upgrade is replacing
    the local Hessian H_X with a downstream-aware refined Hessian before inverse-
    Cholesky updates.
    """
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()

    # Standardize shape and dtype.
    match module:
        case torch.nn.Conv2d():
            W = W.flatten(1)
        case transformers.Conv1D():
            W.transpose_(0, 1)
    W = W.to(dtype=GPTQ_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    # Build refined Hessian from Theorem 3.1 factors.
    H = hessians_dict[module]
    H_init = H.clone()
    Q_k = None
    if next_modules is not None and lam_tensor is not None:
        Q_k = build_downstream_q(
            num_rows=num_rows,
            next_modules=next_modules,
            lam_tensor=lam_tensor,
            device=H.device,
            kernel_mode=kernel_mode,
        )
        H, H_init, Q_k = apply_dptq_refinement(H, Q_k)

        if save_dir is not None and Q_k is not None:
            save_matrix_results(module, f"{name}_H_init", M=H_init, save_dir=save_dir)
            save_matrix_results(module, f"{name}_H_dptq", M=H, save_dir=save_dir)
            save_matrix_results(module, f"{name}_Q_k", M=Q_k, save_dir=save_dir)

    observer = Observer.load_from_registry(
        "memoryless_minmax",
        base_name="weight",
        args=quant_args,
        module=module,
    )

    if strategy == QuantizationStrategy.GROUP:
        g_idx = (
            torch.arange(num_columns, device=W.device, dtype=torch.int)
            // quant_args.group_size
        )

        if actorder == ActivationOrdering.GROUP:
            W, H, perm = _apply_activation_ordering(W, H)
            update_offload_parameter(module, "weight_g_idx", g_idx)
            scale, zero_point = observer(W)
        elif actorder == ActivationOrdering.WEIGHT:
            scale, zero_point = observer(W)
            W, H, perm = _apply_activation_ordering(W, H)
            g_idx = g_idx[perm]
        else:
            scale, zero_point = observer(W)
    else:
        scale, zero_point = observer(W)

    sparsity = tensor_sparsity(W)
    preserve_zeros = sparsity >= SPARSITY_THRESHOLD
    W_nz_mask = (
        (~torch.isclose(W, torch.zeros(1, device=W.device).float())).float()
        if preserve_zeros
        else None
    )

    losses = torch.zeros(num_rows, device=module.weight.device)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    try:
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
    except torch._C._LinAlgError:
        logger.warning(
            "Failed to invert DPTQ Hessian due to numerical instability. "
            "Falling back to round-to-nearest for this module."
        )
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    W_adj = torch.zeros_like(W)

    # Same block-wise GPTQ update, now executed with refined Hessian H_X^opt.
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        W1_adj = torch.zeros_like(W1)

        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if preserve_zeros:
            W1_nz_mask = W_nz_mask[:, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()

            if strategy == QuantizationStrategy.TENSOR:
                q = fake_quantize(q, scale, zero_point, quant_args)
            elif strategy == QuantizationStrategy.CHANNEL:
                q = fake_quantize(q, scale[:, 0], zero_point[:, 0], quant_args)
            elif strategy == QuantizationStrategy.GROUP:
                column_idx = i1 + i
                group_index = g_idx[column_idx]

                altered_qargs = copy(quant_args)
                altered_qargs.strategy = QuantizationStrategy.CHANNEL
                q = fake_quantize(
                    q,
                    scale[:, group_index],
                    zero_point[:, group_index],
                    altered_qargs,
                )
            else:
                raise ValueError(
                    f"Quantization strategy is not supported for DPTQ: {strategy}"
                )

            Q1[:, i] = q
            W1_adj[:, i] = w
            losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            if preserve_zeros:
                W1[:, i:] -= w1_err * W1_nz_mask[:, i:]
            else:
                W1[:, i:] -= w1_err
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        W_adj[:, i1:i2] = W1_adj
        losses += torch.sum(losses1, 1) / 2

        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        if preserve_zeros:
            W[:, i2:] -= w_err * W_nz_mask[:, i2:]
        else:
            W[:, i2:] -= w_err

    has_gidx = False
    if strategy == QuantizationStrategy.GROUP:
        if actorder == ActivationOrdering.WEIGHT:
            invperm = torch.argsort(perm)
            W = W[:, invperm]
        elif actorder == ActivationOrdering.GROUP:
            invperm = torch.argsort(perm)
            W = W[:, invperm]
            g_idx = g_idx[invperm]
            has_gidx = True

    if not has_gidx:
        g_idx = None

    if isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.reshape(final_shape).to(final_dtype)

    loss = torch.sum(losses)
    return (
        loss,
        W,
        W_adj,
        scale.to(dtype=final_dtype),
        zero_point.to(dtype=quant_args.pytorch_dtype()),
        g_idx,
        H,
    )


def _apply_activation_ordering(
    W: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = torch.argsort(torch.diag(H), descending=True)
    return W[:, perm], H[perm][:, perm], perm
