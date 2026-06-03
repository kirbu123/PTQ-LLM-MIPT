import math
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
    "apply_dptq_refinement",
    "quantize_weight",
]


def _as_weight_matrix(module: torch.nn.Module) -> torch.Tensor | None:
    if module is None or not hasattr(module, "weight") or module.weight is None:
        return None
    if module.weight.is_meta:
        return None

    W = module.weight.detach()
    match module:
        case torch.nn.Conv2d():
            W = W.flatten(1)
        case transformers.Conv1D():
            W = W.t()

    return W.to(dtype=GPTQ_PRECISION)


def _lambda_value(lam_tensor: torch.Tensor | float | None, idx: int, device) -> torch.Tensor:
    if lam_tensor is None:
        return torch.zeros((), device=device, dtype=GPTQ_PRECISION)
    if torch.is_tensor(lam_tensor):
        if lam_tensor.numel() == 0:
            return torch.zeros((), device=device, dtype=GPTQ_PRECISION)
        lam = lam_tensor[min(idx, lam_tensor.numel() - 1)].to(device=device)
    else:
        lam = torch.tensor(float(lam_tensor), device=device)

    # The DPTQ objective assumes lambda_t >= 0.  Clamp here so a noisy optimizer
    # cannot turn the downstream penalty into a reward.
    return lam.to(dtype=GPTQ_PRECISION).clamp_min(0)


def build_downstream_q(
    num_rows: int,
    next_modules: list[torch.nn.Module | None] | None,
    lam_tensor: torch.Tensor | float | None,
    device: torch.device,
    kernel_mode: str = "default",
) -> torch.Tensor | None:
    """
    Build the downstream metric from Theorem 1:

        Q_k = I + sum_{t=1}^k lambda_t P_t^T P_t,
        P_t = W_t W_{t-1} ... W_1.

    Q_k acts on the output dimension of the current layer, i.e. on rows of W.
    Incompatible downstream layers are skipped instead of forcing an invalid
    product.
    """
    if not next_modules:
        return None

    Q = torch.eye(num_rows, device=device, dtype=GPTQ_PRECISION)
    P = torch.eye(num_rows, device=device, dtype=GPTQ_PRECISION)
    used_any = False

    for idx, module_next in enumerate(next_modules):
        W_next = _as_weight_matrix(module_next)
        if W_next is None:
            continue
        W_next = W_next.to(device=device)

        if W_next.shape[1] != P.shape[0]:
            logger.warning(
                "Skipping DPTQ downstream layer {}: cannot multiply shapes "
                "{} and {} for P_t = W_t ... W_1",
                idx,
                tuple(W_next.shape),
                tuple(P.shape),
            )
            continue

        P = W_next @ P
        lam = _lambda_value(lam_tensor, idx, device)
        PtP = P.t() @ P

        # Default mode is the exact paper formula.  Non-default kernel modes
        # preserve the old pipeline hook for optional matrix smoothing.
        if kernel_mode != "default":
            PtP = apply_conv(PtP, mode=kernel_mode)

        Q = Q + lam * PtP
        used_any = True

    return Q if used_any else None


def apply_dptq_refinement(
    H: torch.Tensor,
    num_rows: int,
    next_modules: list[torch.nn.Module | None] | None = None,
    lam_tensor: torch.Tensor | float | None = None,
    kernel_mode: str = "default",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Convert the DPTQ Hessian formula into the GPTQ-compatible Hessian.

    The exact DPTQ Hessian is

        H_full = 2 (X X^T kron Q_k),

    where Q_k weights row/output errors.  The existing GPTQ loop accepts only a
    d_in x d_in input Hessian.  For compatible square projections we use the
    paper's multiplicative surrogate

        H_X = L L^T,  H_X^opt = L Q_k L^T,

    which keeps the Hessian symmetric positive definite for Cholesky inversion.
    If Q_k has a different size from H_X, the exact Kronecker Hessian cannot be
    represented in this framework, so we leave H_X unchanged and still return Q_k
    for DPTQ loss accounting and diagnostics.
    """
    H_init = H.clone()
    Q = build_downstream_q(
        num_rows=num_rows,
        next_modules=next_modules,
        lam_tensor=lam_tensor,
        device=H.device,
        kernel_mode=kernel_mode,
    )
    if Q is None:
        return H, H_init, None

    if Q.shape != H.shape:
        logger.warning(
            "DPTQ Q_k has shape {}, but GPTQ input Hessian has shape {}. "
            "Keeping H_X unchanged because the full 2(XX^T kron Q_k) Hessian "
            "does not fit the row-wise GPTQ interface.",
            tuple(Q.shape),
            tuple(H.shape),
        )
        return H, H_init, Q

    try:
        H_sym = (H + H.t()) / 2
        Q_sym = (Q + Q.t()) / 2

        # H_X = L L^T and H_X^opt = L Q_k L^T.  This is the SPD congruence
        # refinement from the DPTQ algorithmic section.
        jitter = torch.finfo(H.dtype).eps * torch.mean(torch.diag(H_sym)).abs()
        diag = torch.arange(H_sym.shape[0], device=H_sym.device)
        H_sym[diag, diag] += jitter
        Q_sym[diag, diag] += jitter
        L = torch.linalg.cholesky(H_sym)
        H_opt = L @ Q_sym @ L.t()
        H_opt = (H_opt + H_opt.t()) / 2
        return H_opt, H_init, Q_sym
    except torch._C._LinAlgError:
        logger.warning("Failed to build DPTQ refined Hessian; falling back to GPTQ H_X")
        return H_init, H_init, Q


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
    Quantize a module weight with the GPTQ block loop and a DPTQ Hessian.

    DPTQ objective:

        L(What) = ||W X - What X||_F^2
                  + sum_t lambda_t ||P_t (W - What) X||_F^2.

    The theorem gives H_full = 2(XX^T kron Q_k).  This implementation builds
    Q_k exactly from downstream weights and applies the GPTQ-compatible
    refinement H_X^opt = L Q_k L^T when dimensions permit.
    """
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()

    match module:
        case torch.nn.Conv2d():
            W = W.flatten(1)
        case transformers.Conv1D():
            W.transpose_(0, 1)
    W = W.to(dtype=GPTQ_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    H = hessians_dict[module]
    if next_modules is not None and lam_tensor is not None:
        H, H_init, Q_dptq = apply_dptq_refinement(
            H=H,
            num_rows=num_rows,
            next_modules=next_modules,
            lam_tensor=lam_tensor,
            kernel_mode=kernel_mode,
        )

        if save_dir is not None:
            save_matrix_results(module, f"{name}_H_init", M=H_init, save_dir=save_dir)
            save_matrix_results(module, f"{name}_H_dptq", M=H, save_dir=save_dir)
            if Q_dptq is not None:
                save_matrix_results(module, f"{name}_Q_dptq", M=Q_dptq, save_dir=save_dir)
    else:
        Q_dptq = None

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
            "Failed to invert DPTQ hessian due to numerical instability. "
            "Falling back to round-to-nearest for this module."
        )
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    W_adj = torch.zeros_like(W)

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
                raise ValueError(f"Quantization strategy is not supported for DPTQ: {strategy}")

            Q1[:, i] = q
            W1_adj[:, i] = w

            # GPTQ loss is ||e_j||_2^2 / d^2.  DPTQ replaces the row metric by
            # e_j^T Q_k e_j / d^2, matching ||P_t E X||_F^2 in the objective.
            col_err = w - q
            if Q_dptq is not None and Q_dptq.shape[0] == col_err.shape[0]:
                Q_loss = Q_dptq.to(device=col_err.device, dtype=col_err.dtype)
                dptq_col_loss = col_err @ Q_loss @ col_err
                losses1[:, i] = dptq_col_loss / (num_rows * d**2)
            else:
                losses1[:, i] = col_err**2 / d**2

            err1 = col_err / d
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
