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

from llmcompressor.modifiers.utils import SPARSITY_THRESHOLD
from llmcompressor.modifiers.utils.kernels import apply_conv
from llmcompressor.modifiers.utils.eigen_count import compute_max_eigenval
from llmcompressor.observers.base import Observer
from llmcompressor.pytorch.utils.helpers import tensor_sparsity

GPTQ_PRECISION = torch.float32

__all__ = ["make_empty_hessian", "accumulate_hessian", "accumulate_hessian_next_reg", "quantize_weight"]


def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)


# Original method with eigenvalues and eigenvectors calculation
def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor | None,
    num_samples: int,
    with_eigens: bool = True
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    num_added = inp.shape[0]

    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

    H *= num_samples / (num_samples + num_added)
    num_samples += num_added

    inp = inp.to(dtype=GPTQ_PRECISION)
    inp = math.sqrt(2 / num_samples) * inp
    H += inp.matmul(inp.t())

    # Eigenvalues and eigenvectors calculation
    eigenvalues, eigenvectors = None, None
    h = (H + H.T) / 2

    if with_eigens:
        U, S, Vh = torch.linalg.svd(h, full_matrices=False)

        eigenvalues = S
        eigenvectors = U

        # DEBUG
        # eigenvalues = torch.full((H.shape[0],), 1e-5, dtype=H.dtype, device=H.device)
        # eigenvectors = torch.full(H.shape, 1e-5, dtype=H.dtype, device=H.device)


    hessian_trace = torch.trace(H)
    eigenvalues_max = compute_max_eigenval(h)
    eigenvalues_max = torch.tensor(eigenvalues_max, dtype=torch.float32, device=H.device, requires_grad=True)

    eigens = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "eigenvalues_max": eigenvalues_max,
        "hessian_trace": hessian_trace
    }

    return H, num_samples, eigens


def accumulate_hessian_next_reg(
    inp: torch.Tensor,
    inp_next: torch.Tensor | None,
    module: torch.nn.Module,
    module_next: torch.nn.Module | None,
    H: torch.Tensor | None,
    num_samples: int,
    next_loss_lam: float,
    kernel_mode: str = 'default'
) -> tuple[torch.Tensor, int]:
    # Get device from H
    device = H.device
    
    # Get module weights and move to correct device
    # Используем weight.data чтобы избежать проблем с мета-тензорами
    if hasattr(module, 'weight'):
        if hasattr(module.weight, 'data'):
            weight = module.weight.data.clone().to(device)
        else:
            weight = module.weight.clone().to(device)
    else:
        # Если у модуля нет weight (например, LayerNorm), создаём пустой тензор
        weight = torch.zeros((0, 0), device=device, dtype=GPTQ_PRECISION)
    
    weight_next = None
    if module_next is not None and hasattr(module_next, 'weight'):
        # Проверяем, не является ли это мета-тензором
        if module_next.weight.is_meta:
            # Пропускаем вычисления с weight_next
            weight_next = None
        else:
            if hasattr(module_next.weight, 'data'):
                weight_next = module_next.weight.data.clone().to(device)
            else:
                weight_next = module_next.weight.clone().to(device)

    inp = inp.to(device=device)
    
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    num_added = inp.shape[0]

    # Process input for current module
    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp_processed = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp_processed = unfold(inp)
            inp_processed = inp_processed.permute([1, 0, 2])
            inp_processed = inp_processed.flatten(1)

    # Process input for next module if it exists
    inp_next_processed = None
    if inp_next is not None and module_next is not None and next_loss_lam != 0:
        inp_next = inp_next.to(device=device)
        if len(inp_next.shape) == 2:
            inp_next = inp_next.unsqueeze(0)

        match module_next:
            case torch.nn.Linear() | transformers.Conv1D():
                if len(inp_next.shape) == 3:
                    inp_next = inp_next.reshape((-1, inp_next.shape[-1]))
                inp_next_processed = inp_next.t()
            case torch.nn.Conv2d():
                unfold = torch.nn.Unfold(
                    module_next.kernel_size,
                    dilation=module_next.dilation,
                    padding=module_next.padding,
                    stride=module_next.stride,
                )
                inp_next_processed = unfold(inp_next)
                inp_next_processed = inp_next_processed.permute([1, 0, 2])
                inp_next_processed = inp_next_processed.flatten(1)

    # Update Hessian
    H *= num_samples / (num_samples + num_added)
    num_samples += num_added

    # Преобразуем инпуты
    inp_processed = inp_processed.to(dtype=GPTQ_PRECISION)
    inp_processed = math.sqrt(2 / num_samples) * inp_processed

    # Для X' используем inp_processed
    X_prime = inp_processed

    # Вычисляем дополнительные члены для нового гессиана
    if (inp_next_processed is not None and 
        weight_next is not None and 
        next_loss_lam != 0 and
        weight.numel() > 0 and weight_next.numel() > 0):
        
        # Get other layer activation
        WX = weight @ inp_processed  # [out_features, batch]
        
        # Apply sigmoid
        sigmoid_WX = torch.sigmoid(WX)  # [out_features, batch]
        
        # Count to simplify
        batch_size = X_prime.shape[1]
        out_features = weight.shape[0]
        
        # Count g_i as mean of all neirons
        g_values = torch.zeros(batch_size, device=device, dtype=GPTQ_PRECISION)

        is_runtime_error = False

        try:
            for i in range(batch_size):
                # For all examples count mean impact
                u_i = weight @ X_prime[:, i]  # [out_features]
                a_i = torch.sigmoid(u_i)  # [out_features]
                
                # sigma_prime и sigma_double_prime для всех нейронов
                sigma_prime = a_i * (1 - a_i)  # [out_features]
                sigma_double_prime = sigma_prime * (1 - 2 * a_i)  # [out_features]
                
                # v_h norm for all neirons
                v_norms = torch.norm(weight_next, dim=0)  # [out_features]
                
                # Mean g_i of all neirons
                g_i_sum = 2 * (v_norms**2) * (sigma_prime**2 + a_i * sigma_double_prime)
                g_values[i] = torch.mean(g_i_sum)
        except RuntimeError:
            is_runtime_error = True

        # Update hessian
        if not is_runtime_error:
            weights_matrix = torch.diag(2 + next_loss_lam * g_values)
            H_update = X_prime @ weights_matrix @ X_prime.t()
            
            H += H_update / num_samples
        else:
            H += X_prime @ X_prime.t()
    else:
        H += X_prime @ X_prime.t()

    return H, num_samples


def quantize_weight(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessians_dict: dict[torch.nn.Module, torch.Tensor],
    blocksize: int = 128,
    percdamp: float = 0.01,
    next_modules: torch.nn.Module | None = None,
    lam_tensor: torch.Tensor | None = 0.,
    kernel_mode: str = 'default'
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Quantize a module weight according to the GPTQ algorithm

    :param module: module with weight being quantized
    :param quant_args: quantization arguments used to find quantization parameters
    :param hessian_dict: dictionary containing preaccumulated hessian for quantization
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :return: loss, quantized_weight, scale, zero_point, g_idx
    """
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    # Caclulate Hessian
    H = hessians_dict[module]  # unfortunately python does not have a `move` keyword
    if next_modules is not None and lam_tensor is not None:
        try:
            for i, module_next in enumerate(next_modules):
                if module_next is not None:
                    H_next = hessians_dict[module_next]
                    H += lam_tensor[i] * apply_conv(H_next, mode=kernel_mode)
        except RuntimeError:
            pass
    
    # debug_mode
    # A = torch.randn(H.shape[0], H.shape[1], device=H.device, dtype=GPTQ_PRECISION)
    # H = ((A + A.T) / 2) * 0.01
    # H = torch.eye(H.shape[0], device=H.device, dtype=GPTQ_PRECISION) * 0.1

    # del hessians_dict[module]  # so we have to delete the original reference manually

    # create observer for calculating quantization parameters
    observer = Observer.load_from_registry(
        "memoryless_minmax",
        base_name="weight",
        args=quant_args,
        module=module,
    )

    # standardize shape and dtype
    match module:
        case torch.nn.Conv2d():
            W = W.flatten(1)
        case transformers.Conv1D():
            W.transpose_(0, 1)
    W = W.to(dtype=GPTQ_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    if strategy == QuantizationStrategy.GROUP:
        # mapping from column index to group index
        g_idx = (
            torch.arange(num_columns, device=W.device, dtype=torch.int)
            // quant_args.group_size
        )

        if actorder == ActivationOrdering.GROUP:
            # permute by activation order first, then update groups
            W, H, perm = _apply_activation_ordering(W, H)
            update_offload_parameter(module, "weight_g_idx", g_idx)
            scale, zero_point = observer(W)

            # use identity g_idx (invert permutation later)

        elif actorder == ActivationOrdering.WEIGHT:
            # update groups first, then permute by activation order
            scale, zero_point = observer(W)
            W, H, perm = _apply_activation_ordering(W, H)

            # permute g_idx to maintain identity mapping after unpermutation
            g_idx = g_idx[perm]

        else:
            scale, zero_point = observer(W)
    else:
        scale, zero_point = observer(W)

    # sparsity mask
    sparsity = tensor_sparsity(W)
    preserve_zeros = sparsity >= SPARSITY_THRESHOLD
    W_nz_mask = (
        (~torch.isclose(W, torch.zeros(1, device=W.device).float())).float()
        if preserve_zeros
        else None
    )

    losses = torch.zeros(num_rows, device=module.weight.device)

    # mask dead hessian values
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # compute inverse hessian in place to save memory
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
            "Failed to invert hessian due to numerical instability. Consider "
            "increasing GPTQModifier.dampening_frac, increasing the number "
            "of calibration samples, or shuffling the calibration dataset. "
            "Falling back to round-to-nearest for this module."
        )
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    W_adj = torch.zeros_like(W)

    # See section 3.4 of https://arxiv.org/abs/2203.07259
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

            # quantize column
            if strategy == QuantizationStrategy.TENSOR:
                q = fake_quantize(
                    q,
                    scale,
                    zero_point,
                    quant_args,
                )
            elif strategy == QuantizationStrategy.CHANNEL:
                q = fake_quantize(
                    q,
                    scale[:, 0],
                    zero_point[:, 0],
                    quant_args,
                )
            elif strategy == QuantizationStrategy.GROUP:
                # get the group index for the current column
                column_idx = i1 + i
                group_index = g_idx[column_idx]

                # Since we're only applying quantization to a slice, this
                # ends up being a channelwise application
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
                    f"Quantization strategy is not supported for GPTQ: {strategy}"
                )

            # propagate column error
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

        # propagate block error
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
            # restore original permutation
            invperm = torch.argsort(perm)
            W = W[:, invperm]

        elif actorder == ActivationOrdering.GROUP:
            # restore original permutation
            invperm = torch.argsort(perm)
            W = W[:, invperm]
            g_idx = g_idx[invperm]

            # only save g_idx if mapping is not identity
            has_gidx = True

    if not has_gidx:
        g_idx = None

    if isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.reshape(final_shape).to(final_dtype)

    # loss = torch.sum(losses).item()
    loss = torch.sum(losses)
    return (
        loss,
        W,
        W_adj,
        scale.to(dtype=final_dtype),
        zero_point.to(dtype=quant_args.pytorch_dtype()),
        g_idx,
        H
    )


def _apply_activation_ordering(
    W: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute weight and hessian in order of greatest outupt activations

    :param W: weight to permute
    :param H: hessian used to determine activation ordering
    :return: permuted weight, permuted hessian, permutation map
    """
    perm = torch.argsort(torch.diag(H), descending=True)
    return W[:, perm], H[perm][:, perm], perm
