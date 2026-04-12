"""
GPT-2 (HuggingFace) uses Conv1D with weight layout (in_features, out_features).
DeepSpeed-style conversion replaces Conv1D with nn.Linear and stores weights as
(out_features, in_features). That layout matches compression / GPTQ, but a
plain `AutoModelForCausalLM.from_pretrained` on the saved folder still builds
Conv1D layers — so checkpoints must be converted back before evaluation.

This module replaces `deepspeed.compression.helper.convert_conv1d_to_linear` and
adds the inverse `convert_linear_to_conv1d` for GPT-2 block linears only.
"""

from __future__ import annotations

import types
from typing import Callable, Type

import torch.nn as nn
from transformers.pytorch_utils import Conv1D

__all__ = [
    "convert_conv1d_to_linear",
    "convert_linear_to_conv1d",
    "is_gpt2_conv1d_module_path",
]

_GPT2_CONV1D_SUFFIXES: tuple[str, ...] = (
    ".attn.c_attn",
    ".attn.c_proj",
    ".mlp.c_fc",
    ".mlp.c_proj",
)


def is_gpt2_conv1d_module_path(module_name: str) -> bool:
    if not module_name:
        return False
    return any(module_name.endswith(s) for s in _GPT2_CONV1D_SUFFIXES)


def _recursive_getattr(model: nn.Module, module_name: str) -> nn.Module:
    obj: nn.Module | types.ModuleType = model
    for part in module_name.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _recursive_setattr(model: nn.Module, module_name: str, module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _copy_public_attributes(src: nn.Module, dst: nn.Module) -> None:
    for key, value in src.__dict__.items():
        if key in ("weight", "bias"):
            continue
        if key.startswith("_"):
            continue
        try:
            setattr(dst, key, value)
        except (AttributeError, TypeError):
            pass


def convert_conv1d_to_linear(
    model: nn.Module,
    convert_type: Type[nn.Module] | None = None,
) -> nn.Module:
    if convert_type is None:
        convert_type = Conv1D

    c_model = model.module if hasattr(model, "module") else model

    to_replace: list[tuple[str, nn.Module]] = []
    for name, module in c_model.named_modules():
        if isinstance(module, convert_type):
            to_replace.append((name, module))

    for name, _ in to_replace:
        old = _recursive_getattr(c_model, name)
        new_module = nn.Linear(
            old.weight.data.size(0),
            old.weight.data.size(1),
            bias=old.bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        new_module.weight.data = old.weight.data.t().contiguous()
        if new_module.bias is not None and old.bias is not None:
            new_module.bias.data = old.bias.data.view(-1).contiguous()

        _copy_public_attributes(old, new_module)
        _recursive_setattr(c_model, name, new_module)

    return model


def convert_linear_to_conv1d(
    model: nn.Module,
    path_filter: Callable[[str], bool] | None = None,
    linear_type: Type[nn.Module] = nn.Linear,
) -> nn.Module:
    if path_filter is None:
        path_filter = is_gpt2_conv1d_module_path

    c_model = model.module if hasattr(model, "module") else model

    to_replace: list[tuple[str, nn.Module]] = []
    for name, module in c_model.named_modules():
        if not name or not isinstance(module, linear_type):
            continue
        if not path_filter(name):
            continue
        to_replace.append((name, module))

    for name, _ in to_replace:
        old = _recursive_getattr(c_model, name)
        nf, nx = old.out_features, old.in_features
        new_module = Conv1D(nf, nx)
        new_module = new_module.to(device=old.weight.device, dtype=old.weight.dtype)
        new_module.weight.data = old.weight.data.t().contiguous()
        if old.bias is not None:
            new_module.bias.data = old.bias.data.view(-1).contiguous()
        else:
            new_module.bias.data.zero_()

        _copy_public_attributes(old, new_module)
        _recursive_setattr(c_model, name, new_module)

    return model