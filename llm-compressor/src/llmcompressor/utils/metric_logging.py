"""
Utility functions for metrics logging and GPU memory monitoring.

This module provides functions for tracking GPU memory usage, measuring model
layer sizes, and comprehensive logging during compression workflows.
Supports both NVIDIA and AMD GPU monitoring with detailed memory
statistics and performance metrics.
"""

import time
from typing import List, Tuple

import torch
from loguru import logger
from torch.nn import Module
import torch
import torch.nn as nn

import transformers
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from compressed_tensors.quantization import fake_quantize
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
)


hessian_logging_dir = '/home/buka2004/PTQ-LLM-MIPT/algo_outputs/basic'

__all__ = ["get_GPU_memory_usage", "get_layer_size_mb", "CompressionLogger"]


def get_GPU_memory_usage() -> List[Tuple[float, float]]:
    if torch.version.hip:
        return get_GPU_usage_amd()
    else:
        return get_GPU_usage_nv()


def get_GPU_usage_nv() -> List[Tuple[float, float]]:
    """
    get gpu usage for Nvidia GPUs using nvml lib
    """
    try:
        import pynvml
        from pynvml import NVMLError

        try:
            pynvml.nvmlInit()
        except NVMLError as _err:
            logger.warning(f"Pynml library error:\n {_err}")
            return []

        device_count = pynvml.nvmlDeviceGetCount()
        usage = []  # [(percentage, total_memory_MB)]

        # Iterate through all GPUs
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage_percentage = mem_info.used / mem_info.total
            total_memory_gb = mem_info.total / (1e9)
            usage.append(
                (memory_usage_percentage, total_memory_gb),
            )
        pynvml.nvmlShutdown()
        return usage

    except ImportError:
        logger.warning("Failed to obtain GPU usage from pynvml")
        return []


def get_GPU_usage_amd() -> List[Tuple[float, float]]:
    """
    get gpu usage for AMD GPUs using amdsmi lib
    """
    usage = []
    try:
        import amdsmi

        try:
            amdsmi.amdsmi_init()
            devices = amdsmi.amdsmi_get_processor_handles()

            for device in devices:
                vram_memory_usage = amdsmi.amdsmi_get_gpu_memory_usage(
                    device, amdsmi.amdsmi_interface.AmdSmiMemoryType.VRAM
                )
                vram_memory_total = amdsmi.amdsmi_get_gpu_memory_total(
                    device, amdsmi.amdsmi_interface.AmdSmiMemoryType.VRAM
                )

                memory_percentage = vram_memory_usage / vram_memory_total
                usage.append(
                    (memory_percentage, vram_memory_total / (1e9)),
                )
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiException as error:
            logger.warning(f"amdsmi library error:\n {error}")
    except ImportError:
        logger.warning("Failed to obtain GPU usage from amdsmi")

    return usage


def get_layer_size_mb(module: Module) -> float:
    param_size = 0
    buffer_size = 0

    for param in module.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in module.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size
    total_size_mb = total_size / (1e6)  # Convert bytes to MB

    return total_size_mb


class CompressionLogger:
    """
    Log metrics related to compression algorithm

    :param start_tick: time when algorithm started"
    :param losses: loss as result of algorithm
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.start_tick = None
        self.loss = None

    def set_loss(self, loss: float):
        self.loss = loss

    def __enter__(self) -> "CompressionLogger":
        self.start_tick = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        stop_tick = time.time()
        patch = logger.patch(lambda r: r.update(function="compress"))

        if self.start_tick is not None:
            duration = stop_tick - self.start_tick
            patch.log("METRIC", f"time {duration:.2f}s")
        if self.loss is not None:
            patch.log("METRIC", f"error {self.loss:.2f}")

        gpu_usage = get_GPU_memory_usage()
        if len(gpu_usage) > 0:
            for i in range(len(gpu_usage)):
                perc = gpu_usage[i][0] * 100
                total_memory = int(gpu_usage[i][1])  # GB
                patch.log(
                    "METRIC",
                    (
                        f"GPU {i} | usage: {perc:.2f}%"
                        f" | total memory: {total_memory} GB"
                    ),
                )

        compressed_size = get_layer_size_mb(self.module)
        patch.log("METRIC", f"Compressed module size: {compressed_size} MB")


def plot_eigenvalues(csv_path: str, trunc: int = 35):
    """
    Plot eigenvalues from CSV file and save as PNG.
    
    Args:
        csv_path: Path to the eigenvalues CSV file
    """
    # Load eigenvalues
    df = pd.read_csv(csv_path)
    eigenvalues = df['eigenvalue'].values
    eigenvalues = eigenvalues[trunc:]
    
    
    # Create save path
    save_path = csv_path.replace('_eigenvalues.csv', '_eigenvalues.png')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(eigenvalues)), eigenvalues, 'b-', linewidth=1)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalues of {os.path.basename(csv_path).replace("_eigenvalues.csv", "")}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    print(f"Saved plot to {save_path}")


def compute_hessian_metrics(
    module: nn.Module,
    name: str = None,
    save_dir: str = hessian_logging_dir,
    H_cal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    """
    Correct L2 Hessian H = 2/n * X X^T for FP layer.
    Pass H_cal = self._hessians[module].clone() (calibration Hessian accumulated
    from real activations). This is the exact Hessian of ||WX - WhatX||^2 w.r.t. What.
    """
    if H_cal is None:
        raise ValueError("H_cal (calibration Hessian) must be passed from base.py")

    device = H_cal.device
    dtype = torch.float32

    H = H_cal.to(device=device, dtype=dtype).clone()
    H = (H + H.T) / 2

    eigenvalues = torch.linalg.eigvalsh(H)
    eigenvalues_sorted, _ = torch.sort(torch.abs(eigenvalues), descending=True)
    eigenvalues_np = eigenvalues_sorted.cpu().numpy()
    save_path = None

    if name is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, 'data')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{name}_eigenvalues.csv")
        df = pd.DataFrame({
            "index": range(len(eigenvalues_np)),
            "eigenvalue": eigenvalues_np,
        })
        df.to_csv(save_path, index=False)

    return H, eigenvalues_sorted, save_path


def compute_quantized_hessian_metrics(
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    quant_args,
    module: nn.Module,
    H_cal: torch.Tensor,
    name: str = None,
    save_dir: str = hessian_logging_dir,
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    """
    Pushed-forward L2 Hessian H_q = W_q H_cal W_q^T.
    Since H_cal = 2/n X X^T (calibration), this equals 2/n (W_q X)(W_q X)^T —
    the exact Hessian of the next layer's L2 loss using real calibration activations.
    W_q = fake_quantize(W, scale, zero_point, quant_args).
    """
    device = H_cal.device
    dtype = torch.float32

    W_q = fake_quantize(quantized_weight, scale, zero_point, quant_args)
    W_q = W_q.to(device=device, dtype=dtype)

    H = H_cal.to(device=device, dtype=dtype).clone()
    H = (H + H.T) / 2

    # H_q = W_q H_cal W_q^T
    # Linear:  W_q (out, in),  H_cal (in, in)  → H_q (out, out)
    # Conv1D:  W_q (in, out),  H_cal (in, in)  → H_q = W_q^T H_cal W_q would be wrong;
    #          accumulate_hessian transposes Conv1D weight first, so treat as Linear
    if isinstance(module, nn.Linear):
        H_q = W_q @ H @ W_q.T          # (out, out)
    elif isinstance(module, transformers.Conv1D):
        # Conv1D: weight stored as (in, out); in gptq_quantize it's transposed to (out, in)
        H_q = W_q.T @ H @ W_q          # W_q.T is (out, in): H_q (out, out)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")

    H_q = (H_q + H_q.T) / 2

    eigenvalues = torch.linalg.eigvalsh(H_q)
    eigenvalues_sorted, _ = torch.sort(torch.abs(eigenvalues), descending=True)
    eigenvalues_np = eigenvalues_sorted.cpu().numpy()
    save_path = None

    if name is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, 'data')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{name}.csv")
        df = pd.DataFrame({
            "index": range(len(eigenvalues_np)),
            "eigenvalue": eigenvalues_np,
        })
        df.to_csv(save_path, index=False)

    return H_q, eigenvalues_sorted, save_path


# def plot_eigenvalue_list(csv_paths: list[str], trunc: int = 30):
#     """
#     Plot eigenvalues from multiple CSV files in one graph and save as PNG.
    
#     Args:
#         csv_paths: List of paths to the eigenvalues CSV files
#         trunc: Number of eigenvalues to truncate from the beginning
#     """
#     plt.figure(figsize=(12, 8))
    
#     colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
#     for i, csv_path in enumerate(csv_paths):
#         # Load eigenvalues
#         df = pd.read_csv(csv_path)
#         eigenvalues = df['eigenvalue'].values
#         eigenvalues = eigenvalues[trunc:]
        
#         # Get name from path
#         name = os.path.basename(csv_path).replace('_eigenvalues.csv', '')

#         # Plot
#         color = colors[i % len(colors)]
#         plt.plot(range(len(eigenvalues)), eigenvalues, 
#                 color=color, linewidth=1, label=name, alpha=0.7)
    
#     plt.xlabel('Index')
#     plt.ylabel('Eigenvalue')
#     plt.title(f'Eigenvalues Comparison (truncated first {trunc} values)')
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()

#     # Save in the same directory as first csv
#     name = f'{name[:name.find("_eigenvalues.csv")]}_trunc={trunc}'
#     save_path = os.path.join(os.path.dirname(csv_paths[0]), f'{name}_eigenvalues_comparison.png')
#     plt.savefig(save_path, dpi=100, bbox_inches='tight')
#     plt.close()
    
#     print(f"Saved comparison plot to {save_path}")

def plot_eigenvalue_list(csv_paths: list[str], trunc: int = 30):
    """
    Plot eigenvalues from multiple CSV files in one graph and save as PNG.
    Saves to <save_dir>/plots/ (one level up from the data/ dir of csv_paths[0]).

    Args:
        csv_paths: List of paths to the eigenvalues CSV files (under .../data/)
        trunc: Number of leading eigenvalues to truncate from the plot
    """
    plt.figure(figsize=(12, 8))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        eigenvalues = df['eigenvalue'].values[trunc:]

        label = os.path.basename(csv_path).replace('_eigenvalues.csv', '').replace('.csv', '')

        color = colors[i % len(colors)]
        plt.plot(range(len(eigenvalues)), eigenvalues,
                 color=color, linewidth=1, label=label, alpha=0.7)

    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalues Comparison (truncated first {trunc} values)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # CSVs live in .../data/, PNGs go in .../plots/
    base_label = os.path.basename(csv_paths[0]).replace('_eigenvalues.csv', '').replace('.csv', '')
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(csv_paths[0])), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f'{base_label}_trunc={trunc}_eigenvalues_comparison.png')

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    module = nn.Linear(768, 768)
    device = "cuda:0"
    module = module.to(device)
    
    compute_hessian_metrics(module, "test_dataset")