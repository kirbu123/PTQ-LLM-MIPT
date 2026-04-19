import torch
from loguru import logger
from torch.nn import Module
import torch
import torch.nn as nn
import os
import pandas as pd


def save_matrix_results(
    module: nn.Module,
    name: str = None,
    save_dir: str = None,
    M: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    if M is None:
        raise ValueError("M (calibration Hessian) must be passed from base.py")

    device = M.device
    dtype = torch.float32

    M = M.to(device=device, dtype=dtype).clone()

    # eigenvalues = torch.linalg.eigvalsh(M)
    # eigenvalues_sorted, _ = torch.sort(torch.abs(eigenvalues), descending=True)
    # eigenvalues_np = eigenvalues_sorted.cpu().numpy()'

    save_path = None

    if name is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, 'data')
        os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"{name}_eigenvalues.csv")
        # df = pd.DataFrame({
        #     "index": range(len(eigenvalues_np)),
        #     "eigenvalue": eigenvalues_np,
        # })
        # df.to_csv(save_path, index=False)

        save_path = os.path.join(save_dir, f"{name}_pure.csv")
        M_np = M.cpu().numpy()
        df_pure = pd.DataFrame(M_np)
        df_pure.to_csv(save_path, index=False, header=False)

    return M, save_path
