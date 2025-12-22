import torch
import torch.nn as nn
from llmcompressor.modifiers.utils.kernels import apply_conv

__all__ = ["LambdaLoss", "HessianLoss"]

class LambdaLoss(nn.Module):
    def forward(self, lam, curr, prev):
        # Returns a tensor connected to 'lam' for autograd
        return prev + 2 * lam

class HessianLoss(nn.Module):
    def forward(self, lam, H, H_next, kernel_mode):
        try:
            h = H + lam * apply_conv(H_next, mode=kernel_mode)
            h = (h + h.T) / 2
        except RuntimeError:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)
        
        eigenvalues = torch.linalg.eigvalsh(h)
        sorted_eigenvalues = torch.sort(eigenvalues, descending=True)[0]
        sorted_eigenval_size = sorted_eigenvalues.shape[0]

        if sorted_eigenval_size >= 2:
            return sorted_eigenvalues[0] * sorted_eigenvalues[1]
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)

class HessianLossUpgrade(nn.Module):
    def forward(self, lam, H, H_next, kernel_mode):
        try:
            h = H + lam * apply_conv(H_next, mode=kernel_mode)
            h = (h + h.T) / 2
        except RuntimeError:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)
        
        eigenvalues = torch.linalg.eigvalsh(h)
        sorted_eigenvalues = torch.sort(eigenvalues, descending=True)[0]
        sorted_eigenval_size = sorted_eigenvalues.shape[0]

        if sorted_eigenval_size >= 2:
            norm_eigens = torch.sum(torch.abs(eigenvalues))
            return sorted_eigenvalues[0] * sorted_eigenvalues[1] / (norm_eigens * norm_eigens)
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)

class HessianLossNormCos(nn.Module):
    def forward(self, lam, H, H_next, kernel_mode):
        try:
            h = H + lam * apply_conv(H_next, mode=kernel_mode)
            h = (h + h.T) / 2
        except RuntimeError:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(h)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        if sorted_eigenvalues.shape[0] >= 2:
            norm_eigens = torch.sum(torch.abs(sorted_eigenvalues))
            
            eigenvec1 = sorted_eigenvectors[:, 0]
            eigenvec2 = sorted_eigenvectors[:, 1]
            
            cos_sim = torch.abs(torch.dot(eigenvec1, eigenvec2))
            
            loss = sorted_eigenvalues[0] * sorted_eigenvalues[1] * (1 - cos_sim) / (norm_eigens * norm_eigens)
            return loss
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)