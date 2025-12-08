import torch
import torch.nn.functional as F

KERNEL_DICT = {
    "gaussian": torch.tensor([
                    [1, 2, 1],
                    [2, 4, 2], 
                    [1, 2, 1]
                ], dtype=torch.float32) / 16.0,
    "default": torch.tensor([
                    [0, 0, 0],
                    [0, 1, 0], 
                    [0, 0, 0]
                ], dtype=torch.float32)
}


def apply_conv(x, mode="default"):
    if mode == 'default':
        return x
    """Apply Gaussian blur to 2D tensor"""
    # Add batch and channel dimensions: [H,W] -> [1,1,H,W]
    x = x.unsqueeze(0).unsqueeze(0)
    kernel = KERNEL_DICT[mode].view(1, 1, 3, 3)
    
    # Apply convolution with padding to maintain size
    x_conv = F.conv2d(x, kernel.to(x.device), padding=1)
    
    # Remove batch and channel dimensions: [1,1,H,W] -> [H,W]
    return x_conv.squeeze(0).squeeze(0)

__all__ = ['apply_conv']
