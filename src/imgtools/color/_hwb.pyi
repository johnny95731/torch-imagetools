"""Color conversion functions between RGB space and HWB space"""

__all__ = [
    'rgb_to_hwb',
    'hwb_to_rgb',
]

import torch

def rgb_to_hwb(rgb: torch.Tensor) -> torch.Tensor: ...
def hwb_to_rgb(hwb: torch.Tensor) -> torch.Tensor: ...
