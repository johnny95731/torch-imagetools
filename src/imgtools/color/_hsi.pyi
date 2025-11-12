__all__ = [
    'rgb_to_hsi',
    'hsi_to_rgb',
]

import torch

def rgb_to_hsi(rgb: torch.Tensor) -> torch.Tensor: ...
def hsi_to_rgb(hsi: torch.Tensor) -> torch.Tensor: ...
