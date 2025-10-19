__all__ = [
    'rgb_to_hed',
    'hed_to_rgb',
]

import torch

def rgb_to_hed(rgb: torch.Tensor) -> torch.Tensor: ...
def hed_to_rgb(hed: torch.Tensor) -> torch.Tensor: ...
