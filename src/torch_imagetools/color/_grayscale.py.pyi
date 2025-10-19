__all__ = [
    'rgb_to_gray',
    'gray_to_rgb',
]

import torch

def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor: ...
def gray_to_rgb(gray: torch.Tensor) -> torch.Tensor: ...
