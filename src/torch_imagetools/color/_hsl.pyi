__all__ = [
    'rgb_to_hsl',
    'hsl_to_rgb',
]

import torch

def rgb_to_hsl(rgb: torch.Tensor) -> torch.Tensor: ...
def hsl_to_rgb(hsl: torch.Tensor) -> torch.Tensor: ...
