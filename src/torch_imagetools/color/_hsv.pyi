__all__ = [
    'rgb_to_hsv',
    'hsv_to_rgb',
]

import torch

def hsv_helper(
    rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

#
def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor: ...
def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor: ...
