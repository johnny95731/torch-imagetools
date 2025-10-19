__all__ = [
    'matrix_transform',
    'filter2d',
    'atan2',
    'p_norm',
]

from typing import Literal

import torch

def matrix_transform(
    img: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor: ...

#
def filter2d(
    img: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor: ...

#
def atan2(
    y: torch.Tensor,
    x: torch.Tensor,
    angle_unit: Literal['rad', 'deg'] = 'deg',
) -> torch.Tensor: ...

#
def p_norm(
    img: torch.Tensor,
    p: float | Literal['inf', '-inf'],
) -> torch.Tensor: ...
