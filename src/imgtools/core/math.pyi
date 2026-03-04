__all__ = [
    'matrix_transform',
    'calc_padding',
    'filter2d',
    'deg_to_rad',
    'rad_to_deg',
    'histogram',
]

from typing import Literal

import torch

def matrix_transform(
    img: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor: ...

#
def _check_ksize(
    ksize: int | tuple[int, int],
    positive: bool = True,
) -> tuple[int, int]: ...
def calc_padding(ksize: tuple[int, int]) -> tuple[int, int, int, int]: ...
def filter2d(
    img: torch.Tensor,
    kernel: torch.Tensor,
    padding: list[int] | str | None = 'same',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
def deg_to_rad(deg: torch.Tensor): ...
def rad_to_deg(deg: torch.Tensor): ...
