__all__ = [
    'matrix_transform',
    '_check_ksize',
    'calc_padding',
    'filter2d',
    'atan2',
    'p_norm',
    'pca',
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
    padding: list[int] | Literal['same'] | None = 'same',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
def atan2(
    y: torch.Tensor,
    x: torch.Tensor,
    angle_unit: Literal['rad', 'deg'] = 'deg',
) -> torch.Tensor: ...

#
def p_norm(img: torch.Tensor, p: float | str) -> torch.Tensor: ...

#
def pca(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
