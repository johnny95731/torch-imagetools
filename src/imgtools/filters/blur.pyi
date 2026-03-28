__all__ = [
    'box_blur',
    'get_gaussian_kernel',
    'gaussian_blur',
    'guided_filter',
    'max_filter',
    'min_filter',
]

from typing import Literal

import torch

def box_blur(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    normalize: bool = True,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
def get_gaussian_kernel(
    ksize: int | tuple[int, int] = 5,
    sigma: float | tuple[float, float] = 0.0,
    normalize: bool = True,
) -> torch.Tensor: ...
def gaussian_blur(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    sigma: float | tuple[float, float] = 0.0,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
def guided_filter(
    img: torch.Tensor,
    guidance: torch.Tensor | None = None,
    ksize: int | tuple[int, int] = 5,
    eps: float = 0.01,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
def max_filter(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...
def min_filter(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...
