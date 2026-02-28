__all__ = ['bilateral_hdr']

from typing import Literal

import torch

#
def _edge_stopping_huber(diff: torch.Tensor, coeff: float) -> torch.Tensor: ...
def _edge_stopping_lorentz(
    diff: torch.Tensor, coeff: float
) -> torch.Tensor: ...
def _edge_stopping_turkey(diff: torch.Tensor, coeff: float) -> torch.Tensor: ...
def _edge_stopping_gaussian(
    diff: torch.Tensor, coeff: float
) -> torch.Tensor: ...

#
def bilateral_hdr(
    img: torch.Tensor,
    sigma_c: float = 0.15,
    sigma_s: float | None = 1.0,
    contrast: float = 1.5,
    downsample: float = 1,
    edge_stopping: Literal[
        'huber', 'lorentz', 'turkey', 'gaussian'
    ] = 'gaussian',
    tone: Literal['soft', 'lighter', 'linear', 'std'] = 'soft',
) -> torch.Tensor: ...
