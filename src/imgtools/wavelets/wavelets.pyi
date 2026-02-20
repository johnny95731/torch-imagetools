__all__ = [
    'dwt2',
    'dwt2_partial',
    'idwt2',
]

from typing import Literal

import torch

def dwt2(
    img: torch.Tensor,
    scaling: torch.Tensor,
    wavelet: torch.Tensor,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
def dwt2_partial(
    img: torch.Tensor,
    scaling: torch.Tensor | None,
    wavelet: torch.Tensor | None,
    target: Literal['LL', 'LH', 'HL', 'HH'],
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
def idwt2(
    img: torch.Tensor,
    scaling: torch.Tensor,
    wavelet: torch.Tensor,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...
