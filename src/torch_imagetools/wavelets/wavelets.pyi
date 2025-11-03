__all__ = [
    'dwt',
    'dwt_partial',
]

from typing import Literal

import torch

def dwt(
    img: torch.Tensor,
    scaling: torch.Tensor,
    wavelet: torch.Tensor,
) -> list[torch.Tensor]: ...

#
def dwt_partial(
    img: torch.Tensor,
    scaling: torch.Tensor | None,
    wavelet: torch.Tensor | None,
    target: Literal['LL', 'LH', 'HL', 'HH'],
) -> torch.Tensor: ...
