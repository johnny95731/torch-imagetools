__all__ = [
    'Tensorlike',
    'align_device_type',
    'to_channel_coeff',
    'arrayize',
    'tensorize',
]

from typing import TypeVar

import numpy as np
import torch
from torch.types import Number

T = TypeVar('T')

Tensorlike = torch.Tensor | np.ndarray | list[Number] | Number

#
def check_valid_image_ndim(img: torch.Tensor, min_dim: int = 3) -> bool: ...

#
def align_device_type(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor: ...

#
def to_channel_coeff(coeff: torch.Tensor, num_ch: int) -> torch.Tensor: ...

#
def arrayize(img: Tensorlike) -> np.ndarray: ...
def tensorize(img: Tensorlike) -> torch.Tensor: ...
