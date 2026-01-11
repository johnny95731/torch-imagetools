__all__ = [
    'Tensorlike',
    'pairing',
    'is_indexable',
    'align_device_type',
    'to_channel_coeff',
    'arrayize',
    'tensorize',
]

from typing import Any, TypeVar, overload

import numpy as np
import torch
from torch.types import Number

T = TypeVar('T')

Tensorlike = torch.Tensor | np.ndarray | list[Number] | Number

@overload
def pairing(item: list[T] | tuple[T, ...]) -> tuple[T, T]: ...
@overload
def pairing(item: T) -> tuple[T, T]: ...

#
def is_indexable(item: Any) -> bool: ...

#
def check_valid_image_ndim(img: torch.Tensor) -> bool: ...

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
