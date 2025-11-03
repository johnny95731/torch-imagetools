__all__ = [
    'Tensorlike',
    'pairing',
    'is_indexable',
    'check_valid_image_ndim',
    'align_device_type',
    'arrayize',
    'tensorize',
]

from typing import Any, overload, TypeVar

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
def check_valid_image_ndim(img: torch.Tensor): ...

#
def align_device_type(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor: ...

#
def arrayize(img: Tensorlike) -> np.ndarray: ...
def tensorize(img: Tensorlike) -> torch.Tensor: ...
