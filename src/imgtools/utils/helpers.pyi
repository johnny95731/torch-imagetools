__all__ = [
    'Tensorlike',
    'align_device_type',
    'arrayize',
    'tensorize',
]

import numpy as np
import torch
from torch.types import Number

Tensorlike = torch.Tensor | np.ndarray | list[Number] | Number

#
def check_valid_image_ndim(img: torch.Tensor, min_dim: int = 3) -> bool: ...

#
def __default_dtype(x: torch.Tensor) -> torch.dtype: ...
def align_device_type(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor: ...

#
def _to_channel_coeff(
    coeff: int | float | torch.Tensor,
    num_ch: int,
) -> torch.Tensor: ...

#
def arrayize(img: Tensorlike) -> np.ndarray: ...
def tensorize(
    img: Tensorlike,
    dtype: torch.dtype | None = None,
    device: torch.device = torch.device('cpu'),
    copy: bool = False,
) -> torch.Tensor: ...
