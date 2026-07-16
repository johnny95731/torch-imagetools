__all__ = [
    'rgb_to_gray',
    'gray_to_rgb',
]

from typing import Literal, overload

import torch

from imgtools.core.math import matrix_transform
from imgtools.utils.helpers import __default_dtype


@overload
def rgb_to_gray(
    rgb: torch.Tensor,
    formula: str
    | Literal[
        'bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg', 'mean', 'max', 'midpoint'
    ] = 'bt.601',
) -> torch.Tensor: ...
def rgb_to_gray(
    rgb: torch.Tensor,
    formula: str = 'bt.601',
) -> torch.Tensor:
    """Converts an image from RGB space to grayscale.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    standard : {'bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg', 'mean', 'max', 'midpoint'}, default='bt.601'
        The conversion formula.

    Returns
    -------
    torch.Tensor
        An grayscale image with shape `(*, 1, H, W)`. The maximum is the same as
        the input.
    """
    dtype = __default_dtype(rgb)
    device = rgb.device
    # All chrominance channel are normalized to the range [-0.5, 0.5].
    if formula in 'bt.601':
        matrix = torch.tensor(
            ((0.299, 0.587, 0.114),), dtype=dtype, device=device
        )
    elif formula == 'bt.709':
        matrix = torch.tensor(
            ((0.2126, 0.7152, 0.0722),),
            dtype=dtype,
            device=device,
        )
    elif formula == 'bt.2020':
        matrix = torch.tensor(
            ((0.2627, 0.6780, 0.0593),),
            dtype=dtype,
            device=device,
        )
    elif formula == 'yiq':
        matrix = torch.tensor(((0.30, 0.59, 0.11),), dtype=dtype, device=device)
    elif formula == 'ycocg':
        matrix = torch.tensor(((0.25, 0.5, 0.25),), dtype=dtype, device=device)
    elif formula == 'mean':
        gray = rgb.mean(-3, keepdim=True, dtype=dtype)
        return gray
    elif formula == 'max':
        gray = rgb.amax(-3, keepdim=True)
        return gray
    elif formula == 'midpoint':
        maxi = rgb.amax(-3, keepdim=True)
        mini = rgb.amin(-3, keepdim=True)
        gray = (maxi + mini) / 2
        return gray
    else:
        raise ValueError(f'Invalid value of argument `formula`: {formula}')
    gray = matrix_transform(rgb, matrix)
    return gray


def gray_to_rgb(gray: torch.Tensor) -> torch.Tensor:
    """Converts an image from grayscale to rgb space.

    Parameters
    ----------
    gray : torch.Tensor
        An grayscale image with shape `(*, 1, H, W)`.

    Returns
    -------
    torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    """
    rgb = torch.cat((gray, gray, gray), dim=-3)
    return rgb
