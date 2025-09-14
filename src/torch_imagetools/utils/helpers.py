__all__ = [
    'pariing',
    'is_indexable',
    'tensorize',
    'matrix_transform',
    'matrix_transform_',
]

from typing import Any, overload, TypeVar

import numpy as np
import torch
from torch.types import Number


T = TypeVar('T')

Tensorlike = torch.Tensor | np.ndarray | list[Number] | Number


# fmt: off
@overload
def pairing(item: list[T] | tuple[T, ...]) -> tuple[T, T]: ...
@overload
def pairing(item: T) -> tuple[T, T]: ...
# fmt: on
def pairing(item: Any):
    """Converts item to a tuple with two items.
    If the item is indexble, returns first two items;
    otherwise, returns (item, item)
    """
    if is_indexable(item):
        return (item[0], item[1])
    return (item, item)


def is_indexable(item: Any) -> bool:
    """Check whether an item contains `__getitem__` method."""
    return hasattr(item, '__getitem__')


def tensorize(img: Tensorlike) -> torch.Tensor:
    """Converts an item to a tensor.

    If input is a np.ndarray:
        1. Permute to (*, C, H, W) for ndim >= 3.
        2. Reshape to (1, H, W) for ndim = 2.
        3. Without any handling for the other cases.
    For other types, convert to a tensor by torch.tensor.

    Parameters
    ----------
    img : Tensorlike
        An item to be converted to tensor.
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        img = (
            img.movedim(-1, -3)  # (*, H, W, C) -> (*, C, H, W)
            if img.ndim >= 3
            else img.unsqueeze(0)  # (H, W) -> (1, H, W)
            if img.ndim == 2
            else img
        )
    elif not torch.is_tensor(img):
        img = torch.tensor(img)
    return img


def matrix_transform(
    img: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Converts the channels of an image by linear transformation.

    Parameters
    ----------
    img : torch.Tensor
        Image, a tensor with shape (*, C, H, W).
    matrix : torch.Tensor
        The transformation matrix with shape (C_out, C).

    Returns
    -------
    torch.Tensor
        The image with shape (*, C_out, H, W).
    """
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    matrix = matrix.to(img.device, dtype)
    output = torch.einsum('oc,...chw->...ohw', matrix, img)
    return output
