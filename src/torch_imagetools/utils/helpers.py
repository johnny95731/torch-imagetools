from typing import Any, TypeVar

import numpy as np
import torch
from torch.types import Number


T = TypeVar('T')

Tensorlike = torch.Tensor | np.ndarray | list[Number] | Number


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


def align_device_type(source: torch.Tensor, target: torch.Tensor):
    """Aligns device and dtype of the source tensor to the target tensor.

    Parameters
    ----------
    source : torch.Tensor
        A tensor need to align.
    target : torch.Tensor
        A tensor provides device and dtype.
    """
    dtype = target.dtype if torch.is_floating_point(target) else torch.float32
    source = source.to(target.device, dtype)
    return source


def arrayize(img: Tensorlike) -> np.ndarray:
    """Converts an item to a np.ndarray.

    If input is a torch.Tensor:
        1. Moves -3-axis to the last for ndim >= 3.
        2. Without any handling for the other cases.
    For other types, convert to a ndarray by np.array.

    Parameters
    ----------
    img : Tensorlike
        An item to be converted to tensor.
    """
    if torch.is_tensor(img):
        img = (
            img.movedim(-3, -1)  # (*, C, H, W) -> (*, H, W, C)
            if img.ndim >= 3
            else img
        )
        img = img.numpy()
    else:
        img = torch.tensor(img)
    return img


def tensorize(img: Tensorlike) -> torch.Tensor:
    """
    Converts an item to a torch.Tensor.

    If input is a np.ndarray:
        1. Moves -1-axis to -3-axis for ndim >= 3.
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
