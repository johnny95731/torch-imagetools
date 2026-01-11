__all__ = [
    'Tensorlike',
    'pairing',
    'is_indexable',
    'align_device_type',
    'to_channel_coeff',
    'arrayize',
    'tensorize',
]

from typing import Any, TypeVar

import numpy.typing as npt
import numpy as np
import torch
from torch.types import Number

T = TypeVar('T')

Tensorlike = torch.Tensor | npt.NDArray | list[Number] | Number


def pairing(item: Any):
    """Converts item to a tuple with two items.
    If the item is indexble, returns first two items;
    otherwise, returns (item, item).

    This function is not jit-able.
    """
    if is_indexable(item):
        return (item[0], item[1])
    return (item, item)


def is_indexable(item: Any) -> bool:
    """Check whether an item contains `__getitem__` method.

    This function is not jit-able.
    """
    return hasattr(item, '__getitem__')


def check_valid_image_ndim(img: torch.Tensor, min_dim: int = 3) -> bool:
    """Check whether 2 <= img.ndim <= 4.

    Parameters
    ----------
    img : torch.Tensor
        A tensor.
    min_dim : int, default=3
        Minimum of the ndim.

    Returns
    -------
    bool
        Whether img.ndim <= 3.

    Raises
    ------
    ValueError
        When img.ndim < min_dim or img.ndim > 4.
    """
    ndim = img.ndim
    if not (min_dim <= ndim <= 4):
        raise ValueError(
            f'Dimention of the image should be in [{min_dim}, 4], but found {ndim}.'
        )
    is_not_batch = ndim <= 3
    return is_not_batch


def align_device_type(source: torch.Tensor, target: torch.Tensor):
    """Aligns device and dtype of the source tensor to the target tensor.

    Parameters
    ----------
    source : torch.Tensor
        A tensor need to align.
    target : torch.Tensor
        A tensor provides device and dtype.

    Returns
    -------
    torch.Tensor
        Source tensor with
        - same device as target tensor
        - same dtype as target tensor if target.dtype is a floating point
        - float32 dtype if target.dtype is not a floating point
    """
    dtype = target.dtype if torch.is_floating_point(target) else torch.float32
    source = source.to(target.device, dtype)
    return source


def to_channel_coeff(
    coeff: int | float | torch.Tensor,
    num_ch: int,
) -> torch.Tensor:
    """Convert shape of coefficeints such that `coeff (op) img` is valid,
    where op may be one of arithmetic operations (+-*/) or other operator.

    Parameters
    ----------
    coeff : int | float | torch.Tensor
        The values to be reshape. Must be one of the following:
            1. int or float
            2. Tensor with only one element.
            3. Tensor with shape (num_ch,) or (batch, num_ch)
    num_ch : int
        Number of image channels.

    Returns
    -------
    torch.Tensor
        Coefficeints with shape
        - (1,) if `coeff` is a number or `coeff.numel() == 1`
        - (*, C, 1, 1) if `coeff.shape` is (*, C).

    Raises
    ------
    ValueError
        If coeff.numel() == 0 or coeff.ndim > 2.
    ValueError
        When coeff.size(-1) is neither 1 nor `num_ch`.
    """
    if isinstance(coeff, (int, float)):
        res = torch.tensor(float(coeff))
        return res
    if coeff.numel() == 1:
        res = coeff.reshape(1)
        return coeff

    if coeff.numel() == 0 or coeff.ndim > 2:
        raise ValueError(
            f'Requires coeff.numel() = {coeff.numel()} >= 1'
            f'and coeff.ndim = {coeff.ndim} <= 2.'
        )
    if coeff.size(-1) != 1 and coeff.size(-1) != num_ch:
        raise ValueError('coeff.size(-1) must equals to 1 or num_ch.')

    res = coeff.unsqueeze(-1).unsqueeze_(-1)
    return res


def arrayize(img: Tensorlike) -> npt.NDArray:
    """Converts an item to a np.ndarray.

    If input is a torch.Tensor:
        1. Moves -3-axis to the last for ndim >= 3.
        2. Without any handling for the other cases.
    For other types, convert to a ndarray by np.array.

    This function is not jit-able.

    Parameters
    ----------
    img : Tensorlike
        An item to be converted to tensor.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach()
        img = (
            img.movedim(-3, -1)  # (*, C, H, W) -> (*, H, W, C)
            if img.ndim >= 3
            else img
        )
        img = img.contiguous().numpy()
    else:
        img = np.asarray(img)
    return img


def tensorize(img: Tensorlike) -> torch.Tensor:
    """
    Converts an item to a contiguous torch.Tensor.

    If input is a np.ndarray:
        1. Moves -1-axis to -3-axis if ndim >= 3.
        2. Reshape to (1, H, W) if ndim = 2.
        3. Otherwise, without any handling.
    For other types, convert to a tensor by torch.tensor.

    This function is not jit-able.

    Parameters
    ----------
    img : Tensorlike
        An item to be converted to tensor.

    Examples
    --------

    >>> import numpy as np
    >>> from imgtools.utils import tensorize
    >>> from PIL import Image
    >>>
    >>> img = np.asarray(Image.open(file))
    >>> img.shape  # (H, W, C)
    >>> img_t = tensorize(img)
    >>> img_t.shape  # (C, H, W)
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        img = (
            img.movedim(-1, -3)  # (..., H, W, C) -> (..., C, H, W)
            if img.ndim >= 3
            else img.unsqueeze(0)  # (H, W) -> (1, H, W)
            if img.ndim == 2
            else img
        )
    elif not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    img = img.contiguous()
    return img
