__all__ = [
    'matrix_transform',
    'calc_padding',
    'filter2d',
    'deg_to_rad',
    'rad_to_deg',
]

from math import ceil, floor

import torch
from torch.nn.functional import conv2d, pad

from ..utils.helpers import align_device_type


def matrix_transform(
    img: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Converts the channels of an image by linear transformation.

    Parameters
    ----------
    img : torch.Tensor
        Image, a tensor with shape `(*, C, H, W)`.
    matrix : torch.Tensor
        The transformation matrix with shape `(*, C_out, C)`.

    Returns
    -------
    torch.Tensor
        The image with shape `(*, C_out, H, W)`.
    """
    matrix = align_device_type(matrix, img)
    output = torch.einsum('...oc,...chw->...ohw', matrix, img)
    return output


def _check_ksize(
    ksize: int | tuple[int, int],
    positive: bool = True,
    odd: bool = False,
) -> tuple[int, int]:
    """Checks the validity of the kernel size and returns a tuple of kernel
    size.

    Parameters
    ----------
    ksize : int | tuple[int, int]
        Kernel size.
    positive : bool, default=True
        Requires the kernel size to be (strict) positive.
    odd : bool, default=False
        Requires the kernel size to be odd.

    Returns
    -------
    tuple[int, int]
        A tuple of kernel size `(ksize_y, ksize_x)`.
    """
    if isinstance(ksize, int):
        _ksize = (ksize, ksize)
    elif isinstance(ksize, (tuple, list)):
        if len(ksize) == 0:
            raise ValueError('len(ksize) can not be 0.')
        elif len(ksize) == 1:
            _ksize = (ksize[0], ksize[0])
        else:
            _ksize = (ksize[0], ksize[1])
        if not isinstance(_ksize[0], int) or not isinstance(_ksize[1], int):
            raise TypeError('ksize must be int type.')
    else:
        raise TypeError(f'Invalid type of ksize: {type(ksize)}')
    if positive and (_ksize[0] <= 0 or _ksize[1] <= 0):
        raise ValueError('ksize must be positive integers.')
    if odd and (_ksize[0] % 2 == 0 or _ksize[1] % 2 == 0):
        raise ValueError(f'`ksize` must be odd integer(s): {ksize}')
    return _ksize


def calc_padding(ksize: tuple[int, int]) -> tuple[int, int, int, int]:
    """Calculate padding by a given ksize.

    Parameters
    ----------
    ksize : tuple[int, int]
        Kernel shape `(y_direction, x_direction)`.

    Returns
    -------
    tuple[int, int, int, int]
        `(padding_left, padding_right, padding_top, padding_bottom)`.
    """
    if (
        ksize[0] <= 0
        or not isinstance(ksize[0], int)
        or ksize[1] <= 0
        or not isinstance(ksize[1], int)
    ):
        raise ValueError('ksize must be postive integers.')
    pad_y = (ksize[0] - 1) / 2
    pad_x = (ksize[1] - 1) / 2
    padding = (floor(pad_x), ceil(pad_x), floor(pad_y), ceil(pad_y))
    return padding


def filter2d(
    img: torch.Tensor,
    kernel: torch.Tensor,
    padding: int | list[int] | str | None = 'same',
    mode: str = 'reflect',
) -> torch.Tensor:
    """Image convolution with a 2D kernel.

    Parameters
    ----------
    img : torch.Tensor
        Image, a tensor with shape `(*, C, H, W)`.
    kernel : torch.Tensor
        A convolution kernel with shape `(k_x,)`, `(k_y, k_x)`,
        `(C * k, k_y, k_x)`, or `(B, C * k, k_y, k_x)`, where k is a positive
        integer.
    padding : int | list[int] | 'same' | None, default='same'
        The padding size. `"same"`: computes from `ksize`.
        `list[int]`: `(left, right, top, bottom)`. `None`: no padding.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        The image with shape `(*, C, H0, W0)`.
    """
    is_single_image = img.ndim == 3
    if is_single_image:
        img = img.unsqueeze(0)
    num_ch = img.size(-3)
    if not torch.is_floating_point(img):
        img = img.float()
    # kernel shape
    kernel = align_device_type(kernel, img)
    if kernel.ndim == 1:
        kernel = kernel.view(1, 1, 1, kernel.shape[0])
        kernel = kernel.repeat(num_ch, 1, 1, 1)
    elif kernel.ndim == 2:
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        kernel = kernel.repeat(num_ch, 1, 1, 1)
    elif kernel.ndim == 3:
        kernel = kernel.unsqueeze(1)
    kernel = kernel.contiguous()
    #
    if padding == 'same':
        padding = calc_padding(kernel.shape[2:])
    elif isinstance(padding, int):
        padding = (padding,) * 4
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    #
    if padding is None:
        res = conv2d(img, weight=kernel, groups=num_ch)
    elif mode == 'constant' and (
        padding[0] == padding[1] and padding[2] == padding[3]
    ):
        res = conv2d(img, weight=kernel, padding=padding[::2], groups=num_ch)
    else:
        img = pad(img, padding, mode)
        res = conv2d(img, weight=kernel, groups=num_ch)
    if is_single_image:
        res = res.squeeze_(0)
    return res


def deg_to_rad(deg: torch.Tensor):
    """Convert the angle unit from degree to radian.

    Parameters
    ----------
    deg : torch.Tensor
        Degree values.

    Returns
    -------
    torch.Tensor
        Radian values.
    """
    rad = deg.mul(180 / torch.pi)
    return rad


def rad_to_deg(deg: torch.Tensor):
    """Convert the angle unit from radian to degree.

    Parameters
    ----------
    deg : torch.Tensor
        Radian values.

    Returns
    -------
    torch.Tensor
        Degree values.
    """
    rad = deg.mul(180 / torch.pi)
    return rad
