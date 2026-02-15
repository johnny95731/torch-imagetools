__all__ = [
    'matrix_transform',
    '_check_ksize',
    'calc_padding',
    'filter2d',
    'atan2',
    'p_norm',
    'pca',
]

from math import ceil, floor

import torch
from torch.nn.functional import conv2d, pad

from ..utils.helpers import align_device_type, check_valid_image_ndim


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
        The transformation matrix with shape `(C_out, C)`.

    Returns
    -------
    torch.Tensor
        The image with shape `(*, C_out, H, W)`.
    """
    matrix = align_device_type(matrix, img)
    output = torch.einsum('oc,...chw->...ohw', matrix, img)
    return output


def _check_ksize(
    ksize: int | tuple[int, int],
    positive: bool = True,
) -> tuple[int, int]:
    """_summary_

    Parameters
    ----------
    ksize : int | tuple[int, int]
        _description_
    positive : bool, optional
        _description_, by default True

    Returns
    -------
    tuple[int, int]
        _description_
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
    padding: list[int] | str | None = 'same',
    mode: str = 'reflect',
) -> torch.Tensor:
    """Image convolution with a 2D kernel.

    Parameters
    ----------
    img : torch.Tensor
        Image, a tensor with shape `(*, C, H, W)`.
    kernel : torch.Tensor
        A convolution kernel with shape `(k_x,)`, `(k_y, k_x)`,
        (1 or k * C, k_y, k_x), or (B, k * C, k_y, k_x), where k is a positive
        integer.
    padding : list[int] | 'same' | None, default='same'
        The padding size.

        - list[int]: padding size: `(left, right, top, bottom)`. See `torch.nn.functional.pad`.
        - 'same': computes from kernel size.
        - None: filter without pad.
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

    kernel = align_device_type(kernel, img)
    if kernel.ndim == 1:
        kernel = kernel.view(1, 1, 1, kernel.shape[0])
        kernel = kernel.repeat(num_ch, 1, 1, 1)
    elif kernel.ndim == 2:
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        kernel = kernel.repeat(num_ch, 1, 1, 1)
    elif kernel.ndim == 3:
        kernel = kernel.repeat(num_ch, 1, 1).unsqueeze_(1)
    kernel = kernel.contiguous()

    if padding == 'same':
        padding = calc_padding(kernel.shape[2:])
    if padding is not None:
        img = pad(img, padding, mode)
    res = conv2d(img, weight=kernel, groups=num_ch)
    if is_single_image:
        res = res.squeeze(0)
    return res


def atan2(
    y: torch.Tensor,
    x: torch.Tensor,
    angle_unit: str = 'deg',
) -> torch.Tensor:
    """Computes the direction of an image gradient.

    Parameters
    ----------
    y : torch.Tensor
        The y-component.
    x : torch.Tensor
        The x-component.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.

    Returns
    -------
    torch.Tensor
        The angle between the vector and x-axis.

    Raises
    ------
    ValueError
        If angle_unit is neither 'rad' or 'deg'.
    """
    if angle_unit != 'deg' and angle_unit != 'rad':
        raise ValueError(
            f"angle_unit must be 'rad' or 'deg', but got {angle_unit}."
        )
    angle = torch.atan2(y, x)
    if angle_unit == 'deg':
        angle.mul(180 / torch.pi)
    return angle


def p_norm(
    img: torch.Tensor,
    p: float | str,
) -> torch.Tensor:
    """Computes the p-norm of an image.

    Parameters
    ----------
    img : torch.Tensor
        Image, a tensor with shape `(*, C, H, W)`.
    p : float | string of float
        The exponent value.

    Returns
    -------
    torch.Tensor
        The p-norm value with shape `(*,)`.
    """
    if isinstance(p, str):
        p = float(p)

    img = img.abs()
    if p == float('inf'):
        res = img.amax(dim=(-3, -2, -1))
    elif p == float('-inf'):
        res = img.amin(dim=(-3, -2, -1))
    else:
        res = (img**p).sum(dim=(-3, -2, -1))
        res = res ** (1 / p)
    return res


def pca(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Image PCA.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`

    Returns
    -------
    L : torch.Tensor
        Eigenvalues in ascending order.
    Vt : torch.Tensor
        Corresponding eigenvectors.
    """
    check_valid_image_ndim(img)
    is_float16 = img.dtype == torch.float16
    if is_float16 or not torch.is_floating_point(img):
        img = img.float()
    flatted = img.flatten(-2)
    # Covariance
    mean = flatted.mean(dim=-1, keepdim=True)
    cov = (flatted @ flatted.movedim(-1, -2)) / (flatted.size(-1) - 1)
    cov -= mean * mean.movedim(-1, -2)

    L, Vt = torch.linalg.eigh(cov)  # noqa: N806
    if is_float16:
        L = L.type(torch.float16)  # noqa: N806
        Vt = Vt.type(torch.float16)  # noqa: N806
    return L, Vt
