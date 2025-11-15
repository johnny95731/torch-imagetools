__all__ = [
    'matrix_transform',
    'filter2d',
    'atan2',
    'p_norm',
]

import torch

from ..utils.helpers import align_device_type


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
    matrix = align_device_type(matrix, img)
    output = torch.einsum('oc,...chw->...ohw', matrix, img)
    return output


def filter2d(
    img: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Image convolution with a 2D kernel.

    Parameters
    ----------
    img : torch.Tensor
        Image, a tensor with shape (*, C, H, W).
    kernel : torch.Tensor
        A convolution kernel with shape (k_x,), (k_y, k_x),
        (1 or k * C, k_y, k_x), or (B, k * C, k_y, k_x), where k is a positive
        integer.

    Returns
    -------
    torch.Tensor
        The image with shape (*, C, H, W).
    """
    is_single_image = img.ndim == 3
    if is_single_image:
        img = img.unsqueeze(0)
    num_ch = img.size(-3)

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

    y, x = kernel.shape[2:]
    padding = ((y - 1) // 2, (x - 1) // 2)
    res = torch.nn.functional.conv2d(
        img,
        weight=kernel,
        padding=padding,
        groups=num_ch,
    )
    return res[0] if is_single_image else res


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
        angle.mul_(180 / torch.pi)
    return angle


def p_norm(
    img: torch.Tensor,
    p: float | str,
) -> torch.Tensor:
    """Computes the p-norm of an image.

    Parameters
    ----------
    img : torch.Tensor
        Image, a tensor with shape (*, C, H, W).
    p : float | string of float
        The exponent value.

    Returns
    -------
    torch.Tensor
        The p-norm value with shape (*,).
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
