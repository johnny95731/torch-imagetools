from typing import Literal, overload

import torch

from ..utils.math import atan2, filter2d
from ..utils.helpers import align_device_type
from .edges import gradient_magnitude


@overload
def sobel(
    img: torch.Tensor,
    *,
    magnitude: Literal['cat', 'inf'] | int | float = 2,
    ret_angle: Literal[False] = False,
    angle_unit: Literal['rad', 'deg'] = 'deg',
) -> torch.Tensor:
    """Edge detection by the Sobel operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    magnitude : Literal['cat', 'inf'] | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : Literal[False], default=False
        Returns the direction of gradient or not.
    angle_unit : Literal['rad', 'deg'], default='deg'
        Represents the angle in radian or in degree.

    Returns
    -------
    torch.Tensor
        When ret_angle is False, returns gradient's magnitude.
        With shape (*, C, H, W) if `magnitude` is not 'cat'; otherwise,
        with shape (2, *, C, H, W) and index zero is the y-direction gradient.
    """


@overload
def sobel(
    img: torch.Tensor,
    *,
    magnitude: Literal['cat', 'inf'] | int | float = 2,
    ret_angle: Literal[True],
    angle_unit: Literal['rad', 'deg'] = 'deg',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Sobel operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    magnitude : Literal['cat', 'inf'] | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : Literal[True]
        Returns the direction of gradient or not.
    angle_unit : Literal['rad', 'deg'], default='deg'
        Represents the angle in radian or in degree.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The edge's magnitude and direction.
    """


def sobel(
    img: torch.Tensor,
    *,
    magnitude: Literal['cat', 'inf'] | int | float = 2,
    ret_angle: bool = False,
    angle_unit: Literal['rad', 'deg'] = 'deg',
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Sobel operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    magnitude : Literal['cat', 'inf'] | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : bool, default=False
        Returns the direction of gradient or not.
    angle_unit : Literal['rad', 'deg'], default='deg'
        Represents the angle in radian or in degree.

    Returns
    -------
    torch.Tensor
        When ret_angle is False, returns gradient's magnitude.
        With shape (*, C, H, W) if `magnitude` is not 'cat'; otherwise,
        with shape (2, *, C, H, W) and index zero is the y-direction gradient.
    tuple[torch.Tensor, torch.Tensor]
        When ret_angle is True, returns gradient's magnitude and direction.
        magnitude with shape (*, C, H, W) if `magnitude` is not 'cat';
        otherwise, with shape (2, *, C, H, W) and index zero is the y-direction gradient.
    """
    kernel_y = torch.tensor((
        (-1, -2, -1),
        (0, 0, 0),
        (1, 2, 1),
    ))
    kernel_y = align_device_type(kernel_y, img)
    # Note: Filtering twice with 2 directional kernel is faster than
    # filtering by a kernel that stacked 2 kernels
    grad_y = filter2d(img, kernel_y)
    grad_x = filter2d(img, kernel_y.T)

    mag = gradient_magnitude(grad_y, grad_x, magnitude=magnitude)
    if ret_angle:
        angle = atan2(grad_y, grad_x, angle_unit=angle_unit)
        return mag, angle
    return mag
