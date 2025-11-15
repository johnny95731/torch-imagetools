__all__ = [
    'prewitt',
]

import torch

from ..utils.math import atan2, filter2d
from ..utils.helpers import align_device_type
from ._edges import gradient_magnitude


def prewitt(
    img: torch.Tensor,
    magnitude: str | int | float = 2,
    ret_angle: bool = False,
    angle_unit: str = 'deg',
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Prewitt operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : bool, default=False
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude.\\
        The magnitude stacks (y-direction, x-direction) if `magnitude`
        is 'stack'.\\
        For details, check `torch_imagetools.filters.gradient_magnitude`.
    torch.Tensor
        Image gradient's direction with shape (*, C, H, W) if `ret_angle` is
        True. Otherwise, returns magnitude only.
    """
    kernel_y = torch.tensor((
        (-1, -1, -1),
        (0, 0, 0),
        (1, 1, 1),
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
