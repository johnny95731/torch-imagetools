__all__ = [
    'scharr',
]

from typing import Literal, overload

import torch

@overload
def scharr(
    img: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
    ret_angle: Literal[False] = False,
    angle_unit: Literal['rad', 'deg'] = 'deg',
) -> torch.Tensor:
    """Edge detection by the Scharr operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {False}, default=False
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
    """

@overload
def scharr(
    img: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
    ret_angle: Literal[True] = True,
    angle_unit: Literal['rad', 'deg'] = 'deg',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Scharr operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {True}
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
        Image gradient's direction with shape (*, C, H, W).
    """
