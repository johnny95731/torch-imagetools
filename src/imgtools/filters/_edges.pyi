__all__ = [
    'gradient_magnitude',
    'laplacian',
    'robinson',
    'kirsch',
    'difference',
    'prewitt',
    'sobel',
    'scharr',
]

from typing import Literal, overload

import torch

def gradient_magnitude(
    grad_y: torch.Tensor,
    grad_x: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
) -> torch.Tensor: ...

#
def laplacian(
    img: torch.Tensor,
    diagonal: bool = False,
    inflection_only: bool = False,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor: ...

#
@overload
def robinson(
    img: torch.Tensor,
    ret_angle: Literal[False] = False,
    angle_unit: Literal['rad', 'deg'] = 'deg',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor:
    """Edge detection by the Robinson compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {False}, default=False
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude. The value is the maximum along all compass
        kernel.
    """

@overload
def robinson(
    img: torch.Tensor,
    ret_angle: Literal[True] = True,
    angle_unit: Literal['rad', 'deg'] = 'deg',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Robinson compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {True}, default=True
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude. The value is the maximum along all compass
        kernel.
    torch.Tensor
        Image gradient's direction with shape `(*, C, H, W)`.
    """

#
@overload
def kirsch(
    img: torch.Tensor,
    ret_angle: Literal[False] = False,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor:
    """Edge detection by the Kirsch compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {False}, default=False
        Returns the direction of gradient or not.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude. The value is the maximum along all compass
        kernel.
    """

@overload
def kirsch(
    img: torch.Tensor,
    ret_angle: Literal[True] = True,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Kirsch compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {True}, default=True
        Returns the direction of gradient or not.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude. The value is the maximum along all compass
        kernel.
    torch.Tensor
        Image gradient's direction with shape `(*, C, H, W)`.
    """

#
@overload
def difference(
    img: torch.Tensor,
    magnitude: str | int | float = 2,
    ret_angle: Literal[False] = False,
    angle_unit: str = 'deg',
)-> torch.Tensor:...
@overload
def difference(
    img: torch.Tensor,
    magnitude: str | int | float = 2,
    ret_angle: Literal[True] = True,
    angle_unit: str = 'deg',
) -> tuple[torch.Tensor, torch.Tensor]:...
    
@overload
def prewitt(
    img: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
    ret_angle: Literal[False] = False,
    angle_unit: Literal['rad', 'deg'] = 'deg',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor:
    """Edge detection by the Prewitt operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {False}, default=False
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude.\\
        The magnitude stacks (y-direction, x-direction) if `magnitude`
        is 'stack'.\\
        For details, check `torch_imagetools.filters.gradient_magnitude`.
    """

@overload
def prewitt(
    img: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
    ret_angle: Literal[True] = True,
    angle_unit: Literal['rad', 'deg'] = 'deg',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Prewitt operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {True}
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude.\\
        The magnitude stacks (y-direction, x-direction) if `magnitude`
        is 'stack'.\\
        For details, check `torch_imagetools.filters.gradient_magnitude`.
    torch.Tensor
        Image gradient's direction with shape `(*, C, H, W)`.
    """

#
@overload
def sobel(
    img: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
    ret_angle: Literal[False] = False,
    angle_unit: Literal['rad', 'deg'] = 'deg',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor:
    """Edge detection by the Sobel operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {False}, default=False
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude.\\
        The magnitude stacks (y-direction, x-direction) if `magnitude`
        is 'stack'.\\
        For details, check `torch_imagetools.filters.gradient_magnitude`.
    """

@overload
def sobel(
    img: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
    ret_angle: Literal[True] = True,
    angle_unit: Literal['rad', 'deg'] = 'deg',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Sobel operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {True}
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude.\\
        The magnitude stacks (y-direction, x-direction) if `magnitude`
        is 'stack'.\\
        For details, check `torch_imagetools.filters.gradient_magnitude`.
    torch.Tensor
        Image gradient's direction with shape `(*, C, H, W)`.
    """

#
@overload
def scharr(
    img: torch.Tensor,
    magnitude: Literal['stack', 'inf', '-inf'] | int | float = 2,
    ret_angle: Literal[False] = False,
    angle_unit: Literal['rad', 'deg'] = 'deg',
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> torch.Tensor:
    """Edge detection by the Scharr operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {False}, default=False
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

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
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'reflect',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Scharr operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        Norm for computing gradient's magnitude.
    ret_angle : {True}
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude.\\
        The magnitude stacks (y-direction, x-direction) if `magnitude`
        is 'stack'.\\
        For details, check `torch_imagetools.filters.gradient_magnitude`.
    torch.Tensor
        Image gradient's direction with shape `(*, C, H, W)`.
    """
