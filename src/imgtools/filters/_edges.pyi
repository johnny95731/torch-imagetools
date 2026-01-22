__all__ = [
    'gradient_magnitude',
    'laplacian',
    'robinson',
    'kirsch',
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
) -> torch.Tensor: ...

#
@overload
def robinson(
    img: torch.Tensor,
    ret_angle: Literal[False] = False,
) -> torch.Tensor:
    """Edge detection by the Robinson compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {False}, default=False
        Returns the direction of gradient or not.

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Robinson compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {True}, default=True
        Returns the direction of gradient or not.

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
) -> torch.Tensor:
    """Edge detection by the Kirsch compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {False}, default=False
        Returns the direction of gradient or not.

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Kirsch compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ret_angle : {True}, default=True
        Returns the direction of gradient or not.

    Returns
    -------
    torch.Tensor
        Image gradient's magnitude. The value is the maximum along all compass
        kernel.
    torch.Tensor
        Image gradient's direction with shape `(*, C, H, W)`.
    """
