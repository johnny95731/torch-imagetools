__all__ = [
    'gradient_magnitude',
    'laplacian',
    'robinson',
    'kirsch',
]

import torch

from ..utils.helpers import align_device_type
from ..utils.math import atan2, filter2d


def gradient_magnitude(
    grad_y: torch.Tensor,
    grad_x: torch.Tensor,
    magnitude: str | int | float = 2,
) -> torch.Tensor:
    """Computes the magnitude of the gradients: norm(gradients)

    Parameters
    ----------
    grad_y
        The derivatives with respect to y of an image.
    grad_x
        The derivatives with respect to x of an image.
    magnitude : {'stack', 'inf', '-inf'} | int | float, default=2
        The strategy of magnitude computation.
        'stack' : Stack derivatives with fusion.
        'inf' : Taking Supremum norm. Preserves the maximum alone all
                derivatives.
        '-inf' : Taking the minimum alone all derivatives.
        int or float : Applying p-norm to the derivatives if p > 0.

    Returns
    -------
    torch.Tensor
        The magnitude of gradient.\\
        The tensor has shape (*, C, H, W) if `magnitude` is not 'stack'.
        Otherwise, with shape (N, *, C, H, W) where N is the number of the
        derivatives.

    Raises
    ------
    ValueError
        When magnitude <= 0.
    TypeError
        When the type of magnitude is not one of None, 'inf', float, and int.
    """
    if isinstance(magnitude, str) and magnitude == 'stack':
        mag = torch.stack((grad_y, grad_x))
        return mag
    elif isinstance(magnitude, str):
        magnitude = float(magnitude)
    elif isinstance(magnitude, int):
        magnitude = float(magnitude)

    mag = torch.stack((grad_y, grad_x))
    mag = mag.abs()
    if magnitude == float('inf'):
        mag = torch.amax(mag, dim=0)
    elif magnitude == float('-inf'):
        mag = torch.amin(mag, dim=0)
    elif isinstance(magnitude, float):
        if magnitude < 0:
            raise ValueError(
                "Argument `magnitude` must be 'stack', 'inf', '-inf', or a "
                + f'positive number, but got {magnitude}'
            )
        if magnitude == 1.0:
            mag = mag.sum(0)
        elif magnitude == 2.0:
            mag = mag.square().sum(0).sqrt()
        else:
            mag = mag.pow(magnitude).sum(0).pow(1 / magnitude)
    else:
        raise TypeError(
            "Argument `magnitude` must be 'stack', 'inf', '-inf', or a "
            + f'positive number, but got {magnitude}'
        )
    return mag


def laplacian(
    img: torch.Tensor,
    diagonal: bool = False,
    inflection_only: bool = False,
) -> torch.Tensor:
    """Computes the laplacian of an image.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape (*, C, H, W).
    diagonal : bool, default=False
        The kernel detects 45 degree and 135 degree.
    inflection_only : bool, default=False
        Filtering to remove non-inflection points.
        A inflection point means that the sign of laplacian changes near
        the point.

    Returns
    -------
    torch.Tensor
        The laplacian of an image with shape (*, C, H, W)..
    """
    if not diagonal:
        kernel = torch.tensor((
            (0, -1, 0),
            (-1, 4, -1),
            (0, -1, 0),
        ))
    else:
        kernel = torch.tensor((
            (-1, -1, -1),
            (-1, 8, -1),
            (-1, -1, -1),
        ))
    kernel = align_device_type(kernel, img)

    grad = filter2d(img, kernel)
    if inflection_only:
        padded = torch.nn.functional.pad(grad, (1, 1, 1, 1))
        x_check = padded[..., 1:-1, :-2] * padded[..., 1:-1, 2:] < 0.0
        y_check = padded[..., :-2, 1:-1] * padded[..., 2:, 1:-1] < 0.0
        mask = x_check | y_check
        grad = torch.where(mask, grad, 0.0)
    return grad


def robinson(
    img: torch.Tensor,
    ret_angle: bool = False,
    angle_unit: str = 'deg',
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Robinson compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    ret_angle : bool, default=False
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.

    Returns
    -------
    mag : torch.Tensor
        Image gradient's magnitude. The value is the maximum along all compass
        kernel.
    angle : torch.Tensor
        Image gradient's direction with shape (*, C, H, W).
        `angle` is returned only if `ret_angle` is true.
    """
    kernel_y = torch.tensor((
        (-1, -2, -1),
        (0, 0, 0),
        (1, 2, 1),
    ))
    kernel_45 = torch.tensor((
        (-2, -1, 0),
        (-1, 0, 1),
        (0, 1, 2),
    ))
    kernel_y = align_device_type(kernel_y, img)
    kernel_45 = align_device_type(kernel_45, img)

    grad_y = filter2d(img, kernel_y)
    grad_x = filter2d(img, kernel_y.T)
    grad_45 = filter2d(img, kernel_45)
    grad_135 = filter2d(img, kernel_45.flip(0))

    mag = torch.stack((grad_y, grad_x, grad_45, grad_135))
    mag = mag.abs().amax(dim=0)
    if ret_angle:
        angle = atan2(grad_y, grad_x, angle_unit=angle_unit)
        return mag, angle
    return mag


def kirsch(
    img: torch.Tensor,
    ret_angle: bool = False,
    angle_unit: str = 'deg',
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Edge detection by the Kirsch compass operators.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    ret_angle : bool, default=False
        Returns the direction of gradient or not.
    angle_unit : {'rad', 'deg'}, default='deg'
        The representation of angle is in radian or in degree.

    Returns
    -------
    mag : torch.Tensor
        Image gradient's magnitude. The value is the maximum along all compass
        kernel.
    angle : torch.Tensor
        Image gradient's direction with shape (*, C, H, W).
        `angle` is returned only if `ret_angle` is true.
    """
    kernel_y = torch.tensor((
        (-3, -3, -3),
        (-3, 0, -3),
        (5, 5, 5),
    ))
    kernel_45 = torch.tensor((
        (-3, -3, -3),
        (-3, 0, 5),
        (-3, 5, 5),
    ))
    kernel_y = align_device_type(kernel_y, img)
    kernel_45 = align_device_type(kernel_45, img)

    # flipped kernel
    kernel_y2 = kernel_y.flip(0)
    kernel_135 = kernel_y.flip(1)

    grad_south = filter2d(img, kernel_y)
    grad_north = filter2d(img, kernel_y2)
    grad_east = filter2d(img, kernel_y.T)
    grad_west = filter2d(img, kernel_y2.T)

    grad_se = filter2d(img, kernel_45)
    grad_sw = filter2d(img, kernel_135)
    grad_ne = filter2d(img, kernel_45.flip(0))
    grad_nw = filter2d(img, kernel_135.flip(0))

    mag = torch.stack((
        grad_south,
        grad_north,
        grad_east,
        grad_west,
        grad_se,
        grad_sw,
        grad_ne,
        grad_nw,
    ))
    mag = mag.abs().amax(dim=0)
    if ret_angle:
        angle = atan2(grad_south, grad_east, angle_unit=angle_unit)
        return mag, angle
    return mag
