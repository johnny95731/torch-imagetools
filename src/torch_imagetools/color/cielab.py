__all__ = [
    'xyz_to_lab',
    'lab_to_xyz',
    'rgb_to_lab',
    'lab_to_rgb',
]

from typing import Literal, overload

import torch

from .rgb import RGBSpec, gammaize_rgb
from .ciexyz import (
    StandardIlluminants,
    get_rgb_to_xyz_matrix,
    rgb_to_xyz,
    xyz_to_rgb,
)

_6_29 = 6 / 29  # threshold for _lab_helper_inv
_6_29_pow3 = _6_29**3  # threshold for _lab_helper
_scaling = 1 / (3 * _6_29**2)  # = 1 / (3 * _6_29**2)
_bias = 4 / 29  # = 16 / 116


def _lab_helper(value: torch.Tensor):
    """Function that be used in the transformation from CIE XYZ to CIE LAB.
    The function maps [0, 1] into [4/29, 1] and is continuous.
    """
    output = torch.empty_like(value)
    torch.where(
        value > _6_29_pow3,
        value.pow(1 / 3),
        value.mul_(_scaling).add_(_bias),
        out=output,
    )
    return output


def _lab_helper_inv(value: torch.Tensor):
    """Function that be used in the transformation from CIE LAB to CIE XYZ.
    The function maps [4/29, 1] into [0, 1].
    """
    output = torch.where(
        value > _6_29,
        value.pow(3.0),
        value.sub(_bias).mul(1 / _scaling),
    )
    return output


@overload
def xyz_to_lab(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def xyz_to_lab(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...
def xyz_to_lab(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to CIE LAB space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, default='srgb'
        The RGB specification or a conversion matrix for transforming image
        from rgb to xyz. The string type is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If False, only the image is returned.
        If True, also return the matrix that maps image from rgb to xyz.

    Returns
    -------
    torch.Tensor
        An image in CIE LAB space with the shape (*, 3, H, W) when
        `ret_matrix` is False.
    tuple[torch.Tensor, torch.Tensor]
        An image and a transformation matrix when `ret_matrix` is True.\\
        The image is in CIE LAB space with the shape (*, 3, H, W).\\
        The matrix is 3x3 for mapping image from rgb to xyz.
    """
    x, y, z = xyz.unbind(-3)

    matrix = (
        get_rgb_to_xyz_matrix(rgb_spec, white, obs)
        if not torch.is_tensor(rgb_spec)
        else rgb_spec
    )
    max_ = matrix.sum(dim=1)

    fx = _lab_helper(x.mul(1 / max_[0]))
    fy = _lab_helper(y.mul(1 / max_[1]))
    fz = _lab_helper(z.mul(1 / max_[2]))
    l = 1.16 * fy - 0.16
    a = fx.sub_(fy).mul_(5.0)
    b = fz.sub_(fy).mul_(-2.0)

    lab = torch.stack((l, a, b), dim=-3)
    if ret_matrix:
        return lab, matrix
    return lab


@overload
def lab_to_xyz(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def lab_to_xyz(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...
def lab_to_xyz(
    lab: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE LAB space to CIE XYZ space.

    Parameters
    ----------
    lab : torch.Tensor
        An image in CIE LAB space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, default='srgb'
        The RGB specification or a conversion matrix for transforming image
        from rgb to xyz. The string type is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If False, only the image is returned.
        If True, also return the matrix that maps image from rgb to xyz.

    Returns
    -------
    torch.Tensor
        An image in CIE XYZ space with the shape (*, 3, H, W) when
        `ret_matrix` is False.
    tuple[torch.Tensor, torch.Tensor]
        An image and a transformation matrix when `ret_matrix` is True.\\
        The image is in CIE XYZ space with the shape (*, 3, H, W).\\
        The matrix is 3x3 for mapping image from rgb to xyz.
    """
    l, a, b = lab.unbind(-3)

    matrix = (
        get_rgb_to_xyz_matrix(rgb_spec, white, obs)
        if not torch.is_tensor(rgb_spec)
        else rgb_spec
    )
    max_ = matrix.sum(dim=1)

    l = l.add(0.16).mul_(1 / 1.16)
    x = _lab_helper_inv(l.add(a, alpha=0.2)).mul_(max_[0])
    y = _lab_helper_inv(l).mul(max_[1])
    z = _lab_helper_inv(l.sub(b, alpha=0.5)).mul_(max_[2])

    xyz = torch.stack((x, y, z), dim=-3)
    if ret_matrix:
        return xyz, matrix
    return xyz


@overload
def rgb_to_lab(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def rgb_to_lab(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...
def rgb_to_lab(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from RGB space to CIE LAB space.

    The input is assumed to be in the range of [0, 1]. If rgb_spec is a
    tensor, then the input rgb is assumed to be linear RGB.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, default='srgb'
        The RGB specification or a conversion matrix for transforming image
        from rgb to xyz. The string type is case-insensitive.\\
        If `rgb_spec` is a tensor, then the input rgb is assumed to be linear.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If False, only the image is returned.
        If True, also return the matrix that maps image from rgb to xyz.

    Returns
    -------
    torch.Tensor
        An image in CIE LAB space with the shape (*, 3, H, W) when
        `ret_matrix` is False.
    tuple[torch.Tensor, torch.Tensor]
        An image and a transformation matrix when `ret_matrix` is True.\\
        The image is in in CIE LAB space with the shape (*, 3, H, W).\\
        The matrix is 3x3 for mapping image from rgb to xyz.
    """
    xyz, matrix = rgb_to_xyz(rgb, rgb_spec, white, obs, ret_matrix=True)
    lab = xyz_to_lab(xyz, matrix)
    if ret_matrix:
        return lab, matrix
    return lab


def lab_to_rgb(
    lab: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE LAB space to RGB space.

    Parameters
    ----------
    lab : torch.Tensor
        An image in CIE LAB space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, default='srgb'
        The RGB specification or a conversion matrix for transforming image
        from rgb to xyz. The string type is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If False, only the image is returned.
        If True, also return the matrix that maps image from xyz to rgb.

    Returns
    -------
    torch.Tensor
        An RGB image in [0, 1] with the shape (*, 3, H, W) when
        `ret_matrix` is False.
    tuple[torch.Tensor, torch.Tensor]
        An RGB image and a transformation matrix when `ret_matrix` is True.\\
        The image is in [0, 1] with the shape (*, 3, H, W).\\
        The matrix is 3x3 for mapping image from xyz to rgb.
    """
    xyz, matrix = lab_to_xyz(lab, rgb_spec, white, obs, ret_matrix=True)
    matrix = matrix.inverse()
    rgb = xyz_to_rgb(xyz, matrix)

    if not torch.is_tensor(rgb_spec):
        gammaize_rgb(rgb, rgb_spec, out=rgb)
    if ret_matrix:
        return rgb, matrix
    return rgb
