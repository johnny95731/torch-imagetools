__all__ = [
    'xyz_to_luv',
    'luv_to_xyz',
    'rgb_to_luv',
    'luv_to_rgb',
]

from typing import Literal

import torch

from .rgb import RGBSpec, gammaize_rgb
from .ciexyz import (
    StandardIlluminants,
    get_rgb_to_xyz_matrix,
    rgb_to_xyz,
    xyz_to_rgb,
)

_6_29 = 6 / 29  # threshold for _luv_helper_inv
_6_29_pow3 = _6_29**3  # threshold for _luv_helper
_scaling = 0.01 / ((_6_29 / 2) ** 3)  # = (29 / 3)**3 / 100


def _luv_helper(value: torch.Tensor):
    """Function that be used in the transformation from CIE XYZ to CIE Luv."""
    output = torch.where(
        value > _6_29_pow3,
        value.pow(1 / 3).mul_(1.16).sub_(0.16),
        value.mul(_scaling),
    )
    return output


def _luv_helper_inv(
    value: torch.Tensor,
):
    """Function that be used in the transformation from CIE Luv to CIE XYZ."""
    output = torch.where(
        value > 0.08,
        value.add(0.16).mul_(1 / 1.16).pow_(3.0),
        value.mul(1 / _scaling),
    )
    return output


def _calc_uv_prime(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    # 4.0 / (x + 15 * y + 3 * z)
    coeff = x.add(y, alpha=15.0).add_(z, alpha=3.0)
    torch.divide(4.0, coeff, out=coeff)
    coeff.nan_to_num_(0.0, 0.0, 0.0)
    u_prime = coeff.mul(x)  # x * coeff
    v_prime = coeff.mul_(y).mul_(2.25)  # 2.25 * y * coeff
    return u_prime, v_prime


def xyz_to_luv(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to CIE Luv space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    ret_matrix : bool, optional
        If True, returns image and conversion matrix (rgb -> xyz).
        By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An image in CIE Luv space with the shape (*, 3, H, W). If ret_matrix
        is True, returns image and the transformation matrix.
    """
    x, y, z = xyz.unbind(-3)

    matrix = (
        get_rgb_to_xyz_matrix(rgb_spec, white, obs)
        if not torch.is_tensor(rgb_spec)
        else rgb_spec
    )
    max_ = matrix.sum(dim=1)

    u_white, v_white = _calc_uv_prime(*max_)
    u_prime, v_prime = _calc_uv_prime(x, y, z)

    l = _luv_helper(y / max_[1])
    l_13 = 13.0 * l
    u = u_prime.sub_(u_white).mul_(l_13)
    v = v_prime.sub_(v_white).mul_(l_13)

    luv = torch.stack((l, u, v), dim=-3)
    if ret_matrix:
        return luv, matrix
    return luv


def luv_to_xyz(
    luv: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor:
    """Converts an image from CIE Luv space to CIE XYZ space.

    Parameters
    ----------
    luv : torch.Tensor
        An image in CIE Luv space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    ret_matrix : bool, optional
        If True, returns image and conversion matrix (rgb -> xyz).
        By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An image in CIE XYZ space with the shape (*, 3, H, W). If ret_matrix
        is True, returns image and the transformation matrix.
    """
    l, u, v = luv.unbind(-3)

    matrix = (
        get_rgb_to_xyz_matrix(rgb_spec, white, obs)
        if not torch.is_tensor(rgb_spec)
        else rgb_spec
    )
    max_ = matrix.sum(dim=1)

    u_white, v_white = _calc_uv_prime(*max_)

    l_13 = 1 / (13.0 * l)
    u_prime = (u * l_13).add_(u_white)
    v_prime = (v * l_13).add_(v_white)

    y = max_[1] * _luv_helper_inv(l)
    # x = 2.25 * u_prime / v_prime * y
    x = u_prime.divide(v_prime).mul_(y).mul_(2.25).nan_to_num_(0.0)
    # z = (3.0 - 0.75 * u_prime - 5.0 * v_prime) / v_prime * y
    z = (
        u_prime.mul_(-0.75)
        .sub_(v_prime, alpha=5.0)
        .add_(3.0)
        .divide_(v_prime)
        .mul_(y)
        .nan_to_num_(0.0)
    )

    xyz = torch.stack((x, y, z), dim=-3)
    if ret_matrix:
        return xyz, matrix
    return xyz


def rgb_to_luv(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor:
    """Converts an image from RGB space to CIE Luv space.

    The input is assumed to be in the range of [0, 1]. If rgb_spec is a
    tensor, then the input rgb is assumed to be linear RGB.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type. If rgb_spec is a
        tensor, then the input rgb is assumed to be in linear RGB space.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    ret_matrix : bool, optional
        If True, returns image and conversion matrix (rgb -> xyz).
        By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An image in CIE Luv space with the shape (*, 3, H, W). If ret_matrix
        is True, returns image and the transformation matrix.
    """
    xyz, matrix = rgb_to_xyz(rgb, rgb_spec, white, obs, ret_matrix=True)
    luv = xyz_to_luv(xyz, matrix)
    if ret_matrix:
        return luv, matrix
    return luv


def luv_to_rgb(
    luv: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor:
    """Converts an image from CIE Luv space to RGB space.

    Parameters
    ----------
    luv : torch.Tensor
        An image in CIE Luv space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    ret_matrix : bool, optional
        If True, returns image and conversion matrix (rgb -> xyz).
        By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An RGB image in the range of [0, 1] with the shape (*, 3, H, W). If
        rgb_spec is a tensor, then the image is in linear RGB space. If
        ret_matrix is True, returns image and the transformation matrix.
    """
    xyz, matrix = luv_to_xyz(luv, rgb_spec, white, obs, ret_matrix=True)
    matrix = matrix.inverse()
    rgb = xyz_to_rgb(xyz, matrix)

    if not torch.is_tensor(rgb_spec):
        gammaize_rgb(rgb, rgb_spec, out=rgb)
    if ret_matrix:
        return rgb, matrix
    return rgb
