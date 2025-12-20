__all__ = [
    'xyz_to_luv',
    'luv_to_xyz',
    'rgb_to_luv',
    'luv_to_rgb',
]

import torch

from ..utils.helpers import align_device_type
from ._cielab import _6_29_POW3
from ._ciexyz import (
    get_rgb_to_xyz_matrix,
    rgb_to_xyz,
    xyz_to_rgb,
)

_SCALING_LUV = 0.01 / (((6 / 29) / 2) ** 3)  # = (29 / 3)**3 / 100


def _calc_uv_prime(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    # 4.0 / (x + 15 * y + 3 * z)
    coeff = x.add(y, alpha=15.0).add(z, alpha=3.0)
    torch.divide(4.0, coeff, out=coeff)
    coeff = coeff.nan_to_num(0.0, 0.0, 0.0)

    u_prime = x * coeff
    v_prime = 2.25 * y * coeff
    return u_prime, v_prime


def xyz_to_luv(
    xyz: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to CIE LUV space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    luv : torch.Tensor
        An image in CIE LUV space with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    x, y, z = xyz.unbind(-3)

    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)
    max_ = matrix.sum(dim=1)
    max_ = align_device_type(max_, xyz)

    u_white, v_white = _calc_uv_prime(*max_)
    u_prime, v_prime = _calc_uv_prime(x, y, z)

    l = y / max_[1]
    l = torch.where(
        l > _6_29_POW3,
        l.pow(1 / 3).mul(1.16).sub(0.16),
        l * _SCALING_LUV,
    )

    l_13 = 13.0 * l
    u = (u_prime - u_white) * l_13
    v = (v_prime - v_white) * l_13

    luv = torch.stack((l, u, v), dim=-3)
    if ret_matrix:
        return luv, matrix
    return luv


def luv_to_xyz(
    luv: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
):
    """Converts an image from CIE LUV space to CIE XYZ space.

    Parameters
    ----------
    luv : torch.Tensor
        An image in CIE LUV space with shape (*, 3, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    xyz : torch.Tensor
        An image in CIE XYZ space with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    l, u, v = luv.unbind(-3)

    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)
    max_ = matrix.sum(dim=1)
    max_ = align_device_type(max_, luv)

    u_white, v_white = _calc_uv_prime(*max_)

    l_13 = 1 / (13.0 * l)
    u_prime = (u * l_13) + u_white
    v_prime = (v * l_13) + v_white

    y = torch.where(
        l > 0.08,  # = _6_29_POW3 ** (1/3) * 1.16 - 0.16
        l.add(0.16).div(1.16).pow(3.0),
        l.div(_SCALING_LUV),
    )
    y = max_[1] * y

    u_v = (u_prime / v_prime).nan_to_num(0.0, 0.0, 0.0)
    # x = 2.25 * u_prime / v_prime * y
    x = 2.25 * u_v * y
    # z = (3.0 - 0.75 * u_prime - 5.0 * v_prime) / v_prime * y
    z = (3.0 / v_prime - 0.75 * u_v - 5.0) * y

    xyz = torch.stack((x, y, z), dim=-3)
    if ret_matrix:
        return xyz, matrix
    return xyz


def rgb_to_luv(
    rgb: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor:
    """Converts an image from RGB space to CIE LUV space.

    The input is assumed to be in the range of [0, 1]. If rgb_spec is a
    tensor, then the input rgb is assumed to be linear RGB.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    luv : torch.Tensor
        An image in CIE LUV space with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    xyz = rgb_to_xyz(rgb, rgb_spec, white, obs)
    luv, matrix = xyz_to_luv(xyz, rgb_spec, white, obs, ret_matrix=True)
    if ret_matrix:
        return luv, matrix
    return luv


def luv_to_rgb(
    luv: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor:
    """Converts an image from CIE LUV space to RGB space.

    Parameters
    ----------
    luv : torch.Tensor
        An image in CIE LUV space with shape (*, 3, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    rgb : torch.Tensor
        An RGB image in [0, 1] with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert CIE XYZ to RGB.
        `mat` is returned only if `ret_matrix` is true.
    """
    xyz = luv_to_xyz(luv, rgb_spec, white, obs)  # type: torch.Tensor
    rgb, matrix = xyz_to_rgb(xyz, rgb_spec, white, obs, ret_matrix=True)
    if ret_matrix:
        return rgb, matrix
    return rgb
