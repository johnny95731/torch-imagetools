from typing import Literal

import torch
from torch_imagetools.color.rgb import RGBSpec, gammaize_rgb

from .ciexyz import (
    StandardIlluminants,
    get_rgb_to_xyz_matrix,
    rgb_to_xyz,
    xyz_to_rgb,
)

_6_29 = 6 / 29  # threshold for _lab_helper_inv
_6_29_pow3 = _6_29**3  # threshold for _lab_helper
_scaling = 0.01 / ((_6_29 / 2) ** 3)  # = (29 / 3)**3 / 100


def _luv_helper(value: torch.Tensor):
    """Function that be used in the transformation from CIE XYZ to CIE LUV."""
    output = torch.where(
        value > _6_29_pow3,
        value.pow(1 / 3).mul_(1.16).sub_(0.16),
        value.mul(_scaling),
    )
    return output


def _luv_helper_inv(
    value: torch.Tensor,
):
    """Function that be used in the transformation from CIE LUV to CIE XYZ."""
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
    """Converts an image from CIE XYZ space to CIE LUV space.

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
        An image in CIE LUV space with the shape (*, 3, H, W). If ret_matrix
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
    """Converts an image from CIE LUV space to CIE XYZ space.

    Parameters
    ----------
    luv : torch.Tensor
        An image in CIE LUV space with shape (*, 3, H, W).
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
    """Converts an image from RGB space to CIE LUV space.

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
        An image in CIE LUV space with the shape (*, 3, H, W). If ret_matrix
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
    """Converts an image from CIE LUV space to RGB space.

    Parameters
    ----------
    luv : torch.Tensor
        An image in CIE LUV space with shape (*, 3, H, W).
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


if __name__ == '__main__':
    from timeit import timeit

    img = (
        torch.randint(0, 256, (3, 1500, 1500)).type(torch.float32).mul_(1 / 255)
    )
    num = 10

    xyz = rgb_to_xyz(img)
    luv = xyz_to_luv(xyz)
    ret = luv_to_xyz(luv)

    d = torch.abs(ret - xyz)
    print('Error:', torch.max(d).item())
    print(timeit('xyz_to_luv(xyz)', number=num, globals=locals()))
    print(timeit('luv_to_xyz(luv)', number=num, globals=locals()))

    luv = rgb_to_luv(img)
    ret = luv_to_rgb(luv)

    d = torch.abs(ret - img)
    print('Error:', torch.max(d).item())
    print(timeit('rgb_to_luv(img)', number=num, globals=locals()))
    print(timeit('luv_to_rgb(luv)', number=num, globals=locals()))

    w = [
        'A',
        'B',
        'C',
        'D50',
        'D55',
        'D65',
        'D75',
        'E',
        'F1',
        'F2',
        'F3',
        'F4',
        'F5',
        'F6',
        'F7',
        'F8',
        'F9',
        'F10',
        'F11',
        'F12',
    ]
    rgbs = [
        'srgb',
        'adobergb',
        'prophotorgb',
        'rec2020',
        'displayp3',
        'widegamut',
        'ciergb',
    ]
    # for rgb in rgbs:
    #     for white in w:
    #         xyz = rgb_to_xyz(img, rgb, white, 2)
    #         ret = xyz_to_rgb(xyz, rgb, white, 2)
    #         d2 = torch.abs(ret - img)

    #         xyz = rgb_to_xyz(img, rgb, white, 10)
    #         ret = xyz_to_rgb(xyz, rgb, white, 10)
    #         d10 = torch.abs(ret - img)

    #         title = f'{rgb}/{white}'
    #         print(f'{title:<16}', torch.max(d2).item(), torch.max(d10).item())
