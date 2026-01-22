__all__ = [
    'xyz_to_lab',
    'lab_to_xyz',
    'rgb_to_lab',
    'lab_to_rgb',
]

import torch

from ..utils.helpers import align_device_type, to_channel_coeff
from ._ciexyz import (
    get_rgb_to_xyz_matrix,
    rgb_to_xyz,
    xyz_to_rgb,
)

_6_29 = 6 / 29  # threshold for _lab_helper_inv
_6_29_POW3 = _6_29**3  # threshold for _lab_helper
_SCALING_LAB = 1 / (3 * _6_29**2)  # = (29 / 6)**2 / 3
_BIAS_LAB = 16 / 116


def _lab_helper(value: torch.Tensor):
    """Function that be used in the transformation from CIE XYZ to CIE LAB.
    The function maps [0, 1] into [4/29, 1] and is continuous.
    """
    output = torch.empty_like(value)
    torch.where(
        value > _6_29_POW3,
        value.pow(1 / 3),
        value.mul(_SCALING_LAB).add(_BIAS_LAB),
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
        value.sub(_BIAS_LAB).div(_SCALING_LAB),
    )
    return output


def xyz_to_lab(
    xyz: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to CIE LAB space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape `(*, 3, H, W)`.
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
    lab : torch.Tensor
        An image in CIE LAB space with the shape `(*, 3, H, W)`.
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)
    max_ = matrix.sum(dim=1)
    max_ = align_device_type(max_, xyz)
    max_ = to_channel_coeff(max_, 3)

    normalized = xyz.div(max_)
    temp = _lab_helper(normalized)
    fx, fy, fz = temp.unbind(-3)

    l = 1.16 * fy - 0.16
    a = fx.sub(fy).mul(5.0)
    b = fz.sub(fy).mul(-2.0)

    lab = torch.stack((l, a, b), dim=-3)
    if ret_matrix:
        return lab, matrix
    return lab


def lab_to_xyz(
    lab: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE LAB space to CIE XYZ space.

    Parameters
    ----------
    lab : torch.Tensor
        An image in CIE LAB space with shape `(*, 3, H, W)`.
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
        An image in CIE XYZ space with the shape `(*, 3, H, W)`.
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    l, a, b = lab.unbind(-3)

    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)
    max_ = matrix.sum(dim=1)
    max_ = align_device_type(max_, lab)
    max_ = to_channel_coeff(max_, 3)

    l = l.add(0.16).mul(1 / 1.16)
    temp_xyz = torch.stack(
        (
            l.add(a, alpha=0.2),
            l,
            l.sub(b, alpha=0.5),
        ),
        dim=-3,
    )

    xyz = _lab_helper_inv(temp_xyz)
    xyz = xyz * max_
    if ret_matrix:
        return xyz, matrix
    return xyz


def rgb_to_lab(
    rgb: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from RGB space to CIE LAB space.

    The input is assumed to be in the range of [0, 1]. If rgb_spec is a
    tensor, then the input rgb is assumed to be linear RGB.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape `(*, 3, H, W)`.
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
    lab : torch.Tensor
        An image in CIE LAB space with the shape `(*, 3, H, W)`.
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    xyz = rgb_to_xyz(rgb, rgb_spec, white, obs)
    lab, matrix = xyz_to_lab(xyz, rgb_spec, white, obs, ret_matrix=True)
    if ret_matrix:
        return lab, matrix
    return lab


def lab_to_rgb(
    lab: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE LAB space to RGB space.

    Parameters
    ----------
    lab : torch.Tensor
        An image in CIE LAB space with shape `(*, 3, H, W)`.
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
        An RGB image in [0, 1] with the shape `(*, 3, H, W)`.
    mat : torch.Tensor
        A transformation matrix used to convert CIE XYZ to RGB.
        `mat` is returned only if `ret_matrix` is true.
    """
    xyz = lab_to_xyz(lab, rgb_spec, white, obs)  # type: torch.Tensor
    rgb, matrix = xyz_to_rgb(xyz, rgb_spec, white, obs, ret_matrix=True)
    if ret_matrix:
        return rgb, matrix
    return rgb
