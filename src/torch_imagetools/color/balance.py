"""Color balance functions including chromatic adaptation transform, gray world
algorithm, etc...
"""
__all__ = [
    'get_von_kries_transform_matrix',
    'von_kries_transform',
    'balance_by_scaling',
    'gray_world_balance',
    'white_patch_balance',
]

from typing import overload

import torch
from torch_imagetools.utils.helpers import matrix_transform

from .lms import CATMethod, xyz_to_lms


def get_von_kries_transform_matrix(
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: CATMethod = 'bradford',
) -> torch.Tensor:
    """Returns a transformation matrix for von Kries adaptation, which
    converts colors from a illuminant to another illuminant.

    Parameters
    ----------
    xyz_white : torch.Tensor
        The source white point in CIE XYZ space.
    xyz_target_white : torch.Tensor
        The target white point in CIE XYZ space.
    method : CATMethod, optional
        Chromatic adaptation method, by default 'bradford'.

    Returns
    -------
    torch.Tensor
        Matrix with shape (3, 3).
    """
    xyz_white = xyz_white.reshape(3, 1, 1)
    xyz_target_white = xyz_target_white.view(3, 1, 1)
    lms_white, lms_matrix = xyz_to_lms(xyz_white, method, ret_matrix=True)
    lms_target_white = matrix_transform(xyz_target_white, lms_matrix)
    ratio = (lms_target_white / lms_white).view(3)

    # Chromatic apaptation transformation matrix
    cat_matrix = lms_matrix.inverse() @ torch.diag(ratio) @ lms_matrix
    return cat_matrix


@overload
def von_kries_transform(
    xyz: torch.Tensor,
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: CATMethod | torch.Tensor = 'bradford',
    *,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def von_kries_transform(
    xyz: torch.Tensor,
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: CATMethod | torch.Tensor = 'bradford',
    *,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def von_kries_transform(
    xyz: torch.Tensor,
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: CATMethod | torch.Tensor = 'bradford',
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Applies chromatic adaptation transformation to an image in CIE XYZ
    space with given source and target white points.

    If method is set to be 'xyz', the transformation matrix between XYZ and
    LMS is the identity matrix. Thus, the result is a wrong von Kries
    transformation and the

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    xyz_white : torch.Tensor
        The source white point in CIE XYZ space.
    xyz_target_white : torch.Tensor
        The target white point in CIE XYZ space.
    method : CATMethod, optional
        Chromatic adaptation method, by default 'bradford'. If method is a
        Tensor, then it will be regarded as the transformation matrix (
        XYZ -> LMS -> scaling LMS -> XYZ).
    ret_matrix : bool, optional
        If True, returns image and chromatic adaptation transformation matrix.
        By default False.

    Returns
    -------
    torch.Tensor
        An image in CIE XYZ space with the shape (*, 3, H, W). If ret_matrix
        is True, returns image and the transformation matrix.
    """
    if torch.is_tensor(method):
        matrix = method
    else:
        matrix = get_von_kries_transform_matrix(
            xyz_white, xyz_target_white, method
        )
    new_xyz = matrix_transform(xyz, matrix)
    if ret_matrix:
        return new_xyz, matrix
    return new_xyz


@overload
def balance_by_scaling(
    img: torch.Tensor,
    scaled_max: int | float | torch.Tensor,
    *,
    ret_factors: bool = False,
) -> torch.Tensor: ...
@overload
def balance_by_scaling(
    img: torch.Tensor,
    scaled_max: int | float | torch.Tensor,
    *,
    ret_factors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def balance_by_scaling(
    img: torch.Tensor,
    scaled_max: int | float | torch.Tensor,
    *,
    ret_factors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Wrong von Kries transform. Multiplies an image by
    coeff_channel = scaled_max / maximum_of_channel.

    Parameters
    ----------
    img : torch.Tensor
        Image in RGB space with shape (*, C, H, W).
    scaled_max : int | float | torch.Tensor
        The maximum(s) after scaling.
    ret_factors : bool, optional
        If True, returns image and scaling factors. By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An image with the shape (*, C, H, W). If ret_factors is True, returns
        image and the scaling factors with shape (C,).
    """
    num_ch = img.shape[-3]
    # Get max of each channel
    ndim = img.ndim
    reduced = list(range(ndim))
    reduced = reduced[:-3] + reduced[-2:]
    ch_max = img.amax(reduced)

    factors = (scaled_max / ch_max).reshape(num_ch, 1, 1)

    balanced = img * factors
    if ret_factors:
        return balanced, factors
    return balanced


@overload
def gray_world_balance(
    rgb: torch.Tensor,
    *,
    ret_factors: bool = False,
) -> torch.Tensor: ...
@overload
def gray_world_balance(
    rgb: torch.Tensor,
    *,
    ret_factors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def gray_world_balance(
    rgb: torch.Tensor,
    *,
    ret_factors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """White balance by the gray-world algorithm. Multiplies each channel by
    coeff_channel = mean / mean_of_channel.

    Parameters
    ----------
    rgb : torch.Tensor
        Image in RGB space with shape (*, 3, H, W).
    ret_factors : bool, optional
        If True, returns image and scaling factors. By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An image with the shape (*, 3, H, W). If ret_factors is True, returns
        image and the scaling factors with shape (3,).
    """
    num_ch = rgb.shape[-3]
    # Get mean values
    ndim = rgb.ndim
    reduced = list(range(ndim))
    reduced = reduced[:-3] + reduced[-2:]
    ch_mean = rgb.mean(reduced)
    img_mean = ch_mean.mean()

    factors = (img_mean / ch_mean).reshape(num_ch, 1, 1)

    balanced = (rgb * factors).clip_(0.0, 1.0)
    if ret_factors:
        return balanced, factors
    return balanced


@overload
def white_patch_balance(
    rgb: torch.Tensor,
    q: float | torch.Tensor = 1.0,
    *,
    ret_factors: bool = False,
) -> torch.Tensor: ...
@overload
def white_patch_balance(
    rgb: torch.Tensor,
    q: float | torch.Tensor = 1.0,
    *,
    ret_factors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def white_patch_balance(
    rgb: torch.Tensor,
    q: float | torch.Tensor = 1.0,
    *,
    ret_factors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """White balance by generalized white patch algorithm. Multiplies each
    channel of an RGB image by coeff_channel = 1 / qtile_of_channel.

    When q = 1.0, it is the standard white patch balance and equivalent to
    balance by scaling for maximum = 1.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB Image in range of [0, 1] with shape (*, 3, H, W).
        If ndim > 3, the quantile value is calculated across multiple images,
        and images will be scaled by same factors.
    q : float | None, optional
        Quantile. Scaling the quantile value to 1. If q is a Tensor with
        shape (3,), the values will be regarded as the quantile of each
        channel.
    ret_factors : bool, optional
        If True, returns image and scaling factors. By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An RGB Image in range of [0, 1] with shape (*, 3, H, W). If
        ret_factors is True, returns image and the scaling factors with
        shape (3,).
    """
    num_ch = rgb.shape[-3]
    if not torch.is_tensor(q) and q >= 1.0:
        ch_quantile = 1.0
    else:
        flatten = torch.flatten(rgb.movedim(-3, 0), 1)
        if torch.is_tensor(q) and q.shape[0] == 3:
            ch_quantile = [
                torch.quantile(flatten[i], q[i]) for i in range(num_ch)
            ]
            ch_quantile = torch.tensor(ch_quantile)
        else:  # scalar
            ch_quantile = torch.quantile(flatten, q, dim=1)

    factors = (1.0 / ch_quantile).reshape(num_ch, 1, 1)

    balanced = (rgb * factors).clip_(0.0, 1.0)
    if ret_factors:
        return balanced, factors
    return balanced
