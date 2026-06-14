"""Color balance functions including chromatic adaptation transform, gray world
algorithm, etc...
"""

__all__ = [
    'get_von_kries_transform_matrix',
    'von_kries_transform',
    'gray_world_balance',
    'gray_edge_balance',
    'white_patch_balance',
    'cheng_pca_balance',
    'simplest_color_balance',
]

import torch

from ..color import gammaize_rgb, rgb_to_xyz, xyz_to_lms
from ..core.math import matrix_transform
from ..utils.helpers import (
    _to_channel_coeff,
    align_device_type,
    check_valid_image_ndim,
)
from .est_illuminant import estimate_illuminant_cheng


def get_von_kries_transform_matrix(
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: str = 'bradford',
) -> torch.Tensor:
    """Returns a transformation matrix for von Kries adaptation, which
    converts colors from a illuminant to another illuminant.

    Parameters
    ----------
    xyz_white : torch.Tensor
        The source white point in CIE XYZ space. Shape `(*, 3)`.
    xyz_target_white : torch.Tensor
        The target white point in CIE XYZ space. Shape `(*, 3)`.
    method : CATMethod, default='bradford'
        Chromatic adaptation method.

    Returns
    -------
    torch.Tensor
        Matrix with shape=`(*, 3, 3)`. Same dtype and device as `xyz_white`.

    Examples
    --------

    >>> from imgtools.balance import get_von_kries_transform_matrix
    >>> from imgtools.color import get_rgb_to_xyz_matrix, rgb_to_xyz, xyz_to_rgb
    >>> from imgtools.utils import matrix_transform
    >>>
    >>> rgb = torch.tensor((0.75, 0.1, 0.23)).reshape(3, 1, 1)
    >>> xyz, mat = rgb_to_xyz(rgb, 'srgb', 'D65', ret_matrix=True)
    >>> white_d65 = mat.sum(1)
    >>> white_d50 = get_rgb_to_xyz_matrix('srgb', 'D50').sum(1)
    >>>
    >>> mat_adap = get_von_kries_transform_matrix(white_d65, white_d50)
    >>> new_xyz = matrix_transform(xyz, mat_adap)
    >>> # Equivalent to: new_xyz = von_kries_transform(xyz, white_d65, white_d50)
    >>> new_rgb = xyz_to_rgb(xyz, 'srgb', 'D50')  # tensor([0.6935, 0.1019, 0.2713])
    """
    xyz_target_white = align_device_type(xyz_target_white, xyz_white)

    xyz_white = xyz_white.view(-1, 3, 1, 1)
    xyz_target_white = xyz_target_white.view(-1, 3, 1, 1)
    lms_white, lms_matrix = xyz_to_lms(xyz_white, method, ret_matrix=True)
    lms_target_white = matrix_transform(xyz_target_white, lms_matrix)
    lms_target_white = align_device_type(lms_target_white, lms_white)
    ratio = (lms_target_white / lms_white).view(-1, 3, 1)

    # Chromatic apaptation transformation matrix
    lms_matrix = align_device_type(lms_matrix, ratio)
    cat_matrix = lms_matrix.inverse() @ (ratio * lms_matrix)
    cat_matrix.squeeze_(0)
    return cat_matrix


def von_kries_transform(
    xyz: torch.Tensor,
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: str = 'bradford',
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
        An image in CIE XYZ space with shape `(*, 3, H, W)`.
    xyz_white : torch.Tensor
        The source white point in CIE XYZ space. A tensor with numel = 3.
    xyz_target_white : torch.Tensor
        The target white point in CIE XYZ space. A tensor with numel = 3.
    method : CATMethod, default='bradford'
        Chromatic adaptation method. If method is a
        Tensor, then it will be regarded as the transformation matrix (
        XYZ -> LMS -> scaling LMS -> XYZ).
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    new_xyz : torch.Tensor
        An image in CIE XYZ space with the shape `(*, 3, H, W)`.
    mat : torch.Tensor
        A chromatic adaptation matrix.
        `mat` is returned only if `ret_matrix` is true.

    Examples
    --------

    >>> from imgtools.balance import von_kries_transform
    >>> from imgtools.color import get_rgb_to_xyz_matrix, rgb_to_xyz, xyz_to_rgb
    >>> from imgtools.utils import matrix_transform
    >>>
    >>> rgb = torch.tensor((0.75, 0.1, 0.23)).reshape(3, 1, 1)
    >>> xyz, mat = rgb_to_xyz(rgb, 'srgb', 'D65', ret_matrix=True)
    >>> white_d65 = mat.sum(1)
    >>> white_d50 = get_rgb_to_xyz_matrix('srgb', 'D50').sum(1)
    >>>
    >>> new_xyz = von_kries_transform(xyz, white_d65, white_d50)
    >>> # Equivalent to:
    >>> # mat_adap = get_von_kries_transform_matrix(white_d65, white_d50)
    >>> # new_xyz = matrix_transform(xyz, mat_adap)
    >>> new_rgb = xyz_to_rgb(xyz, 'srgb', 'D50')  # tensor([0.6935, 0.1019, 0.2713])
    """
    mat = get_von_kries_transform_matrix(xyz_white, xyz_target_white, method)
    new_xyz = matrix_transform(xyz, mat)
    if ret_matrix:
        return new_xyz, mat
    return new_xyz


def gray_world_balance(
    rgb: torch.Tensor,
    ret_illum: bool = False,
) -> torch.Tensor:
    """White balance by the gray-world algorithm. Multiplies each channel by

    `coeff_channel = mean / mean_of_channel`.

    Parameters
    ----------
    rgb : torch.Tensor
        An Image in RGB space with shape `(*, C, H, W)`.
    ret_illum : bool, default=False
        Return the estimated illuminant color instead of returning the
        balanced image.

    Returns
    -------
    balanced : torch.Tensor
        A balanced image with the shape `(*, C, H, W)` when `ret_factors` is
        `False`.
    illuminant : torch.Tensor
        Estimated illuminant color with shape `(*, C, 1, 1)` when
        `ret_factors` is `True`.

    Examples
    --------

    >>> from imgtools.balance import gray_world_balance
    >>>
    >>> rgb = torch.rand((3, 512, 512))
    >>> balanced = gray_world_balance(rgb)
    >>>
    >>> illum = gray_world_balance(rgb, ret_factors=True)
    >>> balanced2 = rgb * (illum.mean() / illum)
    >>> # or
    >>> # balanced2 = rgb * (illum[1] / illum)
    """
    check_valid_image_ndim(rgb)
    ch_mean = rgb.mean((-1, -2), keepdim=True)
    if ret_illum:
        return ch_mean
    factors = ch_mean.mean() / ch_mean
    balanced = (rgb * factors).clip_(0.0, 1.0)
    return balanced


def gray_edge_balance(
    rgb: torch.Tensor,
    edge: torch.Tensor,
    ret_illum: bool = False,
) -> torch.Tensor:
    """White balance by the gray-edge algorithm. Multiplies each channel by

    `coeff_channel = mean_of_gradient / mean_of_gradient_of_channel`.

    Parameters
    ----------
    rgb : torch.Tensor
        Image in RGB space with shape `(*, C, H, W)`.
    edge : torch.Tensor
        The edge of the image with shape `(*, C, H, W)`.
    ret_illum : bool, default=False
        Return the estimated illuminant color instead of returning the
        balanced image.

    Returns
    -------
    balanced : torch.Tensor
        A balanced image with the shape `(*, C, H, W)` when `ret_factors` is
        `False`.
    illuminant : torch.Tensor
        Estimated illuminant color with shape `(*, C, 1, 1)` when
        `ret_factors` is `True`.

    Examples
    --------

    >>> from imgtools.balance import gray_edge_balance
    >>> from imgtools.filter import laplacian
    >>>
    >>> rgb = torch.rand((3, 512, 512))
    >>> edge = laplacian(rgb)
    >>> balanced = gray_edge_balance(rgb, edge)
    >>>
    >>> illum = gray_edge_balance(rgb, edge, ret_factors=True)
    >>> balanced2 = rgb * (illum.mean() / illum)
    >>> # or
    >>> # balanced2 = rgb * (illum[1] / illum)
    """
    check_valid_image_ndim(rgb)
    check_valid_image_ndim(edge)
    edge = edge.abs()
    ch_grad_mean = edge.mean((-1, -2), keepdim=True)
    if ret_illum:
        return ch_grad_mean
    factors = ch_grad_mean.mean() / ch_grad_mean
    factors = align_device_type(factors, rgb)
    balanced = (rgb * factors).clip_(0.0, 1.0)
    return balanced


def white_patch_balance(
    rgb: torch.Tensor,
    q: int | float | torch.Tensor = 1.0,
    ret_illum: bool = False,
) -> torch.Tensor:
    """White balance by generalized white patch algorithm. Multiplies each
    channel of an RGB image by

    `coeff_channel = q_quantile_of_image / q_quantile_of_channel`.

    When q = 1.0, it is the standard white patch balance and equivalent to
    balance by scaling for maximum = 1.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB Image in range of [0, 1] with shape `(*, C, H, W)`.
        If ndim > 3, the quantile value is calculated across images,
        and images will be scaled by same factors.
    q : int | float | torch.Tensor, default=1.0
        q-quantile. The values will be cliped to [0, 1].
        - A single number: the quantile for all channels.
        - Tensor with shape `(3,)`: the quantiles of channels.
    ret_illum : bool, default=False
        Return the estimated illuminant color instead of returning the
        balanced image.

    Returns
    -------
    balanced : torch.Tensor
        A balanced image with the shape `(*, C, H, W)` when `ret_factors` is
        `False`.
    illuminant : torch.Tensor
        Estimated illuminant color with shape `(*, C, 1, 1)` when
        `ret_factors` is `True`.

    Examples
    --------

    >>> from imgtools.balance import white_patch_balance
    >>>
    >>> rgb = torch.rand((3, 512, 512))
    >>> balanced = white_patch_balance(rgb, 0.9)
    >>>
    >>> illum = gray_edge_balance(rgb, edge, ret_factors=True)
    >>> balanced2 = rgb * (illum.mean() / illum)
    >>> # or
    >>> # balanced2 = rgb * (illum[1] / illum)
    """
    is_not_batch = check_valid_image_ndim(rgb)
    if is_not_batch:
        rgb = rgb.unsqueeze(0)
    flatten = torch.flatten(rgb, -2)
    flatten = flatten.sort()[0].contiguous()

    shape = rgb.shape[:-2]
    length = flatten.size(-1) - 1
    if isinstance(q, float):
        q = torch.full(
            shape, int(round(q * length)), dtype=torch.int64, device=rgb.device
        )
    elif isinstance(q, int):
        q = torch.full(
            shape, int(round(q * length)), dtype=torch.int64, device=rgb.device
        )
    else:
        q = (q * length).round_().long().to(rgb.device)
    q = q.clip_(0, length - 1).broadcast_to(*shape, 1)

    ch_quantile = flatten.gather(2, q).unsqueeze_(-1)
    if ret_illum:
        return ch_quantile

    factors = ch_quantile[:, 1:2] / ch_quantile
    balanced = (rgb * factors).clip_(0.0, 1.0)
    if is_not_batch:
        balanced = balanced.squeeze(0)
    return balanced


def cheng_pca_balance(
    rgb: torch.Tensor,
    n_selected: float = 3.5,
    adaptation: str = 'von kries',
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
) -> torch.Tensor:
    """White balance by Cheng's PCA method [1]. Estimate the illuminant and
    applies chromatic adaptation transformation.

    If you want to estimate the illuminant, call
    `balance.estimate_illuminant_cheng`.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape `(*, C, H, W)`.
    n_selected : float, default=3.5
        A percentage value for picking pixels to estimate illuminant.
    adaptation : Literal['rgb', 'von kries'], default='von kries'
        Chromatic adaptation method. RGB scaling or von Kries transformation.
        - 'RGB': Scaling the illuminant to 1.
        - 'von kries': von Kries transformation.
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
        Only works for `adaptation='von kries'`.
    white : StandardIlluminants, default='D65'
        White point. The input is case-insensitive. Only works for
        `adaptation='von kries'`.
    obs : {2, '2', 10, '10'}, default=10
        The degree of oberver. Only works for `adaptation='von kries'`.

    Returns
    -------
    torch.Tensor
        A balanced image with shape `(*, C, H, W)`.

    Raises
    ------
    ValueError
        When `adaptation` is not in ('rgb', 'von kries')

    References
    ----------
    [1] Cheng, Dongliang, Dilip K. Prasad, and Michael S. Brown. "Illuminant
        estimation for color constancy: why spatial-domain methods work and
        the role of the color distribution." JOSA A 31.5 (2014): 1049-1058.

    Examples
    --------

    >>> from imgtools.balance import cheng_pca_balance
    >>>
    >>> rgb = torch.rand((3, 512, 512))
    >>> balanced = cheng_pca_balance(rgb)
    """
    adaptation = adaptation.lower()
    if adaptation not in ('rgb', 'von kries'):
        raise ValueError(
            f"`adaptation` should be 'rgb' or 'von kries', but got {adaptation}."
        )

    illuminant = estimate_illuminant_cheng(rgb, n_selected)
    illuminant = _to_channel_coeff(illuminant, 3)
    if adaptation == 'rgb':
        coeff = illuminant.mean(-3, keepdim=True) / illuminant
        balanced = (coeff * rgb).clip(0.0, 1.0)
    else:  # 'von kries'
        # Do rgb->xyz under different white point makes result more close to
        # the chosen white point.
        # When both conversion under the same white point, the result always
        # looks similar.
        xyz, xyz_mat = rgb_to_xyz(rgb, rgb_spec, white, obs, ret_matrix=True)
        white_est = rgb_to_xyz(illuminant, rgb_spec, 'D65', obs)

        white_est = white_est / white_est[1]  # normalize
        white_std = xyz_mat.sum(1)
        balanced_xyz = von_kries_transform(xyz, white_est, white_std)  # type: torch.Tensor

        balanced = matrix_transform(balanced_xyz, xyz_mat.inverse())
        balanced = gammaize_rgb(balanced, rgb_spec).clip(0.0, 1.0)
    return balanced


def simplest_color_balance(
    img: torch.Tensor,
    dark_percent: float = 0.0,
    light_percent: float = 0.0,
):
    """Clip top-k1 and bottom-k2 percentage values and normalize to `[0, 1]`.
    The algorithm is proposed by Limare et al [1].

    This function will not estimate the illuminant.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    dark_percent : float, default=0.0
        The percentage value for clipping the lowest `dark_percent`% values.
    light_percent : float, default=0.0
        The percentage value for clipping the highest `light_percent`% values.

    Returns
    -------
    torch.Tensor
        A balanced image with shape `(*, C, H, W)`.

    References
    ----------
    [1] Nicolas Limare, Jose-Luis Lisani, Jean-Michel Morel, Ana Belén Petro, and Catalina Sbert, Simplest Color Balance, Image Processing On Line, 1 (2011), pp. 297–315. https://doi.org/10.5201/ipol.2011.llmps-scb
    """
    flatted = img.flatten(-2)
    num_pixel = flatted.shape[-1]
    thresh_dark = int(dark_percent * num_pixel)
    thresh_light = num_pixel - 1 - int(light_percent * num_pixel)
    #
    sorted = flatted.sort().values
    mini = sorted[..., thresh_dark].unsqueeze_(-1).unsqueeze_(-1)
    maxi = sorted[..., thresh_light].unsqueeze_(-1).unsqueeze_(-1)
    balanced = (img - mini) / (maxi - mini)
    balanced = balanced.clip(0.0, 1.0)
    return balanced
