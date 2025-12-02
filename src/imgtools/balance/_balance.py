"""Color balance functions including chromatic adaptation transform, gray world
algorithm, etc...
"""

__all__ = [
    'get_von_kries_transform_matrix',
    'von_kries_transform',
    'balance_by_scaling',
    'gray_world_balance',
    'gray_edge_balance',
    'white_patch_balance',
    'linear_regression_balance',
    'cheng_pca_balance',
]

import torch

from ..color import rgb_to_xyz, xyz_to_rgb
from ..utils.helpers import align_device_type, to_channel_coeff
from ..utils.math import matrix_transform
from ._lms import xyz_to_lms
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
        The source white point in CIE XYZ space.
    xyz_target_white : torch.Tensor
        The target white point in CIE XYZ space.
    method : CATMethod, default='bradford'
        Chromatic adaptation method.

    Returns
    -------
    torch.Tensor
        Matrix with shape (3, 3).

    Examples
    --------

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
    xyz_white = xyz_white.reshape(3, 1, 1)
    xyz_target_white = xyz_target_white.view(3, 1, 1)
    lms_white, lms_matrix = xyz_to_lms(xyz_white, method, ret_matrix=True)
    lms_target_white = matrix_transform(xyz_target_white, lms_matrix)
    ratio = (lms_target_white / lms_white).view(3)

    # Chromatic apaptation transformation matrix
    cat_matrix = lms_matrix.inverse() @ torch.diag(ratio) @ lms_matrix
    return cat_matrix


def von_kries_transform(
    xyz: torch.Tensor,
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: str | torch.Tensor = 'bradford',
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
        An image in CIE XYZ space with the shape (*, 3, H, W).
    mat : torch.Tensor
        A chromatic adaptation matrix.
        `mat` is returned only if `ret_matrix` is true.

    Examples
    --------

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
    if torch.is_tensor(method):
        mat = method
    else:
        mat = get_von_kries_transform_matrix(
            xyz_white, xyz_target_white, method
        )
    new_xyz = matrix_transform(xyz, mat)
    if ret_matrix:
        return new_xyz, mat
    return new_xyz


def balance_by_scaling(
    img: torch.Tensor,
    scaled_max: int | float | torch.Tensor,
    ret_factors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Wrong von Kries transform. Multiplies an image by
    > `coeff_channel = scaled_max / maximum_of_channel`.

    Parameters
    ----------
    img : torch.Tensor
        Image in RGB space with shape (*, C, H, W).
    scaled_max : int | float | torch.Tensor
        The maximum(s) after scaling.
    ret_factors : bool, default=False
        If true, returns image and scaling factors.

    Returns
    -------
    balanced : torch.Tensor
        An image with the shape (*, C, H, W).
    factors : torch.Tensor
        Scaling factors with shape (C,).
        `factors` is returned only if `ret_factors` is true.
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


def gray_world_balance(
    rgb: torch.Tensor,
    ret_factors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """White balance by the gray-world algorithm. Multiplies each channel by
    coeff_channel = mean / mean_of_channel.

    Parameters
    ----------
    rgb : torch.Tensor
        An Image in RGB space with shape (*, 3, H, W).
    ret_factors : bool, default=False
        If true, returns image and scaling factors.

    Returns
    -------
    balanced : torch.Tensor
        An image with the shape (*, C, H, W).
    factors : torch.Tensor
        Scaling factors with shape (C,).
        `factors` is returned only if `ret_factors` is true.
    """
    num_ch = rgb.shape[-3]
    # Get mean values
    ndim = rgb.ndim
    reduced = list(range(ndim))
    reduced = reduced[:-3] + reduced[-2:]
    ch_mean = rgb.mean(reduced)
    img_mean = ch_mean.mean()

    factors = (img_mean / ch_mean).reshape(num_ch, 1, 1)

    balanced = (rgb * factors).clip(0.0, 1.0)
    if ret_factors:
        return balanced, factors
    return balanced


def gray_edge_balance(
    rgb: torch.Tensor,
    edge: torch.Tensor,
    ret_factors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """White balance by the gray-edge algorithm. Multiplies each channel by
    coeff_channel = mean_of_gradient / mean_of_gradient_of_channel.

    Parameters
    ----------
    rgb : torch.Tensor
        Image in RGB space with shape (*, 3, H, W).
    edge : torch.Tensor
        The edge of the image with shape (*, 3, H, W).
    ret_factors : bool, default=False
        If true, returns image and scaling factors.

    Returns
    -------
    balanced : torch.Tensor
        An image with the shape (*, C, H, W).
    factors : torch.Tensor
        Scaling factors with shape (C,).
        `factors` is returned only if `ret_factors` is true.
    """
    num_ch = rgb.shape[-3]
    # Get mean values of gradients
    ndim = edge.ndim
    reduced = list(range(ndim))
    reduced = reduced[:-3] + reduced[-2:]
    ch_grad_mean = edge.mean(reduced)
    img_grad_mean = ch_grad_mean.mean()

    factors = (img_grad_mean / ch_grad_mean).reshape(num_ch, 1, 1)

    balanced = (rgb * factors).clip(0.0, 1.0)
    if ret_factors:
        return balanced, factors
    return balanced


def white_patch_balance(
    rgb: torch.Tensor,
    q: int | float | torch.Tensor = 1.0,
    ret_factors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """White balance by generalized white patch algorithm. Multiplies each
    channel of an RGB image by
        coeff_channel = q_quantile_of_image / q_quantile_of_channel.

    When q = 1.0, it is the standard white patch balance and equivalent to
    balance by scaling for maximum = 1.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB Image in range of [0, 1] with shape (*, 3, H, W).
        If ndim > 3, the quantile value is calculated across multiple images,
        and images will be scaled by same factors.
    q : int | float | torch.Tensor, default=1.0
        q-quantile. If q is a Tensor with shape (3,), the values will be
        regarded as the quantile of each channel. The values will be cliped to
        [0, 1].
    ret_factors : bool, default=False
        If false, only the image is returned.
        If true, also return the scaling factors.

    Returns
    -------
    balanced : torch.Tensor
        An image with the shape (*, C, H, W).
    factors : torch.Tensor
        Scaling factors with shape (C,).
        `factors` is returned only if `ret_factors` is true.
    """
    if not torch.is_floating_point(rgb):
        raise ValueError
    flatten = torch.flatten(rgb.movedim(-3, 0), 1)
    flatten = flatten.sort()[0].contiguous()

    num_ch = rgb.size(-3)
    length = flatten.size(1) - 1

    if isinstance(q, float):
        q = torch.full((num_ch,), q)
    elif isinstance(q, int):
        q = torch.full((num_ch,), q)
    if q.numel() == 1:
        q = q.repeat(num_ch)
    q.clip(0.0, 1.0)

    q = align_device_type(q, flatten)
    ch_quantile_ = []  # type: list[torch.Tensor]
    for i, _q in enumerate(q):
        _q = int(round(_q.item() * length))
        ch_quantile_.append(flatten[i, _q])
    ch_quantile = torch.stack(ch_quantile_)
    img_quantile = ch_quantile.quantile(q)  # approximation

    factors = (img_quantile / ch_quantile).reshape(num_ch, 1, 1)

    balanced = (rgb * factors).clip(0.0, 1.0)
    if ret_factors:
        return balanced, factors
    return balanced


def linear_regression_balance(
    rgb: torch.Tensor,
) -> torch.Tensor:
    """White balance by the linear regression. Estimation the coefficient by the
    red and green channel and predict the blue channel.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape (*, C, H, W). The model will evaluate the
        coefficients over all images in the batch.

    Returns
    -------
    torch.Tensor
        A balanced image with shape (*, C, H, W).
    """
    flattened = rgb.movedim(-3, 0).flatten(1)
    r, g, b = flattened.unbind()

    ones = torch.ones_like(r)
    x = torch.stack([ones, r, g], dim=0)  # n x 3
    beta = (x @ x.T).inverse() @ x @ b

    balanced_b = (beta[2] * g).add(r, alpha=beta[1]).add(beta[0])
    balanced_b.clip(0.0, 1.0)
    balanced = torch.stack([r, g, balanced_b], dim=0).reshape_as(rgb)
    return balanced


def cheng_pca_balance(
    rgb: torch.Tensor,
    adaptation: str = 'von kries',
) -> torch.Tensor:
    """White balance by Cheng's PCA method [1]. Estimate the illuminant and
    applies chromatic adaptation transformation.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, C, H, W).
    adaptation : Literal['rgb', 'von kries'], default='von kries'
        Chromatic adaptation method. RGB scaling or von Kries transformation.

    Returns
    -------
    torch.Tensor
        A balanced image with shape (*, C, H, W).

    Raises
    ------
    ValueError
        When `adaptation` is not in ('rgb', 'von kries')

    References
    ----------
    [1] Cheng, Dongliang, Dilip K. Prasad, and Michael S. Brown. "Illuminant
        estimation for color constancy: why spatial-domain methods work and
        the role of the color distribution." JOSA A 31.5 (2014): 1049-1058.
    """
    adaptation = adaptation.lower()
    if adaptation not in ('rgb', 'von kries'):
        raise ValueError(
            f"`adaptation` should be 'rgb' or 'von kries', but got {adaptation}."
        )

    illuminant = estimate_illuminant_cheng(rgb)
    illuminant = to_channel_coeff(illuminant, 3)

    if adaptation == 'rgb':
        coeff = 1.0 / illuminant
        balanced = (coeff * rgb).clip(0.0, 1.0)
    elif adaptation == 'von kries':
        xyz, xyz_mat = rgb_to_xyz(rgb, 'widegamut', ret_matrix=True)

        white_img = matrix_transform(illuminant, xyz_mat)
        white_xyz = xyz_mat.sum(dim=1)
        balanced_xyz = von_kries_transform(xyz, white_img, white_xyz)  # type: torch.Tensor

        balanced = xyz_to_rgb(balanced_xyz, 'widegamut')
    return balanced
