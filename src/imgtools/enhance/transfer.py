__all__ = [
    'transfer_reinhard',
]

import torch

from ..color._ciexyz import get_rgb_to_xyz_matrix
from ..color._lms import get_xyz_to_lms_matrix
from ..color._rgb import gammaize_rgb, linearize_rgb
from ..utils.helpers import align_device_type
from ..utils.math import matrix_transform
from .equlization import match_mean_std

_MAT_FROM_LMS = torch.tensor(
    (
        (0.5773502691896258, 0.5773502691896258, 1.1547005383792517),
        (0.4082482904638631, 0.4082482904638631, -0.8164965809277261),
        (0.7071067811865475, -0.7071067811865475, 0.0),
    ),
    dtype=torch.float64,
)


def transfer_reinhard(
    src: torch.Tensor,
    tar: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    adap_method: str = 'bradford',
):
    """Color transfer by E. Reinhard's work [1]. Match the mean and standard
    deviation in the color space L-α-β (Si)

    Parameters
    ----------
    src : torch.Tensor
        Source image in RGB space with shape (*, C, H, W).
    tar : torch.Tensor
        Target image in RGB space with shape (*, C, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        White point. The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        The degree of oberver.
    method : CATMethod, default='bradford'
        Chromatic adaptation method.

    Returns
    -------
    torch.Tensor
        Transferred image in RGB space. Shape=max(src.shape, tar.dtype).

    References
    ----------
    [1] E. Reinhard, M. Adhikhmin, B. Gooch and P. Shirley, "Color
        transfer between images," in IEEE Computer Graphics and Applications,
        vol. 21, no. 5, pp. 34-41, July-Aug. 2001, doi: 10.1109/38.946629
    """
    tar = align_device_type(tar, src)

    mat_xyz = get_rgb_to_xyz_matrix(
        rgb_spec, white, obs, dtype=src.dtype, device=src.device
    )
    mat_lms = get_xyz_to_lms_matrix(
        adap_method, dtype=src.dtype, device=src.device
    )
    mat1 = mat_lms @ mat_xyz

    linear_src = linearize_rgb(src, rgb_spec)
    linear_tar = linearize_rgb(tar, rgb_spec)
    # To LMS space.
    src_lms = matrix_transform(linear_src, mat1)
    tar_lms = matrix_transform(linear_tar, mat1)
    # To L-alpha-beta space.
    max2 = align_device_type(_MAT_FROM_LMS, src)
    src_lab = matrix_transform(src_lms.log1p(), max2)
    tar_lab = matrix_transform(tar_lms.log1p(), max2)

    matched = match_mean_std(src_lab, tar_lab)
    # To LMS space.
    mat2i = align_device_type(_MAT_FROM_LMS.inverse(), src)
    res_lms = matrix_transform(matched, mat2i).expm1()
    # To RGB space.
    if src.dtype == torch.float16:
        mat1i = mat1.float().inverse().type(torch.float16)
    else:
        mat1i = mat1.inverse()
    res_linear_rgb = matrix_transform(res_lms, mat1i)

    res_linear_rgb = res_linear_rgb.clip(0.0, 1.0)
    res_rgb = gammaize_rgb(res_linear_rgb, rgb_spec)
    return res_rgb
