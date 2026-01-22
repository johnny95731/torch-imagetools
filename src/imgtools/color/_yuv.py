__all__ = [
    'rgb_to_yuv',
    'yuv_to_rgb',
]

import torch

from ..utils.math import matrix_transform


def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to YUV space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB imag in the range of [0, 1] with shape `(*, 3, H, W)`.

    Returns
    -------
    torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`. The range of Y is [0, 1]
        and the range of U and V are [-0.5, 0.5].
    """
    # fmt: off
    dtype = rgb.dtype if torch.is_floating_point(rgb) else torch.float32
    matrix = torch.tensor(
        [[ 0.299,  0.587,  0.114],
         [-0.169, -0.331,  0.500],
         [ 0.500, -0.419, -0.081]],
        dtype=dtype,
        device=rgb.device
    )
    # fmt: on
    yuv = matrix_transform(rgb, matrix)
    return yuv


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    """Converts an image from YUV space to RGB space.

    The input is assumed to be in the range of [0, 1] (for Y channel) and
    [-0.5, 0.5] (for U and V channels). The output will be clip to [0, 1].

    Parameters
    ----------
    yuv : torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`.

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape `(*, 3, H, W)`.
    """
    dtype = yuv.dtype if torch.is_floating_point(yuv) else torch.float32
    # fmt: off
    matrix = torch.tensor(
        [[ 1.0, -0.00093, 1.401687],
         [ 1.0, -0.3437, -0.71417],
         [ 1.0,  1.77216, 0.00099]],
        dtype=dtype,
        device=yuv.device
    )
    # fmt: on
    rgb = matrix_transform(yuv, matrix).clip(0.0, 1.0)
    return rgb
