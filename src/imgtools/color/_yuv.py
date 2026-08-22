__all__ = [
    'rgb_to_yuv',
    'yuv_to_rgb',
]

from typing import Literal

import torch

from ..core.math import matrix_transform
from ..utils.helpers import __default_dtype


def rgb_to_yuv(
    rgb: torch.Tensor,
    standard: Literal['bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'] = 'bt.601',
) -> torch.Tensor:
    """Converts an image from RGB space to YUV space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB imag in the range of [0, 1] with shape `(*, 3, H, W)`.
    standard : {'bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'}, default='bt.601'
        The specification. The chrominance channels are normalized to the
        range [-0.5, 0.5].

    Returns
    -------
    torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`. The range of Y is [0, 1]
        and the range of U and V are [-0.5, 0.5].
    """
    # fmt: off
    dtype = __default_dtype(rgb)
    device = rgb.device
    # All chrominance channel are normalized to the range [-0.5, 0.5].
    if standard == 'bt.601':
        matrix = torch.tensor(
            [[0.299,  0.587,  0.114],
            [-0.169, -0.331,  0.500],
            [ 0.500, -0.419, -0.081]],
            dtype=dtype,
            device=device
        )
    elif standard == 'bt.709':
        matrix = torch.tensor(
            [[0.2126,  0.7152,  0.0722],
            [-0.1146, -0.3854,  0.5000],
            [ 0.5000, -0.4542, -0.0458]],
            dtype=dtype,
            device=device
        )
    elif standard == 'bt.2020':
        matrix = torch.tensor(
            [[0.2627,  0.6780,  0.0593],
            [-0.1396, -0.3604,  0.5000],
            [ 0.5000, -0.4598, -0.0402]],
            dtype=dtype,
            device=device
        )
    elif standard == 'yiq':
        matrix = torch.tensor(
            [[0.30,  0.59,   0.11],
            [ 0.5000, -0.2315, -0.2685],
            [ 0.2028, -0.5000, 0.2972]],
            dtype=dtype,
            device=device
        )
    elif standard == 'ycocg':
        matrix = torch.tensor(
            [[0.25, 0.5,  0.25],
            [ 0.50, 0.0, -0.50],
            [-0.25, 0.5, -0.25]],
            dtype=dtype,
            device=device
        )
    else:
        raise ValueError(f'Invalid value of argument `standard`: {standard}')
    # fmt: on
    yuv = matrix_transform(rgb, matrix)
    return yuv


def yuv_to_rgb(
    yuv: torch.Tensor,
    standard: Literal['bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'] = 'bt.601',
) -> torch.Tensor:
    """Converts an image from YUV space to RGB space.

    The input is assumed to be in the range of [0, 1] (for Y channel) and
    [-0.5, 0.5] (for U and V channels). The output will be clip to [0, 1].

    Parameters
    ----------
    yuv : torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`.
    standard : {'bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'}, default='bt.601'
        The specification. The chrominance channels are normalized to the
        range [-0.5, 0.5].

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape `(*, 3, H, W)`.
    """
    dtype = __default_dtype(yuv)
    device = yuv.device
    # fmt: off
    if standard == 'bt.601':
        inv_matrix = torch.tensor(
            [[1.0,  0.0000,  1.4020],
            [ 1.0, -0.3441, -0.7141],
            [ 1.0,  1.7720,  0.0000]],
            dtype=dtype,
            device=device
        )
    elif standard == 'bt.709':
        inv_matrix = torch.tensor(
            [[1.0,  0.0000,  1.5748],
            [ 1.0, -0.1873, -0.4681],
            [ 1.0,  1.8556,  0.0000]],
            dtype=dtype,
            device=yuv.device
        )
    elif standard == 'bt.2020':
        inv_matrix = torch.tensor(
            [[1.0,  0.0000,  1.4746],
            [ 1.0, -0.1646, -0.5714],
            [ 1.0,  1.8814,  0.0000]],
            dtype=dtype,
            device=device
        )
    elif standard == 'yiq':
        inv_matrix = torch.tensor(
            [[1.0,  1.1344,  0.6548],
            [ 1.0, -0.3292, -0.6676],
            [ 1.0, -1.3280,  1.7949]],
            dtype=dtype,
            device=device
        )
    elif standard == 'ycocg':
        inv_matrix = torch.tensor(
            [[1.0,  1.0, -1.0],
            [ 1.0,  0.0,  1.0],
            [ 1.0, -1.0, -1.0]],
            dtype=dtype,
            device=device
        )
    else:
        raise ValueError(f'Invalid value of argument `standard`: {standard}')
    # fmt: on
    rgb = matrix_transform(yuv, inv_matrix).clip(0.0, 1.0)
    return rgb
