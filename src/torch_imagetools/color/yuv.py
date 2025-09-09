import torch
import numpy as np

from ..utils.helpers import matrix_transform, tensorlize


def rgb_to_yuv(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a YUV image.

    The input is assumed to be in the range of [0, 1]. The output will be
    normalized to [0, 1] (for Y channel) and [-0.5, 0.5] (for U and V channels)

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        An RGB image in the range of [0, 1]. For a ndarray, the shape should
        be (*, H, W, 3). For a Tensor, the shape should be (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        YUV image with shape (3, H, W) or (N, 3, H, W). The range of Y
        is [0, 1] and the range of U and V are [-0.5, 0.5].
    """
    rgb = tensorlize(rgb)

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
    yuv = matrix_transform(rgb, matrix.T)
    return yuv


def rgb_to_yuv2(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a YUV image.

    The input is assumed to be in the range of [0, 1]. The output will be
    normalized to [0, 1] (for Y channel) and [-0.5, 0.5] (for U and V channels)

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        An RGB image in the range of [0, 1]. For a ndarray, the shape should
        be (*, H, W, 3). For a Tensor, the shape should be (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        A YUV image with shape (3, H, W) or (N, 3, H, W). The range of Y
        is [0, 1] and the range of U and V are [-0.5, 0.5].
    """
    rgb = tensorlize(rgb)

    r, g, b = torch.unbind(rgb, dim=-3)

    y = (0.299 * r).add_(g, alpha=0.587).add_(g, alpha=0.114)
    cb = torch.sub(b, y).mul_(0.564)
    cr = torch.sub(r, y).mul_(0.713)
    # y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    # cb: torch.Tensor = -0.169 * r + -0.331 * g + 0.500 * b + 0.5
    # cr: torch.Tensor = 0.500 * r + -0.419 * g + -0.081 * b + 0.5

    return torch.stack((y, cb, cr), dim=-3)


def yuv_to_rgb(yuv: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver a YUV image to an RGB image.

    The input is assumed to be in the range of [0, 1] (for Y channel) and
    [-0.5, 0.5] (for U and V channels). The output will be clip to [0, 1].

    Parameters
    ----------
    yuv : np.ndarray | torch.Tensor
        A YUV image. For a ndarray, the shape should be (*, H, W, 3). For a
        Tensor, the shape should be (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An RGB image with shape (*, 3, H, W). The range of channels will be
        normalized to [0, 1].
    """
    yuv = tensorlize(yuv)

    # fmt: off
    dtype = yuv.dtype if torch.is_floating_point(yuv) else torch.float32
    matrix = torch.tensor(
        [[ 1.0, -0.00093, 1.401687],
         [ 1.0, -0.3437, -0.71417],
         [ 1.0,  1.77216, 0.00099]],
        dtype=dtype,
        device=yuv.device
    )
    # fmt: on
    rgb = matrix_transform(yuv, matrix.T).clip_(0.0, 1.0)
    return rgb
