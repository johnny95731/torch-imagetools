"""Color conversion functions between RGB space and HWB space"""

__all__ = ['rgb_to_hwb', 'hwb_to_rgb']
import torch
import numpy as np

from ..utils.helpers import tensorlize
from .hsv import hsv_helper, hsv_to_rgb


def rgb_to_hwb(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a HWB image.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        An RGB image in the range of [0, 1]. For a ndarray, the shape should be
        (*, H, W, 3). For a Tensor, the shape should be (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        HWB image with shape (*, 3, H, W). The H channel values are in the
        range [0, 360), W and B are in the range [0, 1]
    """
    hue, amax, amin, _ = hsv_helper(rgb)
    whiteness = amin
    blackness = torch.sub(1, amax, out=amax)

    hwb = torch.stack((hue, whiteness, blackness), dim=-3)
    return hwb


def hwb_to_rgb(hwb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver a HWB image to an RGB image.

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        An RGB image in the range of [0, 1]. For a ndarray, the shape should be
        (*, H, W, 3). For a Tensor, the shape should be (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        HWB image with shape (*, 3, H, W). The channels are hue, whiteness,
        blackness.
    """
    hwb = tensorlize(hwb)

    h: torch.Tensor = hwb[..., 0, :, :]
    w: torch.Tensor = hwb[..., 1, :, :]
    b: torch.Tensor = hwb[..., 2, :, :]

    total = w + b
    exceed = total > 1.0

    bri = 1.0 - b
    bri[exceed] = w[exceed] / total[exceed]
    sat = 1.0 - w / bri
    sat[exceed] = 0
    torch.nan_to_num(sat, 0.0, 0.0, 0.0, out=sat)

    hsv = torch.stack((h, sat, bri), dim=-3)
    rgb = hsv_to_rgb(hsv)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 20

    hsl = rgb_to_hwb(img)
    ret = hwb_to_rgb(hsl)

    d = torch.abs(ret - img)
    print(torch.max(d))

    print(timeit('rgb_to_hwb(img)', number=num, globals=locals()))
    print(timeit('hwb_to_rgb(hsl)', number=num, globals=locals()))
