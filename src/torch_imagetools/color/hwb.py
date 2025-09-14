"""Color conversion functions between RGB space and HWB space"""

__all__ = ['rgb_to_hwb', 'hwb_to_rgb']
import torch

from .hsv import hsv_helper, hsv_to_rgb


def rgb_to_hwb(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to HWB space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An image in HWB space with shape (*, 3, H, W). The H channel values
        are in the range [0, 360), W and B are in the range of [0, 1].
    """
    hue, amax, amin, _ = hsv_helper(rgb)
    whiteness = amin
    blackness = torch.sub(1, amax, out=amax)

    hwb = torch.stack((hue, whiteness, blackness), dim=-3)
    return hwb


def hwb_to_rgb(hwb: torch.Tensor) -> torch.Tensor:
    """Converts an image from HWB space to RGB space.

    Parameters
    ----------
    hwb : torch.Tensor
        An image in HWB space with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape (*, 3, H, W).
    """
    h, w, b = hwb.unbind(-3)

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

    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 20

    hsl = rgb_to_hwb(img)
    ret = hwb_to_rgb(hsl)

    d = torch.abs(ret - img)
    print(torch.max(d))

    print(timeit('rgb_to_hwb(img)', number=num, globals=locals()))
    print(timeit('hwb_to_rgb(hsl)', number=num, globals=locals()))
