import torch
import numpy as np

from ..utils.helpers import tensorlize


def hsv_helper(
    rgb: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate hue (H channel of HSL/HSV) from rgb. Also, returns minimum
    and maximum of rgb.

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        [Hue, min, max, delta = max - min] of an RGB image.
    """
    rgb = tensorlize(rgb)

    r: torch.Tensor = rgb[..., 0, :, :]
    g: torch.Tensor = rgb[..., 1, :, :]
    b: torch.Tensor = rgb[..., 2, :, :]

    amax = torch.max(rgb, dim=-3).values
    amin = torch.min(rgb, dim=-3).values
    delta = amax - amin

    numerator = g - b
    torch.mul(numerator, 3.0**0.5, out=numerator)
    denominator = 2 * r
    torch.sub(denominator, g, out=denominator)
    torch.sub(denominator, b, out=denominator)
    hue = torch.atan2(numerator, denominator) * (180 / torch.pi)
    hue = torch.remainder(hue, 360.0, out=hue)
    # Handle delta = 0
    return (hue, amax, amin, delta)


def rgb_to_hsv(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    hue, amax, _, delta = hsv_helper(rgb)
    sat = delta / amax
    torch.nan_to_num(sat, 0.0, out=sat)
    bri = amax

    hsv = torch.stack((hue, sat, bri), dim=-3)
    return hsv


def hsv_to_rgb(hsv: np.ndarray | torch.Tensor) -> torch.Tensor:
    hsv = tensorlize(hsv)

    def fn(n):
        val = n + hue_60
        val = torch.remainder(val, 6, out=val)
        # Evaluates min(val, 4 - val)
        val = torch.where(val < 2, val, 4 - val, out=val)
        val = torch.clip(val, 0.0, 1.0, out=val)
        return bri - bri * sat * val

    hue: torch.Tensor = hsv[..., 0, :, :]
    sat: torch.Tensor = hsv[..., 1, :, :]
    bri: torch.Tensor = hsv[..., 2, :, :]

    hue_60 = hue * (1 / 60.0)

    rgb = torch.stack((fn(5), fn(3), fn(1)), dim=-3)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (3, 1024, 1024)).type(torch.float32) / 255
    num = 10

    print(timeit('hsv_helper(img)', number=num, globals=locals()))

    hsv = rgb_to_hsv(img)
    ret = hsv_to_rgb(hsv)

    print(torch.max(torch.abs(ret - img)))

    print(timeit('rgb_to_hsv(img)', number=num, globals=locals()))
    print(timeit('hsv_to_rgb(hsv)', number=num, globals=locals()))
