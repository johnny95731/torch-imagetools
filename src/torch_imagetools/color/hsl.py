import torch
import numpy as np

from ..utils.helpers import tensorlize
from .hsv import hsv_helper


def rgb_to_hsl(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    hue, amax, amin, delta = hsv_helper(rgb)
    lum = (amax + amin) * 0.5

    sat = delta / (1 - torch.abs(2 * lum - 1))
    torch.nan_to_num(sat, 0.0, out=sat)

    hsl = torch.stack((hue, sat, lum), dim=-3)
    return hsl


def hsl_to_rgb(hsl: np.ndarray | torch.Tensor) -> torch.Tensor:
    hsl = tensorlize(hsl)

    def fn(n):
        val = n + hue_30
        val = torch.remainder(val, 12, out=val)
        # Evaluates min(val, 4 - val)
        val = 3 - torch.abs(val - 6, out=val)  # min(val - 3, 9 - val)
        val = torch.clip(val, -1.0, 1.0, out=val)
        return lum - a * val

    hue: torch.Tensor = hsl[..., 0, :, :]
    sat: torch.Tensor = hsl[..., 1, :, :]
    lum: torch.Tensor = hsl[..., 2, :, :]

    hue_30 = hue * (1 / 30.0)
    a = sat * torch.where(lum < 0.5, lum, 1 - lum)

    rgb = torch.stack((fn(0), fn(8), fn(4)), dim=-3)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 10

    hsl = rgb_to_hsl(img)
    ret = hsl_to_rgb(hsl)

    print(torch.max(torch.abs(ret - img)))

    print(timeit('rgb_to_hsl(img)', number=num, globals=locals()))
    print(timeit('hsl_to_rgb(hsl)', number=num, globals=locals()))
