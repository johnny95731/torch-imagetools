import torch
import numpy as np

from ..utils.helpers import tensorlize
from .hsv import hsv_helper, hsv_to_rgb


def rgb_to_hwb(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    hue, amax, amin, _ = hsv_helper(rgb)
    whiteness = amin
    blackness = 1 - amax

    hwb = torch.stack((hue, whiteness, blackness), dim=-3)
    return hwb


def hwb_to_rgb(hwb: np.ndarray | torch.Tensor) -> torch.Tensor:
    hwb = tensorlize(hwb)

    h: torch.Tensor = hwb[..., 0, :, :]
    w: torch.Tensor = hwb[..., 1, :, :]
    b: torch.Tensor = hwb[..., 2, :, :]

    total = w + b
    exceed = total > 1.0
    w[exceed] /= total[exceed]
    b[exceed] /= total[exceed]

    bri = 1.0 - b
    sat = 1.0 - w / bri
    torch.nan_to_num(sat, 1.0, 1.0, 1.0, out=sat)

    hsv = torch.stack((h, sat, bri), dim=-3)
    rgb = hsv_to_rgb(hsv)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 10

    hsl = rgb_to_hwb(img)
    ret = hwb_to_rgb(hsl)

    d = torch.abs(ret - img)
    print(torch.max(d))
    print(torch.count_nonzero(d < 1e-5) / d.numel())

    # print(timeit('rgb_to_hwb(img)', number=num, globals=locals()))
    # print(timeit('hwb_to_rgb(hsl)', number=num, globals=locals()))
