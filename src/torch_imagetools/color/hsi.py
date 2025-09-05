import torch
import numpy as np

from ..utils.helpers import tensorlize
from .hsv import hsv_helper, hsv_to_rgb


def rgb_to_hsi(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    hue, _, amin, _ = hsv_helper(rgb)

    r: torch.Tensor = rgb[..., 0, :, :]
    g: torch.Tensor = rgb[..., 1, :, :]
    b: torch.Tensor = rgb[..., 2, :, :]

    intensity = (r + g + b) / 3.0
    sat = 1 - amin / intensity
    torch.nan_to_num(sat, 0.0, 0.0, 1.0, out=sat)

    hsi = torch.stack((hue, sat, intensity), dim=-3)
    return hsi


def hsi_to_rgb(hwb: np.ndarray | torch.Tensor) -> torch.Tensor:
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

    hsl = rgb_to_hsi(img)
    ret = hsi_to_rgb(hsl)

    d = torch.abs(ret - img)
    print(torch.max(d))
    print(torch.count_nonzero(d < 1e-5) / d.numel())

    # print(timeit('rgb_to_hsi(img)', number=num, globals=locals()))
    # print(timeit('hsi_to_rgb(hsl)', number=num, globals=locals()))
