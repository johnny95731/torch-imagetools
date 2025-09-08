import torch
import numpy as np
from kornia import color

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
        [Hue, min, max, delta = max - min] of an RGB image. The range of hue
        is [0, 360), and the range of other tensors are as same as input.
    """
    rgb = tensorlize(rgb)

    amax, argmax_rgb = torch.max(rgb, dim=-3)
    amin = torch.min(rgb, dim=-3).values
    delta = amax - amin

    deltac = torch.where(delta == 0.0, torch.ones_like(delta), delta)
    rc, gc, bc = torch.unbind((amax.unsqueeze(-3) - rgb), dim=-3)

    h1 = bc - gc
    h2 = (2.0 * deltac).add_(rc).subtract_(bc)
    h3 = (4.0 * deltac).add_(gc).subtract_(rc)

    hue = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    hue = torch.gather(hue, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    hue = hue.mul_(1 / 6).remainder_(1.0).mul_(360.0)
    return (hue, amax, amin, delta)


def hsv_helper2(
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
        [Hue, min, max, delta = max - min] of an RGB image. The range of hue
        is [0, 360), and the range of other tensors are as same as input.
    """
    rgb = tensorlize(rgb)

    r, g, b = torch.unbind(rgb, dim=-3)

    amax = torch.max(rgb, dim=-3).values
    amin = torch.min(rgb, dim=-3).values
    delta = amax - amin

    numerator = g - b
    numerator.mul_(3.0**0.5)
    denominator = 2 * r
    denominator.sub_(g).sub_(b)
    hue = torch.atan2(numerator, denominator).mul_(180 / torch.pi)
    hue = torch.remainder(hue, 360.0, out=hue)
    return (hue, amax, amin, delta)


def rgb_to_hsv(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    hue, amax, _, delta = hsv_helper(rgb)
    sat = delta / amax
    torch.nan_to_num(sat, 0.0, 0.0, 0.0, out=sat)
    bri = amax

    hsv = torch.stack((hue, sat, bri), dim=-3)
    return hsv


def hsv_to_rgb(hsv: np.ndarray | torch.Tensor) -> torch.Tensor:
    hsv = tensorlize(hsv)

    def fn(n):
        val = n + hue_60
        val = torch.remainder(val, 6.0, out=val)
        # Evaluates min(val, 4 - val)
        temp = 4 - val
        val = torch.where(val < temp, val, temp, out=val)
        val = torch.clip(val, 0.0, 1.0, out=val)
        return bri * (1.0 - sat * val)

    hue: torch.Tensor = hsv[..., 0, :, :]
    sat: torch.Tensor = hsv[..., 1, :, :]
    bri: torch.Tensor = hsv[..., 2, :, :]

    hue_60 = hue * (1 / 60.0)

    rgb = torch.stack((fn(5.0), fn(3.0), fn(1.0)), dim=-3)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (8, 3, 512, 512)).type(torch.float32) / 255
    num = 15

    print(timeit('hsv_helper(img)', number=num, globals=locals()))
    print(timeit('hsv_helper2(img)', number=num, globals=locals()))

    hsv = rgb_to_hsv(img)
    ret = hsv_to_rgb(hsv)

    d = torch.abs(ret - img)
    print(torch.max(d).item())

    # print(timeit('rgb_to_hsv(img)', number=num, globals=locals()))
    # print(timeit('hsv_to_rgb(hsv)', number=num, globals=locals()))
