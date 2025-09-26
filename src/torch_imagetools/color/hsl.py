__all__ = [
    'rgb_to_hsl',
    'hsl_to_rgb',
]

import torch

from .hsv import hsv_helper


def rgb_to_hsl(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to HSL space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An image in HSL space with shape (*, 3, H, W). The H channel values
        are in the range [0, 360), S and L are in the range of [0, 1].
    """
    hue, amax, amin, delta = hsv_helper(rgb)
    lum = (amax + amin) * 0.5

    sat = delta / (1 - torch.abs(2 * lum - 1))
    torch.nan_to_num(sat, 0.0, out=sat)

    hsl = torch.stack((hue, sat, lum), dim=-3)
    return hsl


def hsl_to_rgb(hsl: torch.Tensor) -> torch.Tensor:
    """Converts an image from HSL space to RGB space.

    Parameters
    ----------
    sv : np.ndarray | torch.Tensor
        An image in HSL space with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape (*, 3, H, W).
    """

    def fn(n):
        val = n + hue_30
        val = torch.remainder(val, 12, out=val)
        # Evaluates min(val, 4 - val)
        val = 3 - torch.abs(val - 6, out=val)  # min(val - 3, 9 - val)
        val = torch.clip(val, -1.0, 1.0, out=val)
        return lum - a * val

    hue, sat, lum = hsl.unbind(-3)

    hue_30 = hue * (1 / 30.0)
    a = sat * torch.where(lum < 0.5, lum, 1 - lum)

    rgb = torch.stack((fn(0), fn(8), fn(4)), dim=-3)
    return rgb
