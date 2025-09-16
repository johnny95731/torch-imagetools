import torch


def hsv_helper(
    rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns hue (H channel of HSL/HSV) from rgb, maximum, minimum,
    and (maximum - minimum).

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        [Hue, min, max, delta = max - min] of an RGB image. The range of hue
        is [0, 360), and the range of other tensors are as same as input.
    """
    amax, argmax_rgb = torch.max(rgb, dim=-3)
    amin = torch.min(rgb, dim=-3).values
    delta = amax - amin

    r, g, b = torch.unbind(rgb, dim=-3)

    h1 = (g - b).divide_(delta)
    h2 = (b - r).divide_(delta).add_(2.0)
    h3 = (r - g).divide_(delta).add_(4.0)

    hue = torch.stack((h1, h2, h3), dim=-3)
    hue = torch.gather(hue, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    hue = hue.remainder_(6.0).mul_(60.0).nan_to_num_(0.0, 0.0, 0.0)
    return (hue, amax, amin, delta)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to HSV space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An image in HSV space with shape (*, 3, H, W). The H channel values
        are in the range [0, 360), S and V are in the range of [0, 1].
    """
    hue, amax, _, delta = hsv_helper(rgb)
    sat = delta.divide_(amax).nan_to_num_(0.0, 0.0, 0.0)
    bri = amax

    hsv = torch.stack((hue, sat, bri), dim=-3)
    return hsv


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Converts an image from HSV space to RGB space.

    Parameters
    ----------
    hsv : torch.Tensor
        An image in HSV space with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape (*, 3, H, W).
    """

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
