__all__ = [
    'light_compensation_htchen',
]

import torch
from torch.nn.functional import avg_pool2d

from ..color import (
    hsi_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    lab_to_rgb,
    luv_to_rgb,
    rgb_to_hsi,
    rgb_to_hsl,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_luv,
    rgb_to_yuv,
    yuv_to_rgb,
)
from ..utils.helpers import check_valid_image_ndim


def light_compensation_htchen(
    rgb: torch.Tensor,
    scaler: int | float = 1.0,
    power: int | float = 0.1,
    space: str = 'LAB',
    overflow: str = 'norm',
) -> torch.Tensor:
    """Light compensation by H. T. Chen's method [1]. This method can be found
    in [2] and [3].

    Enhance the luminance channel by
    ```
        lum_brighter = scaler * lum * (1 + log(2 - lum))
        lum_dark = (blurred_sat + 1) / (2 * (1 + blurred_lum) ** power)
        new_lum = lum_brighter * lum_dark
    ```

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in range of [0, 1] with shape (*, C, H, W).
    scaler : int | float, default=1.0
        The strength of the enhancement.
    power : int | float, default=0.1
        The strength of overexpose suppression.
    space : {'LAB', 'LUV', 'YUV', 'HSV', 'HSL', 'HSI'}, default='LAB'
        The color space to get luminance and saturation.
    overflow : {'clip', 'norm', 'both'}, default='norm'
        Handle the result luminance: clip, normalize (to [0, 1]), or clip
        then normalize. Other value means ignoring.

    Returns
    -------
    torch.Tensor
        Balanced RGB image in range of [0, 1] with shape (*, C, H, W).

    Raises
    ------
    ValueError
        When argument `sapce` is not valid.

    References
    ----------
    [1] H. T. Chen, "An Algorithm for Light Compensation," Master Thesis,
        Department of Computer Science and Information Engineering,
        National Taiwan University, 2008.
    [2] C. F. Chang, C. S. Fuh, "Light Compensation," Proceedings of IPPR
        Conference on Computer Vision, Graphics, and Image Processing,
        Shitou, Taiwan, B3-9, p. 87, 2009.
    [3] PDF from C. S. Fuh's personal webpsite
        https://www.csie.ntu.edu.tw/~fuh/personal/LightCompensation.pdf
    """
    check_valid_image_ndim(rgb)

    space = space.upper()
    if space == 'LAB':
        lab = rgb_to_lab(rgb)
        lum, a, b = lab.unbind(-3)
        sat = (a.square() + b.square()).sqrt()
    elif space == 'LUV':
        lab = rgb_to_luv(rgb)
        lum, u, v = lab.unbind(-3)
        sat = (u.square() + v.square()).sqrt()
    elif space == 'YUV':
        yuv = rgb_to_yuv(rgb)
        lum, u, v = yuv.unbind(-3)
        sat = (u.square() + v.square()).sqrt()
    elif space == 'HSV':
        hsv = rgb_to_hsv(rgb)
        hue, sat, lum = hsv.unbind(-3)
    elif space == 'HSL':
        hsl = rgb_to_hsl(rgb)
        hue, sat, lum = hsl.unbind(-3)
    elif space == 'HSI':
        hsi = rgb_to_hsi(rgb)
        hue, sat, lum = hsi.unbind(-3)
    else:
        raise ValueError(
            f'Invalid argument, space={space}. Expect to be one of the '
            f'following: {", ".join(("LAB", "LUV", "YUV", "HSV", "HSL", "HSI"))}'
        )

    # Bright image.
    img_brighter = scaler * (1.0 + torch.log(2.0 - lum)) * lum

    mean_lum = avg_pool2d(lum.unsqueeze(-3), 5, 1, 2).squeeze(-3)
    mean_sat = avg_pool2d(sat.unsqueeze(-3), 5, 1, 2).squeeze(-3)
    # Suppression overexposed region.
    d_saturation = (1.0 + mean_sat).mul_(0.5)  # range: [0.5, 1]
    d_brightness = (1.0 + mean_lum).pow_(power)  # range: [1, 2**power]
    img_dark = d_saturation / d_brightness  # range: [2 ** (-1 - power), 1]

    new_lum = img_brighter * img_dark

    overflow = overflow.lower()
    if overflow == 'clip' or overflow == 'both':
        new_lum = new_lum.clip(0.0, 1.0)
    if overflow == 'norm' or overflow == 'both':
        mini = torch.amin(new_lum)
        maxi = torch.amax(new_lum)
        new_lum = (new_lum - mini) * (1 / (maxi - mini))

    if space == 'LAB':
        merged = torch.stack((new_lum, a, b), dim=-3)
        res = lab_to_rgb(merged)
    elif space == 'LUV':
        merged = torch.stack((new_lum, u, v), dim=-3)
        res = luv_to_rgb(merged)
    elif space == 'YUV':
        merged = torch.stack((new_lum, u, v), dim=-3)
        res = yuv_to_rgb(merged)
    elif space == 'HSV':
        merged = torch.stack((hue, sat, new_lum), dim=-3)
        res = hsv_to_rgb(merged)
    elif space == 'HSL':
        merged = torch.stack((hue, sat, new_lum), dim=-3)
        res = hsl_to_rgb(merged)
    elif space == 'HSI':
        merged = torch.stack((hue, sat, new_lum), dim=-3)
        res = hsi_to_rgb(merged)
    return res
