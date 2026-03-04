__all__ = [
    'retinex',
    'msrcr',
    'msrcp',
    'light_compensation_htchen',
]

import torch
from torch.nn.functional import avg_pool2d

from ..balance._balance import clipping_balance
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
from ..color._grayscale import rgb_to_gray
from ..filters.rfft import get_gaussian_lowpass
from ..utils.helpers import check_valid_image_ndim


def retinex(
    img: torch.Tensor,
    scales: list[float] = [100, 500, 1000],
    weights: list[float] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Low-light enhance by using multi-scale retinex.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    scales : list[float], default=[100, 500, 1000]
        The strength of the blurriness. The sigma of the Gaussian filter is
        `1 / scale`.
    weights : list[float] | None, default=None
        Weights of the scales. If None, the weight is `1 / len(scales)` for
        each scale.
    normalize : bool, default=True
        Normalize the result to `[0, 1]`.

    Returns
    -------
    torch.Tensor
        Low-light enhanced image with shape `(*, C, H, W)`.

    Notes
    -----
    The blurring is performed on frequency domain. And the relation of the
    sigma for a Gaussian filter between spatial domain and frequency domain
    is `sigma_s * sigma_f = 1 / 2 * pi`.

    Examples
    --------

    >>> from imgtools import enhance
    >>> res = enhance.msr(img)
    """
    log_img = img.log1p()
    img_f = torch.fft.rfft2(img)
    rec_size = img.shape[-2:]
    retinex: torch.Tensor | None = None
    for i, s in enumerate(scales):
        lowpass = get_gaussian_lowpass(img_f, 1 / s, d=1.0, device=img_f.device)
        blurred = torch.fft.irfft2(img_f * lowpass, s=rec_size)  # type: torch.Tensor
        single = log_img - blurred.log1p()  # single-scale
        if weights is not None:
            single = single * weights[i]
        retinex = single if retinex is None else (retinex + single)
    if weights is None:
        retinex = retinex / len(scales)
    if normalize:
        mini = retinex.amin((-1, -2), keepdim=True)
        delta = retinex.amax((-1, -2), keepdim=True).sub_(mini)
        retinex = (retinex - mini) / delta
    return retinex


def msrcr(
    img: torch.Tensor,
    scales: list[float] = [100, 500, 1000],
    weights: list[float] | None = None,
    dark_percent: float = 0.02,
    light_percent: float = 0.02,
):
    """Low-light enhance by using multi-scale retinex with color restoration [1].

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    scales : list[float], default=[100, 500, 1000]
        The strength of the blurriness. The sigma of the Gaussian filter is
        `1 / scale`.
    weights : list[float] | None, default=None
        Weights of the scales. If None, the weight is `1 / len(scales)` for
        each scale.
    dark_percent : float, default=0.02
        The percentage of clipped darkest intensities.
    light_percent : float, default=0.02
        The percentage of clipped brightest intensities.

    Returns
    -------
    torch.Tensor
        Low-light enhanced image with shape `(*, C, H, W)`.

    Notes
    -----
    The blurring is performed on frequency domain. And the relation of the
    sigma for a Gaussian filter between spatial domain and frequency domain
    is `sigma_s * sigma_f = 1 / 2 * pi`.

    References
    ----------
    [1] Ana Belén Petro, Catalina Sbert, and Jean-Michel Morel,
        Multiscale Retinex, Image Processing On Line, (2014), pp. 71-88.
        https://doi.org/10.5201/ipol.2014.107

    Examples
    --------

    >>> from imgtools import enhance
    >>> res = enhance.msrcr(img)
    """
    _msrcr = retinex(img, scales, weights, normalize=False)
    # color restoration
    bias = img.sum((-3), keepdim=True).log1p()
    _msrcr = _msrcr * (img.mul(125.0).log1p() - bias)
    # color balance
    _msrcr = clipping_balance(_msrcr, dark_percent, light_percent)
    return _msrcr


def msrcp(
    rgb: torch.Tensor,
    scales: list[float] = [100, 500, 1000],
    weights: list[float] | None = None,
    dark_percent: float = 0.04,
    light_percent: float = 0.04,
):
    """Low-light enhance by using multi-scale retinex with color
    preservation [1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, C, H, W)`.
    scales : list[float], default=[100, 500, 1000]
        The strength of the blurriness. The sigma of the Gaussian filter is
        `1 / scale`.
    weights : list[float] | None, default=None
        Weights of the scales. If None, the weight is `1 / len(scales)` for
        each scale.
    dark_percent : float, default=0.02
        The percentage of clipped darkest intensities.
    light_percent : float, default=0.02
        The percentage of clipped brightest intensities.

    Returns
    -------
    torch.Tensor
        Low-light enhanced image with shape `(*, C, H, W)`.

    Notes
    -----
    The blurring is performed on frequency domain. And the relation of the
    sigma for a Gaussian filter between spatial domain and frequency domain
    is `sigma_s * sigma_f = 1 / 2 * pi`.

    References
    ----------
    [1] Ana Belén Petro, Catalina Sbert, and Jean-Michel Morel,
        Multiscale Retinex, Image Processing On Line, (2014), pp. 71-88.
        https://doi.org/10.5201/ipol.2014.107

    Examples
    --------

    >>> from imgtools import enhance
    >>> res = enhance.msrcp(img_rgb)
    """
    gray = rgb_to_gray(rgb)
    _msr = retinex(gray, scales, weights, normalize=False)
    # color balance
    temp = clipping_balance(_msr, dark_percent, light_percent)
    maxi_ch = rgb.amax(-3, True)
    coeff = torch.minimum(1.0 / maxi_ch, temp / gray).nan_to_num(0.0)
    _msrcr = rgb * coeff
    return _msrcr


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

    .. code-block:: python
        :linenos:

        lum_brighter = scaler * lum * (1 + log(2 - lum))
        lum_dark = (blurred_sat + 1) / (2 * (1 + blurred_lum) ** power)
        new_lum = lum_brighter * lum_dark

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in range of [0, 1] with shape `(*, C, H, W)`.
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
        Balanced RGB image in range of [0, 1] with shape `(*, C, H, W)`.

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

    Examples
    --------

    >>> from imgtools.balance import light_compensation_htchen
    >>>
    >>> rgb = torch.rand((3, 512, 512))
    >>> balanced = light_compensation_htchen(rgb)
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
    d_saturation = (1.0 + mean_sat).mul(0.5)  # range: [0.5, 1]
    d_brightness = (1.0 + mean_lum).pow(power)  # range: [1, 2**power]
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
