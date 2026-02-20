__all__ = [
    'retinex',
    'msrcr',
    'msrcp',
]

import torch

from ..balance._balance import clipping_balance
from ..color._grayscale import rgb_to_gray
from ..filters.rfft import get_gaussian_lowpass


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
    retinex: torch.Tensor | None = None
    for i, s in enumerate(scales):
        lowpass = get_gaussian_lowpass(img_f, 1 / s, d=1.0, device=img_f.device)
        blurred = torch.fft.irfft2(img_f * lowpass)  # type: torch.Tensor
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
