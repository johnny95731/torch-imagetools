__all__ = [
    'retinex',
    'msrcr',
    'msrcp',
    'faster_lime',
]

import torch

from ..balance._balance import clipping_balance
from ..color._grayscale import rgb_to_gray
from ..filters.rfft import get_butterworth_lowpass, get_gaussian_lowpass
from ..utils.helpers import __default_dtype


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
    dtype = __default_dtype(img)
    device = img.device
    retinex: torch.Tensor | None = None
    for i, s in enumerate(scales):
        lowpass = get_gaussian_lowpass(
            img_f,
            s,
            d=1.0,
            spatial_sigma=True,
            dtype=dtype,
            device=device,
        )
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


def faster_lime(
    img: torch.Tensor,
    alpha: float = 2.0,
    order: float = 1.0,
    gamma: float = 0.5,
):
    """The faster-LIME algorithm [1]. The original LIME algorithm,
    see [2] and [3].

    Parameters
    ----------
    img : torch.Tensor
        Image in the range of [0, 1] with shape `(*, C, H, W)`.
    alpha : float, default=2.0,
        The blurrness of filter.
    order : float, default=1.0
        The order of Butterworth filter.
    gamma : float, default=0.5
        Gamma correction of the esimated illuminant. The higher value means

    Returns
    -------
    torch.Tensor
        Enhanced image. shape `(*, C, H, W)`.

    References
    ----------
    [1] My HackMD, https://hackmd.io/@johnny95731/B1-Hb_Cnbx
    [2] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [3] https://arxiv.org/abs/1605.05034
    """
    t_hat = torch.amax(img, -3, keepdim=True)
    dtype = __default_dtype(img)
    device = img.device

    t_hat_f = torch.fft.rfft2(t_hat)
    fft_filter = get_butterworth_lowpass(
        t_hat_f,
        1 / alpha,
        order,
        d=1,
        dtype=dtype,
        device=device,
    )
    res_f = t_hat_f.mul_(fft_filter)
    img_t = torch.fft.irfft2(res_f, s=t_hat.shape[-2:], out=t_hat)
    # normalize
    mini = img_t.amin((-1, -2), keepdim=True)
    maxi = img_t.amax((-1, -2), keepdim=True)
    img_t.sub_(mini).div_(maxi.sub_(mini))

    img_t.pow_(gamma).add_(1e-8)
    res = img.div(img_t).clip_(0.0, 1.0)
    return res
