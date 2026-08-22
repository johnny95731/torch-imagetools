"""Automatic contrast enhancement."""

__all__ = [
    'auto_gamma_correction',
    'local_gamma_correction',
    'lide',
]

from typing import Literal

import torch

from imgtools.color import rgb_to_yuv, yuv_to_rgb
from imgtools.filters.rfft import get_gaussian_lowpass
from imgtools.statistics.basic import histogram, mean, mean_std, moving_mean
from imgtools.utils.helpers import (
    __default_dtype,
    _to_channel_coeff,
    check_valid_image_ndim,
)


# global
def auto_gamma_correction(
    img: torch.Tensor,
    target: float | torch.Tensor = 0.5,
    weight: torch.Tensor | None = None,
):
    """Gamma-correction with the automatically estimated gamma.

    1. `gray = rgb.mean(-3)`.
    2. Computes mean value of `gray`: `mean(gray)`
    3. Computes `gamma = log(target) / log(mean(gray))`
    4. Applies gamma correction with the computed gamma in step 3.

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    target : float | torch.Tensor, default=0.5
        Target brightness. Shape `(*, C)`.
    weight : torch.Tensor | None, default=None
        The weight for computing the mean value.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.

    References
    ----------
    [1] P. Babakhani1, P. Zarei "Automatic gamma correction based on average
        of brightness," Advances in Computer Science: an International Journal.
        Vol. 4, Issue 6, No.18 , Nov. 2015.
    """
    check_valid_image_ndim(img)
    dtype = __default_dtype(img)
    num_ch = img.size(-3)
    target = _to_channel_coeff(target, num_ch, dtype=dtype, device=img.device)
    m = mean(img, weight=weight)
    gamma = target.log_() / m.log()
    res = img.pow(gamma)
    return res


# Local
def local_gamma_correction(
    rgb: torch.Tensor,
    sigma_blur: int = 50,
    gain: int | float | torch.Tensor = 1.3,
    basic_gamma: float = 1.0,
):
    """Adaptive Gamma-correction based on local brightness.

    1. `gray = rgb.mean(-3)`.
    2. Computes local mean `local_mean = GaussianFilter(gray, sigma_blur)`.
    3. Computes the gamma by `gamma = (local_mean - 0.5) * gain + basic_gamma`.
    4. Gamma correction `res = rgb.pow(gamma)`

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : int, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    gain : int | float | torch.Tensor, default=1.3
        The effect of local mean. Must be float or a tensor with shape `(*, 1)`.
    basic_gamma : float, default=1.0
        The basic gamma value. Must be float or a tensor with shape `(*, 1)`.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.
    """
    check_valid_image_ndim(rgb)
    dtype = __default_dtype(rgb)
    device = rgb.device
    num_ch = rgb.size(-3)

    gain = _to_channel_coeff(gain, 1, dtype, device)
    basic_gamma = _to_channel_coeff(basic_gamma, 1, dtype, device)
    if num_ch == 3:
        gray = rgb.mean(-3, keepdim=True)
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    gray = gray.add(1e-8)
    #
    gray_f = torch.fft.rfft2(gray)
    sigma_blur = 1 / (2 * torch.pi * sigma_blur)
    lowpass = get_gaussian_lowpass(
        gray_f, sigma_blur, d=1.0, dtype=dtype, device=device
    )
    local_mean = gray_f.mul_(lowpass)
    local_mean = torch.fft.irfft2(local_mean, s=gray.shape[-2:])
    # gamma = gain * (local_mean - 0.5)  + basic_gamma
    #       = local_mean * gain + (basic_gamma - 0.5 * gain)
    gamma = local_mean.mul(gain).add(basic_gamma.sub(gain, alpha=0.5))
    gamma.relu_()
    res = rgb.pow(gamma)
    return res


def lide(
    rgb: torch.Tensor,
    sigma_blur: float | None = 20.0,
    eta_m: float = 0.6,
    eta_s: float = 0.7,
    std_min: float | torch.Tensor | None = 0.2,
    std_max: float | torch.Tensor | None = None,
    alpha: float = 0,
    gamma: float = 1,
    distrib: Literal[
        'cauchy', 'gaussian', 'hyperbolic', 'laplace', 'logistic'
    ] = 'gaussian',
):
    """Automatic contrast enhancement by modified local intensity
    distribution equalization (LIDE) [1].

    1. Convert image `I` to grayscale `g_in`.
    2. Computes local stats and global stats (mean and std).
    3. Convex combination local stats and global stats:
       `mean = eta_m * m_global + (1-eta_m) * m_local` and
       `std = eta_s * s_global + (1-eta_s) * s_local`.
    4. Translate mean `center = mean - alpha * std`.
    5. Clip standard deviation `std = clip(std, std_min, std_max)`.
    6. Intensity transform by the CDF of probability distribution:
       `g_out = F(g_in, center, std)`.
    7. Tone mapping `O = (g_out/g_in) ** gamma * I`.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : float | None, default=20
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness. If `None` is provided, then the local mean is disabled
    eta_m : float, default=0.6
        The convex combination factor for mean value.
        `mean = eta_m * m_global + (1-eta_m) * m_local`
    eta_s : float, default=0.7
        The convex combination factor for standard deviation.
        `std = eta_s * s_global + (1-eta_s) * s_local`
    std_min : float | torch.Tensor, default=0.02
        The minimum value of the standard deviation.
    std_max : float | torch.Tensor | None, default=None
        The maximum value of the standard deviation.
    alpha : float, alpha = 0
        Brightness controls. A larger value means brighter result.
    gamma : float, alpha = 1
        Stength of the enhancement. A larger value means stronger contrast.
    distrib : {"cauchy", "gaussian", "hyperbolic", "laplace", "logistic"}, default = "gaussian"
        The distribution model.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.

    References
    ------
    [1] Marukatat, S. Image enhancement using local intensity distribution
        equalization. J Image Video Proc. 2015, 31 (2015).
        https://doi.org/10.1186/s13640-015-0085-2
    """
    check_valid_image_ndim(rgb)
    assert distrib in (
        valid_li := (
            'cauchy',
            'gaussian',
            'hyperbolic',
            'laplace',
            'logistic',
        )
    ), f'Invalid value of `model`: {distrib}, valid values: {valid_li}'
    assert sigma_blur is None or (
        isinstance(sigma_blur, (int, float)) and sigma_blur > 0
    ), f'`sigma_blur` must be `None` or a positive number: {sigma_blur}'
    assert isinstance(eta_m, (int, float)), f'`eta_m` must a number: {eta_m}'
    assert isinstance(eta_s, (int, float)), f'`eta_s` must a number: {eta_s}'
    dtype = __default_dtype(rgb)
    device = rgb.device
    num_ch = rgb.size(-3)
    if num_ch == 3:
        yuv = rgb_to_yuv(rgb)
        gray = yuv[..., :1, :, :]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    if std_min is not None:
        std_min = _to_channel_coeff(std_min, 1, dtype, device)
        assert std_min.ndim <= rgb.ndim
    else:
        std_min = rgb.new_zeros((1, 1, 1), dtype=dtype)
    if std_max is not None:
        std_max = _to_channel_coeff(std_max, 1, dtype, device)
        assert std_max.ndim <= rgb.ndim
    #
    mean, std = mean_std(gray)
    if sigma_blur is not None:
        local_mean = moving_mean(gray, sigma_blur, fft_approx=True)
        sq_mean = moving_mean(gray.square(), sigma_blur, fft_approx=True)
        # std(x) = mean(x**2) - mean(x)**2
        local_std = sq_mean.sub_(mean.square())
        # Convex combination
        mean = (
            local_mean
            if abs(eta_m) < 1e-10
            else (
                mean
                if abs(1 - eta_m) < 1e-10
                else torch.add(
                    mean,
                    local_mean,
                    alpha=(1 / eta_m - 1),
                ).mul_(eta_m)
            )
        )
        std = (
            local_std
            if abs(eta_s) < 1e-10
            else (
                std
                if abs(1 - eta_s) < 1e-10
                else torch.add(
                    std,
                    local_std,
                    alpha=(1 / eta_s - 1),
                ).mul_(eta_s)
            )
        )
    # Clip std for controling the contrast
    std = std.clip_(std_min, std_max)
    # Shift mean for controling the brightness
    mean = mean.sub_(std, alpha=alpha) if abs(alpha) > 1e-10 else mean
    # Distibutions
    if distrib == 'cauchy':
        z_score = (gray - mean).div_(std)
        res = z_score.arctan_().mul_(1 / torch.pi).add_(0.5)
    elif distrib == 'gaussian':
        z_score = (gray - mean).div(std.mul_(2**0.5))
        res = torch.erf_(z_score).add_(1.0).mul_(0.5)
    elif distrib == 'hyperbolic':
        z_score = (gray - mean).div_(std)
        res = z_score.mul_(torch.pi / 2).exp_().arctan_().mul_(2 / torch.pi)
    elif distrib == 'laplace':
        diff = gray - mean
        sign = diff.sign()
        z_score = diff.abs_().div_(std).mul_(-(2**0.5))
        part = z_score.exp_().mul_(sign)
        res = sign.sub_(part).add_(1.0).mul_(0.5)
    elif distrib == 'logistic':
        z_score = (gray - mean).div_(std)
        res = z_score.sigmoid_()

    if abs(gamma - 1) > 1e-10:
        res.pow_(gamma)
    if num_ch == 3:
        yuv[..., :1, :, :] = res
        res = yuv_to_rgb(yuv)
    return res


def agcwd(rgb: torch.Tensor, alpha: float = 1.5, bins: int = 256):
    """An implementation of the adaptive gamma correction with weighting
    distribution (AGCWD).

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    alpha : float, default=1.5
        A parameter for correcting the weights.
    bins : int, default=256
        The number of groups in data range.

    References
    ----------
    [1] S. -C. Huang, F. -C. Cheng and Y. -S. Chiu, "Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution," in IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041, March 2013, doi: 10.1109/TIP.2012.2226047
    """
    dtype = __default_dtype(rgb)
    device = rgb.device
    num_ch = rgb.size(-3)
    if num_ch == 3:
        yuv = rgb_to_yuv(rgb)
        gray = yuv[..., :1, :, :]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    pdf, idx = histogram(gray, bins, density=True, ret_index=True)
    mini_pdf = pdf.amin(-1, keepdim=True)
    maxi_pdf = pdf.amax(-1, keepdim=True)
    pdf_w = (
        (pdf.sub_(mini_pdf))
        .div_(maxi_pdf - mini_pdf)
        .pow_(alpha)
        .mul_(maxi_pdf)
    )
    cdf_w = torch.cumsum(pdf_w, -1, dtype=dtype)
    cdf_w /= cdf_w[..., -1:].clone()
    #
    gamma = 1 - cdf_w
    table = (
        torch
        .linspace(0, 1, bins, dtype=dtype, device=device)
        .expand_as(gamma)
        .pow_(gamma)
    )

    flatted_idx = idx.flatten(-2)
    res = torch.gather(table, -1, index=flatted_idx).reshape(gray.shape)
    if num_ch == 3:
        yuv[..., :1, :, :] = res
        res = yuv_to_rgb(yuv)
    return res
