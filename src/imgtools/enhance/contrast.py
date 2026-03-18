"""Automatic contrast enhancement."""

from math import log
from typing import Literal

import torch

from ..color._grayscale import rgb_to_gray
from ..color._yuv import rgb_to_yuv, yuv_to_rgb
from ..filters.rfft import get_gaussian_lowpass
from ..statistics.basic import mean, mean_std
from ..utils.helpers import (
    _to_channel_coeff,
    align_device_type,
    check_valid_image_ndim,
)


# global
def auto_gamma_correction(
    img: torch.Tensor,
    target: float | torch.Tensor = 0.5,
    weight: torch.Tensor | None = None,
):
    """Gamma-correction with the automatically estimated gamma.

    1. Computes `gray = rgb_to_gray(rgb)`
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
    num_ch = img.size(-3)
    target = _to_channel_coeff(target, num_ch)
    target = align_device_type(target, img)
    _mean = mean(img, weight=weight)
    gamma = log(target) / _mean.log()
    res = img.pow(gamma)
    return res


# Local
def local_gamma_correction(
    rgb: torch.Tensor,
    sigma_blur: int = 50,
    basic_gamma: float = 1.0,
    gain: int | float | torch.Tensor = 0.5,
    weight: torch.Tensor | None = None,
):
    """Adaptive Gamma-correction based on local brightness. Inspired by [1].

    1. `gray = rgb_to_gray(rgb)`.
    2. Computes global mean and std `mean, std = mean_std(gray)`.
    3. Computes local mean `local_mean = blur(gray)`.
    4. Computes the gamma by
        `gamma = ((local_mean - mean) / std) * gain + basic_gamma`.
    5. Gamma correction `res = rgb.pow(gamma)`

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : int, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    basic_gamma : float, default=1.0
        The basic gamma value. Must be float or a tensor with shape `(*, 1)`.
    gain : int float | torch.Tensor, default=0.5
        The effect of local mean. Must be float or a tensor with shape `(*, 1)`.
    weight : torch.Tensor | None, default=None
        The weight for computing the mean value and the standard deviation.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.

    References
    ----------
    [1] A. Acharya and A. V. Giri, "Contrast Improvement using Local
        Gamma Correction," 2020 6th International Conference on Advanced
        Computing and Communication Systems (ICACCS), Coimbatore, India,
        2020, pp. 110-114, doi: 10.1109/ICACCS48705.2020.9074386.
    """
    check_valid_image_ndim(rgb)
    num_ch = rgb.size(-3)
    gain = _to_channel_coeff(gain, 1)
    basic_gamma = _to_channel_coeff(basic_gamma, 1)
    if num_ch == 3:
        yuv = rgb_to_gray(rgb)
        gray = yuv[..., :1, :, :]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    mean, std = mean_std(gray, weight=weight)
    gray = gray.add(1e-8)
    #
    gray_f = torch.fft.rfft2(gray)
    sigma_blur = 1 / (2 * torch.pi * sigma_blur)
    lowpass = get_gaussian_lowpass(gray_f, sigma_blur, d=1.0)
    lowpass = align_device_type(lowpass, gray)
    local_mean = gray_f.mul_(lowpass)
    local_mean = torch.fft.irfft2(local_mean, s=gray.shape[-2:])
    #
    m_coeff = gain / std
    gamma = local_mean * m_coeff + (basic_gamma - mean * m_coeff)
    res = rgb.pow(gamma)
    return res


def lide(
    rgb: torch.Tensor,
    std_min: float | torch.Tensor | None = 0.005,
    std_max: float | torch.Tensor | None = None,
    sigma_blur: float = 300,
    model: Literal['gauss', 'laplace'] = 'gauss',
):
    """Automatic contrast enhancement by local intensity distribution
    equalization (LIDE) [1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    std_min : float, default=0.005
        The minimum value of local standard deviation. Must be float, None,
        or a tensor with shape `(*, 1)`.
    std_max : float, default=10.0
        The minimum value of local standard deviation. Must be float, None,
        or a tensor with shape `(*, 1)`.
    sigma_blur : float, default=300
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    model : {'gauss', 'laplace'}, default = 'gauss'
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
    assert model in ('gauss', 'laplace'), (
        f'`model` must be "gauss" or "laplace": {model}'
    )
    num_ch = rgb.size(-3)
    if num_ch == 3:
        yuv = rgb_to_yuv(rgb)
        gray = yuv[..., :1, :, :]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    if std_min is not None:
        std_min = _to_channel_coeff(std_min, 1)
        std_min = align_device_type(std_min, gray)
    if std_max is not None:
        std_max = _to_channel_coeff(std_max, 1)
        std_max = align_device_type(std_max, gray)
    gray = gray.add(1e-8)
    #
    gray_f = torch.fft.rfft2(gray)
    sigma_blurf = 1 / (2 * torch.pi * sigma_blur)
    lowpass = get_gaussian_lowpass(gray_f, sigma_blurf, d=1.0)
    lowpass = align_device_type(lowpass, gray)
    local_mean = gray_f.mul_(lowpass)
    local_mean = torch.fft.irfft2(local_mean, s=gray.shape[-2:])  # type: torch.Tensor
    #
    gray_sq_f = torch.fft.rfft2(gray.square())
    local_sq_mean = gray_sq_f.mul_(lowpass)
    local_sq_mean = torch.fft.irfft2(local_sq_mean, s=gray.shape[-2:])  # type: torch.Tensor
    local_std = local_sq_mean.sub_(local_mean.square()).clip_(0.0).sqrt_()
    local_std = local_std.clip(std_min, std_max)
    #
    if model == 'gauss':
        z_score = (gray - local_mean).div_(local_std.mul_(2**0.5))
        res = torch.erf_(z_score).add_(1.0).mul_(0.5)
    elif model == 'laplace':
        diff = gray - local_mean
        sign = diff.sign()
        z_score = diff.abs_().div_(local_std).mul_(-(2**0.5))
        part = z_score.exp_().mul_(sign)
        res = sign.sub_(part).add_(1.0).mul_(0.5)
    if num_ch == 3:
        yuv[..., :1, :, :] = res
        res = yuv_to_rgb(yuv)
    return res
