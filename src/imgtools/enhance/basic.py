"""Fundamental intensity transformation functions such as linear, gamma
correction, log transformation.
"""

__all__ = [
    'adjust_linear',
    'adjust_gamma',
    'adjust_log',
    'adjust_sigmoid',
    'adjust_inverse',
    'high_frequency_emphasis_filter',
]

from math import log1p

import torch

from ..utils.helpers import (
    align_device_type,
    check_valid_image_ndim,
    to_channel_coeff,
)


# Intensity
def adjust_linear(
    img: torch.Tensor,
    slope: int | float | torch.Tensor,
    center: int | float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """Intensity linear enhancement with a specified center point:

    `result = (img - center) * slope + center`

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    slope : int | float | torch.Tensor
        Slope of the linear function.
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`.
    center : int | float | torch.Tensor, default=0.5
        The center of the image.
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`.

    Returns
    -------
    torch.Tensor
        Enhanced image. Shape is broadcasting according to the shape of the
        arguments.
    """
    check_valid_image_ndim(img)

    num_ch = img.size(-3)
    slope = to_channel_coeff(slope, num_ch)
    slope = align_device_type(slope, img)
    center = to_channel_coeff(center, num_ch)
    center = align_device_type(center, img)

    # (img - center) * slope + center
    # = slope * img + center * (1 - slope)
    bias = center * (1.0 - slope)
    res = img.mul(slope).add(bias)
    return res


def adjust_gamma(
    img: torch.Tensor,
    gamma: int | float | torch.Tensor,
    scale: int | float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Intensity enhencement by gamma correction:

    `result = scale * (img ** gamma)`

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    gamma : int | float | torch.Tensor
        The exponent.
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`
    scale : int | float | torch.Tensor | None, default=None
        Linear scale coefficients.
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`

    Returns
    -------
    torch.Tensor
        Enhanced image. Shape is broadcasting according to the shape of the
        arguments.
    """
    check_valid_image_ndim(img)

    num_ch = img.size(-3)
    gamma = to_channel_coeff(gamma, num_ch)
    gamma = align_device_type(gamma, img)

    res = img.pow(gamma)
    if scale is not None:
        scale = to_channel_coeff(scale, num_ch)
        scale = align_device_type(scale, img)
        res.mul(scale)
    return res


def adjust_log(
    img: torch.Tensor,
    scale: int | float | torch.Tensor | None = log1p(1.0),
) -> torch.Tensor:
    """Intensity enhencement by log transformation:

    `result = scale * log(base, 1 + img)`

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    scale : int | float | torch.Tensor | None, default=log1p(1.0)
        Linear scale coefficients.
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`

    Returns
    -------
    torch.Tensor
        Enhanced image. Shape is broadcasting according to the shape of the
        arguments.
    """
    check_valid_image_ndim(img)

    num_ch = img.size(-3)
    res = img.log()
    if scale is not None:
        scale = to_channel_coeff(scale, num_ch)
        scale = align_device_type(scale, img)
        res.mul(scale)
    return res


def adjust_sigmoid(
    img: torch.Tensor,
    shift: int | float | torch.Tensor = 0.5,
    gain: int | float | torch.Tensor = 10.0,
) -> torch.Tensor:
    """Intensity enhencement by sigmoid function:

    `result = sigmoid(gain * (img - cutoff))`

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    shift : int | float | torch.Tensor | None, default=0.5
        Shift of the image intensity.
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`
    gain : int | float | torch.Tensor | None, default=10.0
        Multipler of the derivative of the sigmoid function
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`

    Returns
    -------
    torch.Tensor
        Enhanced image. Shape is broadcasting according to the shape of the
        arguments.
    """
    check_valid_image_ndim(img)

    num_ch = img.size(-3)
    shift = to_channel_coeff(shift, num_ch)
    shift = align_device_type(shift, img)
    gain = to_channel_coeff(gain, num_ch)
    gain = align_device_type(gain, img)

    res = (img - shift).mul(gain).sigmoid()
    return res


def adjust_inverse(
    img: torch.Tensor,
    maxi: int | float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Invert the intensity of the image::

        result = maxi - img

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    maxi : int | float | torch.Tensor | None, default=1.0
        The maximum of the range of the image.
        The tensor must have shape `(1 or C,)` or `(B, 1 or C)`

    Returns
    -------
    torch.Tensor
        Enhanced image. Shape is broadcasting according to the shape of the
        arguments.
    """
    check_valid_image_ndim(img)

    num_ch = img.size(-3)
    maxi = to_channel_coeff(maxi, num_ch)
    maxi = align_device_type(maxi, img)

    res = maxi - img
    return res


# Frequency
def high_frequency_emphasis_filter(
    lowpass: torch.Tensor,
    k: float = 1.0,
    offset: float = 1.0,
) -> torch.Tensor:
    """Create a high frequency emphasis filter (HFEF) from a given lowpass filter.
    `hfef = offset + k * (1 - lowpass)`.

    Parameters
    ----------
    lowpass : torch.Tensor
        An lowpass filter.
    k : float, default=1.0
        The scale of the high frequency.
    offset : float, default=1.0
        The offset of the filter.

    Returns
    -------
    torch.Tensor
        The high frequency emphasis filter.

    Notes
    -----
    When `offset` equals 1, the filter is the same as the unsharp masking.
    When this filter is applied to the Fourier transform of log(image),
    the HFEF can be regarded as homomorphic filter

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> lowpass = get_gaussian_lowpass(img_f, 2)
    >>> unsharp_masking = high_frequency_emphasis_filter(lowpass, 1.2)
    >>> enhanced_f = img_f * unsharp_masking
    >>> enhanced = torch.fft.irfft2(enhanced_f)
    >
    >>> hfef = high_frequency_emphasis_filter(lowpass, 1.2, 1.5)
    >>> enhanced_f = img_f * hfef
    >>> enhanced = torch.fft.irfft2(enhanced_f)
    >
    >>> log_img = torch.log1p(img)
    >>> img_f = torch.fft.rfft2(log_img)
    >>> lowpass = get_gaussian_lowpass(img_f, 2)
    >>> homomorphic = high_frequency_emphasis_filter(lowpass, 2.6, 3.0)
    >>> enhanced_f = img_f * homomorphic
    >>> log_enhanced = torch.fft.irfft2(enhanced_f)
    >>> enhanced = torch.expm1(log_enhanced)
    """
    if not isinstance(k, (int, float)):
        raise TypeError(f'Invalid type of `k`: {type(k)}')
    if not isinstance(offset, (int, float)):
        raise TypeError(f'Invalid type of `offset`: {type(offset)}')
    hfef = lowpass.mul(-k).add(offset + k)  # = offset + k * (1 - lowpass)
    return hfef
