__all__ = [
    'combine_mean_std',
    'histogram',
    'mean',
    'moving_mean',
    'var',
    'moving_var',
    'std',
    'mean_std',
    'moving_mean_std',
    'moments',
    'covar',
    'covar_matrix',
]

from typing import Literal

import torch

from ..filters.blur import box_blur
from ..filters.rfft import get_gaussian_lowpass
from ..utils.helpers import (
    __default_dtype,
    align_device_type,
    check_valid_image_ndim,
)


def combine_mean_std(
    *stats: tuple[torch.Tensor, torch.Tensor, int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Calculate the mean, standard deviation (std), and dataset size of the
    combination of two datasets. The

    The function is present for evaluating the mean and std of a large dataset
    by computing its sub-datasets. To see the inference of the formula,
    check [1].

    This function is not jit-able.

    Parameters
    ----------
    stats : tuple[torch.Tensor, torch.Tensor, int]
        The [mean, standard deviation, number of samples] of dataset(s).
        np.ndarray type is also acceptable.

    Returns
    -------
    torch.tensor
        The mean value of the combined dataset.
    torch.tensor
        The standard deviation of the combined dataset.
    int
        The number of samples of the combined dataset.

    References
    ----------
    [1] stack exchange - How do I combine standard deviations of two groups?
        https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups

    Examples
    --------

    >>> from imgtools.statistics import combine_mean_std, mean_std
    >>>
    >>> stats = [
    >>>     (*mean_std(img, channelwise=True), img.shape[-1] * img.shape[-2])
    >>>     for img in imgs
    >>> ]
    >>> _mean, _std, num_pixels = combine_mean_std(*stats)
    """
    mean_x, std_x, num_x = stats[0][:3]
    if len(stats) == 1:
        return mean_x, std_x, num_x
    for mean_y, std_y, num_y in stats[1:]:
        num_z = num_x + num_y
        mean_z = (num_x * mean_x + num_y * mean_y) / num_z

        var_x = std_x * std_x
        var_y = std_y * std_y

        part_1 = ((num_x - 1) * var_x + (num_y - 1) * var_y) / (num_z - 1)
        part_2 = (mean_x - mean_y) ** 2 * (
            num_x * num_y / (num_z * (num_z - 1))
        )
        std_z = (part_1 + part_2) ** 0.5
        # Set variable to x
        mean_x = mean_z
        std_x = std_z
        num_x = num_z

    return mean_z, std_z, num_z


def histogram(
    img: torch.Tensor,
    bins: int = 256,
    density: bool = False,
) -> torch.Tensor:
    """Compute the histogram of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image in the range of [0, 1] with 2 <= img.ndim <= 4.
    bins : int, default=256
        The number of groups in data range.
    density : bool, default=False
        If true, return the pdf of each channel.

    Returns
    -------
    torch.Tensor
        The histogram or density.

    Examples
    --------

    >>> from imgtools.statistics import histogram
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _hist1 = histogram(img)  # torch.Size([3, 256])
    >>> _hist2 = histogram(img, bins=300)  # torch.Size([3, 300])
    >>> _hist3 = mean(img, density=True)  # torch.Size([3, 256])
    >>> _hist3.sum(-1)  # (1., 1., 1.)
    """
    if not isinstance(bins, int):
        raise TypeError(f'`bins` must be an integer: {type(bins)}.')
    check_valid_image_ndim(img, 2)
    img = (img * (bins - 1)).type(torch.uint8)

    flat_image = img.flatten(start_dim=-2).long()
    hist = torch.zeros(
        img.shape[:-2] + (bins,),
        dtype=torch.int32,
        device=img.device,
    )
    hist.scatter_add_(
        dim=-1, index=flat_image, src=hist.new_ones(1).expand_as(flat_image)
    )
    if density:
        num_el = flat_image.size(-1)
        hist = hist.float() / num_el
    return hist


def mean(
    img: torch.Tensor,
    channelwise: bool = False,
    weight: torch.Tensor | None = None,
):
    """Returns the mean value of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`
    channelwise : bool, default=False
        Computes the mean for each channel instead of for the entire image.
    weight : torch.Tensor | None, default=None
        The weights of pixels.

    Returns
    -------
    torch.Tensor
        The mean value. If `channelwise` is False, the shape is
        `(*, 1, 1, 1)`; otherwise, the shape is `(*, C, 1, 1)`.

    Examples
    --------

    >>> from imgtools.statistics import mean
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _mean = mean(img, channelwise=False)  # torch.Size([1, 1, 1])
    >>> _mean2 = mean(img, channelwise=True)  # torch.Size([3, 1, 1])
    >>>
    >>> weight = torch.rand(512, 512)
    >>> _mean3 = mean(img, weight=weight)  # torch.Size([1, 1, 1])
    >>> torch.allclose(_mean, _mean3)  # False
    """
    dim = (-1, -2) if channelwise else (-1, -2, -3)
    if weight is None:
        mean = torch.mean(img, dim=dim, keepdim=True)
    elif isinstance(weight, torch.Tensor):
        if weight.size(-1) != img.size(-1) or weight.size(-2) != img.size(-2):
            raise ValueError(
                'The shape of `img` and `weight` are not match: '
                f'img.shape = {img.shape} and weight.shape = {weight.shape}.'
            )
        wdim = (-1, -2) if weight.ndim == 2 and len(dim) == 3 else dim
        weight = align_device_type(weight, img)
        weight_sum = weight.sum(wdim, keepdim=True)
        mean = (img * weight).sum(dim, keepdim=True) / weight_sum
    else:
        raise TypeError(f'`weight` must be None or a Tensor: {type(weight)}')
    return mean


def moving_mean(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    fft_approx: bool = False,
    sigma: float | tuple[float, float] = 1,
    mode: str = 'reflect',
) -> torch.Tensor:
    """The 2D moving average of an image. Equals mean blur.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ksize : int | tuple[int, int], default=3
        The size of window.
    fft_approx : bool, default=False
        Uses the frequency domain Gaussian filter to approach mean blur.
        Recommend to use when window size is large.
    sigma : float | tuple[float, float], default 1
        The strength of blurrness.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Moving average with shape `(*, C, H, W)`.

    Examples
    --------

    >>> from imgtools.statistics import moving_mean
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _mean = moving_mean(img)  # torch.Size([3, 512, 512])
    """
    check_valid_image_ndim(img)
    if not fft_approx:
        _mean = box_blur(img, ksize, mode=mode)
    else:
        img_f = torch.fft.rfft2(img)
        dtype = __default_dtype(img)
        kernel = get_gaussian_lowpass(
            img_f,
            sigma,
            d=1,
            spatial_sigma=True,
            dtype=dtype,
            device=img.device,
        )
        _mean = torch.fft.irfft2(img_f * kernel, s=img.shape[-2:])
    return _mean


def var(
    img: torch.Tensor,
    channelwise: bool = False,
    weight: torch.Tensor | None = None,
):
    """Returns the variance of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`
    channelwise : bool, default=False
        Computes the variance for each channel instead of for the
        entire image.
    weight : torch.Tensor | None, default=None
        The weights of pixels.

    Returns
    -------
    torch.Tensor
        The variance. If `channelwise` is False, the shape is
        `(*, 1, 1, 1)`; otherwise, the shape is `(*, C, 1, 1)`.

    Examples
    --------

    >>> from imgtools.statistics import var
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _var = var(img, channelwise=False)  # (1, 1, 1)
    >>> _var2 = var(img, channelwise=True)  # (3, 1, 1)
    >>>
    >>> weight = torch.rand(512, 512)
    >>> _var3 = var(img, weight=weight)  # (1, 1, 1)
    >>> torch.allclose(_var, _var3)  # False
    """
    dim = (-1, -2) if channelwise else (-1, -2, -3)
    if weight is None:
        var = torch.var(img, dim=dim, keepdim=True)
    elif isinstance(weight, torch.Tensor):
        if weight.size(-1) != img.size(-1) or weight.size(-2) != img.size(-2):
            raise ValueError(
                'The shape of `img` and `weight` are not match: '
                f'img.shape = {img.shape} and weight.shape = {weight.shape}.'
            )
        wdim = (-1, -2) if weight.ndim == 2 and len(dim) == 3 else dim
        weight = align_device_type(weight, img)
        weight_sum = weight.sum(wdim, keepdim=True)
        mean = (img * weight).sum(dim, keepdim=True).div(weight_sum)
        sq_mean = (img.square() * weight).sum(dim, keepdim=True).div(weight_sum)
        var = sq_mean - mean.square()
    else:
        raise TypeError(f'`weight` must be None or a Tensor: {type(weight)}')
    return var


def moving_var(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    fft_approx: bool = False,
    sigma: float = 1,
    mode: str = 'reflect',
) -> torch.Tensor:
    """The 2D moving variance of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ksize : int | tuple[int, int], default=3
        The size of window.
    fft_approx : bool, default=False
        Uses the frequency domain Gaussian filter to approach mean blur.
        Recommend to use when window size is large.
    sigma : float | tuple[float, float], default 1
        The strength of blurrness.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Moving variance with shape `(*, C, H, W)`.

    Examples
    --------

    >>> from imgtools.statistics import moving_var
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _var = moving_var(img)  # torch.Size([3, 512, 512])
    """
    check_valid_image_ndim(img)
    if not fft_approx:
        _mean = box_blur(img, ksize, mode=mode)
        _sq_mean = box_blur(img.square(), ksize, mode=mode)
        res = _sq_mean.sub_(_mean.square_())
    else:
        img_f = torch.fft.rfft2(img)
        img_sq_f = torch.fft.rfft2(img.square())
        ori_size = img.shape[-2:]
        dtype = __default_dtype(img)
        kernel = get_gaussian_lowpass(
            img_f,
            sigma,
            d=1,
            spatial_sigma=True,
            dtype=dtype,
            device=img.device,
        )
        _mean = torch.fft.irfft2(img_f.mul_(kernel), s=ori_size)
        _sq_mean = torch.fft.irfft2(img_sq_f.mul_(kernel), s=ori_size)
        res = _sq_mean.sub_(_mean.square_())
    return res


def std(
    img: torch.Tensor,
    channelwise: bool = False,
    weight: torch.Tensor | None = None,
):
    """Returns the standard deviation of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`
    channelwise : bool, default=False
        Computes the standard deviation for each channel instead of for the
        entire image.
    weight : torch.Tensor | None, default=None
        The weights of pixels.

    Returns
    -------
    torch.Tensor
        The standard deviation. If `channelwise` is False, the shape is
        `(*, 1, 1, 1)`; otherwise, the shape is `(*, C, 1, 1)`.

    Examples
    --------

    >>> from imgtools.statistics import std, var
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _std = std(img, channelwise=False)  # (1, 1, 1)
    >>> _std2 = std(img, channelwise=True)  # (3, 1, 1)
    >>> _var = var(img)  # (1, 1, 1)
    >>> torch.allclose(_std, _var.sqrt())  # True
    >>>
    >>> weight = torch.rand(512, 512)
    >>> _std3 = std(img, weight=weight)  # (1, 1, 1)
    >>> torch.allclose(_std, _std3)  # False
    """
    _var = var(img, channelwise, weight)
    std = _var.sqrt()
    return std


def mean_std(
    img: torch.Tensor,
    channelwise: bool = False,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the mean value and the standard deviation (std) of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`
    channelwise : bool, default=False
        Computes the mean and std for each channel instead of for the
        entire image.
    weight : torch.Tensor | None, default=None
        The weights of pixels.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The tuple `(mean, std)`. If `channelwise` is False, the shape of both
        tensors are `(*, 1, 1, 1)`; otherwise, the shape of both
        tensors are `(*, C, 1, 1)`.

    Examples
    --------

    >>> from imgtools.statistics import mean, mean_std, std
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _mean, _std = mean_std(img)
    >>> _mean2 = mean(img)
    >>> _std2 = std(img)
    >>> torch.allclose(_mean, _mean2)  # True
    >>> torch.allclose(_std, _std2)  # True
    """
    dim = (-1, -2) if channelwise else (-1, -2, -3)
    if weight is None:
        std, mean = torch.std_mean(img, dim=dim, keepdim=True)
    elif isinstance(weight, torch.Tensor):
        if weight.size(-1) != img.size(-1) or weight.size(-2) != img.size(-2):
            raise ValueError(
                'The shape of `img` and `weight` are not match: '
                f'img.shape = {img.shape} and weight.shape = {weight.shape}.'
            )
        wdim = (-1, -2) if weight.ndim == 2 and len(dim) == 3 else dim
        weight = align_device_type(weight, img)
        weight_sum = weight.sum(wdim, keepdim=True)
        mean = (img * weight).sum(dim, keepdim=True).div(weight_sum)
        sq_mean = (img.square() * weight).sum(dim, keepdim=True).div(weight_sum)
        std = (sq_mean - mean.square()).sqrt()
    else:
        raise TypeError(f'`weight` must be None or a Tensor: {type(weight)}')
    return mean, std


def moving_mean_std(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    fft_approx: bool = False,
    sigma: float = 10,
    mode: str = 'reflect',
) -> tuple[torch.Tensor, torch.Tensor]:
    """The 2D moving variance of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    ksize : int | tuple[int, int], default=3
        The size of window.
    fft_approx : bool, default=False
        Uses the frequency domain Gaussian filter to approach mean blur.
        Recommend to use when window size is large.
    sigma : float | tuple[float, float], default 1
        The strength of blurrness.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Moving average with shape `(*, C, H, W)`.
    torch.Tensor
        Moving variance with shape `(*, C, H, W)`.

    Examples
    --------

    >>> from imgtools.statistics import moving_mean, moving_mean_std, moving_var
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _mean, _std = moving_mean_std(img)
    >>> _mean2 = moving_mean(img)
    >>> _std2 = moving_var(img).sqrt_()
    >>> torch.allclose(_mean, _mean2)  # True
    >>> torch.allclose(_std, _std2)  # True
    """
    check_valid_image_ndim(img)
    if not fft_approx:
        _mean = box_blur(img, ksize, mode=mode)
        _sq_mean = box_blur(img.square(), ksize, mode=mode)
        _std = _sq_mean.sub_(_mean.square()).sqrt_()
    else:
        img_f = torch.fft.rfft2(img)
        img_sq_f = torch.fft.rfft2(img.square())
        ori_size = img.shape[-2:]
        dtype = __default_dtype(img)
        kernel = get_gaussian_lowpass(
            img_f,
            sigma,
            d=1,
            spatial_sigma=True,
            dtype=dtype,
            device=img.device,
        )
        _mean = torch.fft.irfft2(img_f.mul_(kernel), s=ori_size)
        _sq_mean = torch.fft.irfft2(img_sq_f.mul_(kernel), s=ori_size)
        _std = _sq_mean.sub_(_mean.square()).sqrt_()
    return _mean, _std


def moments(
    img: torch.Tensor,
    order: Literal[3, 4],
    channelwise: bool = False,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Returns the mean value, variance, skewness and excess kurtosis (
    when `order` is 4).

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`
    channelwise : bool, default=False
        Computes the mean and std for each channel instead of for the
        entire image.
    order : {3, 4}, default=3
        The weights of pixels.
    weight : torch.Tensor | None, default=None
        The weights of pixels.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The tuple `(mean, std, skewness)` when `order == 3`. If `channelwise`
        is False, the shape of tensors are `(*, 1, 1, 1)`; otherwise,
        the shape of tensors are `(*, C, 1, 1)`.
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The tuple `(mean, std, skewness, excess_kurtosis)` when `order == 4`.
        If `channelwise` is False, the shape of tensors are `(*, 1, 1, 1)`;
        otherwise, the shape of tensors are `(*, C, 1, 1)`.

    Examples
    --------

    >>> from imgtools.statistics import moments
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _mean, _std, _skewness = moments(img, order=3)
    >>> _mean, _std, _skewness, _kurtosis = moments(img, order=4)
    """
    assert order in (3, 4), f'`order` must be 3 or 4: {order}'
    dim = (-1, -2) if channelwise else (-1, -2, -3)
    if weight is None:
        std, mean = torch.std_mean(img, dim=dim, keepdim=True)
        z_score = (img - mean) / std
        skewness = z_score.pow(3).mean(dim, keepdim=True)
        if order == 3:
            return mean, std, skewness
        kurtosis = z_score.pow(4).mean(dim, keepdim=True)
    elif isinstance(weight, torch.Tensor):
        if weight.size(-1) != img.size(-1) or weight.size(-2) != img.size(-2):
            raise ValueError(
                'The shape of `img` and `weight` are not match: '
                f'img.shape = {img.shape} and weight.shape = {weight.shape}.'
            )
        wdim = (-1, -2) if weight.ndim == 2 and len(dim) == 3 else dim
        weight = align_device_type(weight, img)
        weight_sum = weight.sum(wdim, keepdim=True)
        #
        mean = (img * weight).sum(dim, keepdim=True).div(weight_sum)
        sq_mean = (img.square() * weight).sum(dim, keepdim=True).div(weight_sum)
        std = (sq_mean - mean.square()).sqrt()
        #
        z_score = (img - mean) / std
        skewness = (
            z_score.pow(3).mul(weight).sum(dim, keepdim=True).div(weight_sum)
        )
        if order == 3:
            return mean, std, skewness
        kurtosis = (
            z_score.pow(4).mul(weight).sum(dim, keepdim=True).div(weight_sum)
        )
    else:
        raise TypeError(f'`weight` must be None or a Tensor: {type(weight)}')
    excess_kurtosis = kurtosis - 3.0
    return mean, std, skewness, excess_kurtosis


def covar(
    img1: torch.Tensor,
    img2: torch.Tensor,
    channelwise: bool = False,
) -> torch.Tensor:
    """Computes the covariance of two images.

    Parameters
    ----------
    img1 : torch.Tensor
        Image with shape `(*, C, H, W)`.
    img2 : torch.Tensor
        Image with shape `(*, C, H, W)`.
    channelwise : bool, default=False
        Computes the covariance for each channel instead of for the
        entire image.

    Returns
    -------
    torch.Tensor
        The covariance of two images. If `channelwise` is False, the shape is
        `(*, 1, 1, 1)`; otherwise, the shape is `(*, C, 1, 1)`.

    Examples
    --------

    >>> from imgtools.statistics import covar
    >>>
    >>> img1 = torch.rand(3, 512, 512)
    >>> img2 = torch.rand(3, 512, 512)
    >>> _covar1 = covar(img1, img2)  #  torch.Size([1, 1, 1])
    >>> _covar2 = covar(img1, img2, True)  #  torch.Size([3, 1, 1])
    """
    check_valid_image_ndim(img1)
    check_valid_image_ndim(img2)
    img2 = align_device_type(img2, img1)
    dim = (-1, -2) if channelwise else (-1, -2, -3)
    mean1 = img1.mean(dim=dim, keepdim=True)
    mean2 = img2.mean(dim=dim, keepdim=True)
    mul_mean = (img1 * img2).mean(dim=dim, keepdim=True)

    cov = mul_mean - mean1 * mean2
    return cov


def covar_matrix(img: torch.Tensor) -> torch.Tensor:
    """Computes the covariance matrix of the input image.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.

    Returns
    -------
    torch.Tensor
        The covariance matrix with shape `(*, C, C)`.

    Examples
    --------

    >>> from imgtools.statistics import covar_matrix
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _cov1 = covar_matrix(img)  #  torch.Size([3, 3])
    >>> _cov2 = torch.cov(img.flatten(-2))
    >>> torch.allclose(_cov1, _cov2, atol=1e-6)  # True
    >>>
    >>> img2 = torch.rand(5, 3, 512, 512)
    >>> _cov1 = covar_matrix(img2)  #  torch.Size([5, 3, 3])
    """
    check_valid_image_ndim(img)
    flatted = img.flatten(-2)
    n = flatted.size(-1)
    mean = flatted.mean(dim=-1, keepdim=True)

    covmat = torch.matmul(flatted, flatted.movedim(-1, -2)).div_(n - 1)
    covmat = covmat - (n / (n - 1)) * mean * mean.movedim(-1, -2)
    return covmat
