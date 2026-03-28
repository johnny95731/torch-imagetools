__all__ = [
    'box_blur',
    'get_gaussian_kernel',
    'gaussian_blur',
    'guided_filter',
    'max_filter',
    'min_filter',
]

import torch
from torch.nn.functional import avg_pool2d, max_pool2d, pad

from ..core.math import _check_ksize, calc_padding, filter2d
from ..utils.helpers import align_device_type, check_valid_image_ndim


def box_blur(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    normalize: bool = True,
    mode: str = 'reflect',
) -> torch.Tensor:
    """Blurs an image with box kernel (filled with the same value).
    Computes local mean if normalize is True; local sum if normalize
    is False.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    ksize : int | tuple[int, int], default=3
        Kernel size. Must be a positive integer or a sequence of positive
        integers.
    normalize : bool, default=True
        Normalize the kernel to `sum(kernel) == 1`.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Blurred image with the same shape as input.
    """
    check_valid_image_ndim(img, 3)
    _ksize = _check_ksize(ksize, odd=True)
    if mode != 'constant':
        padding = calc_padding(_ksize)
        img = pad(img, padding, mode)
        pool_pad = 0
    else:
        pool_pad = [s // 2 for s in _ksize]
    res = avg_pool2d(
        img, _ksize, stride=1, padding=pool_pad, count_include_pad=False
    )
    if not normalize:
        # (avg_pool2d + mul) faster than conv2d.
        res = res * (_ksize[0] * _ksize[1])
    return res


def get_gaussian_kernel(
    ksize: int | tuple[int, int] = 5,
    sigma: float | tuple[float, float] = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Create a 2D Gaussian kernel.

    Parameters
    ----------
    ksize : int | tuple[int, int], default=5
        Kernel size, `(ky, kx)`. If ksize is non-positive, the value will
        be computed from `sigma_s`:
        `ksize = odd(max(6 * sigma_s / downsample + 1, 3))`,
        where `odd(x)` returns the smallest odd integer such that `odd(x) >= x`.
    sigma : float | tuple[float, float], default=0.0
        The width of gaussian function. If sigma is non-positive, the
        value will be computed from ksize:
        `sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`
    normalize : bool, default=True
        Normalize the summation of the kernel to 1.0.

    Returns
    -------
    torch.Tensor
        2D Gaussian kernel with given ksize.
    """
    _ksize = _check_ksize(ksize, positive=False)
    if isinstance(sigma, (int, float)):
        _sigma = (sigma, sigma)
    elif isinstance(sigma, (tuple, list)):
        if len(sigma) == 0:
            raise ValueError('len(gamma) can not be 0.')
        elif len(sigma) == 1:
            _sigma = (sigma[0], sigma[0])
        else:
            _sigma = (sigma[0], sigma[1])
    else:
        raise TypeError(f'Invalid type of gamma: {type(sigma)}')

    kernels = []
    for ks, std in zip(_ksize, _sigma):
        if ks <= 0 and std <= 0:
            raise ValueError(f'ksize and sigma can not be both non-positive.')
        elif ks <= 0:
            ks = max(int(6 * std + 1), 3) | 1
        elif std <= 0:
            std = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
        half = ks // 2
        kernel1d = torch.linspace(-half, half, ks)
        kernel1d.div_(std).square_().mul_(-0.5).exp_()
        if normalize:
            kernel1d /= torch.sum(kernel1d)
        kernels.append(kernel1d)
    kernel2d = torch.outer(kernels[0], kernels[1])
    return kernel2d


def gaussian_blur(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    sigma: float | tuple[float, float] = 0.0,
    mode: str = 'reflect',
) -> torch.Tensor:
    """Blurs an image with a gaussian kernel.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    ksize : int | tuple[int, int], default=5
        Kernel size. If ksize is non-positive, the value will be computed
        from sigma: `ksize = odd(6 * sigma + 1)`, where odd() returns the
        closest odd integer.
    sigma : float | tuple[float, float], default=0.0
        The width of gaussian function. If sigma is non-positive, the
        value will be computed from ksize:
        `sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Blurred image with the same shape as input.
    """
    check_valid_image_ndim(img, 3)
    kernel = get_gaussian_kernel(ksize, sigma, True)
    kernel = align_device_type(kernel, img)
    bluured = filter2d(img, kernel, mode=mode)
    return bluured


def guided_filter(
    img: torch.Tensor,
    guidance: torch.Tensor | None = None,
    ksize: int | tuple[int, int] = 5,
    eps: float = 0.01,
    mode: str = 'reflect',
):
    """Guided image filter, an edge-preserving smoothing filter [1].

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    guidance : torch.Tensor | None, default=None
        Guidance image with shape `(*, C, H, W)`. If `guidance` is None, then
        `img` will be regard as guidance.
    ksize : int | tuple[int, int], default=5
        Kernel size.
    eps : float, optional
        Regularization parameter. A larger value means the output is more
        smoothing.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Smooth image with shape `(*, C, H, W)`.

    References
    ----------
    [1] He, Kaiming; Sun, Jian; Tang, Xiaoou (2013).
        "Guided Image Filtering". IEEE Transactions on Pattern Analysis and
        Machine Intelligence. 35 (6): 1397-1409. doi:10.1109/TPAMI.2012.213.
    """
    check_valid_image_ndim(img, 3)
    if not torch.is_floating_point(img):
        img = img.float()
    _ksize = _check_ksize(ksize, odd=True)
    # padding
    if mode != 'constant':
        padding = calc_padding(_ksize)
        pool_pad = 0
        _img = pad(img, padding, mode)
    else:
        pool_pad = [s // 2 for s in _ksize]
        _img = img
    #
    mean_i = avg_pool2d(
        _img, _ksize, padding=pool_pad, stride=1, count_include_pad=False
    )
    if guidance is None:
        guidance = img
        mean_g = mean_i
        corr_gi = avg_pool2d(
            _img.square(),
            _ksize,
            stride=1,
            padding=pool_pad,
            count_include_pad=False,
        )
        corr_g = corr_gi
    else:
        guidance = align_device_type(guidance, img)
        _guidance = (
            pad(guidance, padding, mode) if mode != 'constant' else guidance
        )
        mean_g = avg_pool2d(
            _guidance,
            _ksize,
            stride=1,
            padding=pool_pad,
            count_include_pad=False,
        )
        corr_gi = avg_pool2d(
            _guidance * _img,
            _ksize,
            stride=1,
            padding=pool_pad,
            count_include_pad=False,
        )
        corr_g = avg_pool2d(
            _guidance * _guidance,
            _ksize,
            stride=1,
            padding=pool_pad,
            count_include_pad=False,
        )

    var_g = corr_g - mean_g * mean_g
    cov = corr_gi - mean_g * mean_i

    a = cov / (var_g + eps)
    b = mean_i - a * mean_g
    if pool_pad == 0:
        a = pad(a, padding, mode)
        b = pad(b, padding, mode)

    mean_a = avg_pool2d(
        a,
        _ksize,
        stride=1,
        padding=pool_pad,
        count_include_pad=False,
    )
    mean_b = avg_pool2d(
        b,
        _ksize,
        stride=1,
        padding=pool_pad,
        count_include_pad=False,
    )
    res = mean_a * guidance + mean_b
    return res


def max_filter(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    mode: str = 'reflect',
) -> torch.Tensor:
    """Computes the local maximum for each pixel.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    ksize : int | tuple[int, int], default=3
        Kernel size.
    stride : int | tuple[int, int], default=1
        The stride of the sliding window.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Local maximum of `img`.
    """
    is_not_batch = check_valid_image_ndim(img, 3)
    if is_not_batch:
        img = img.unsqueeze(0)

    _ksize = _check_ksize(ksize, odd=True)
    if mode != 'constant':
        padding = calc_padding(_ksize)
        img = pad(img, padding, mode)
        pool_pad = 0
    else:
        pool_pad = [s // 2 for s in _ksize]

    res = max_pool2d(
        img,
        _ksize,
        stride=stride,
        padding=pool_pad,
        ceil_mode=True,
    )
    if is_not_batch:
        res = res.squeeze(0)
    return res


def min_filter(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    mode: str = 'reflect',
) -> torch.Tensor:
    """Computes the local minimum for each pixel.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`.
    ksize : int | tuple[int, int], default=3
        Kernel size.
    stride : int | tuple[int, int], default=1
        The stride of the sliding window.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in
        `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Local minimum of `img`.
    """
    res = -max_filter(-img, ksize, stride, mode)
    return res
