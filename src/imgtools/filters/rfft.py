"""Basic filters for rfft images. Including Gaussian filter, Butterworth
filter, and Laplacian filter.
"""

__all__ = [
    'get_gaussian_lowpass',
    'get_gaussian_highpass',
    'get_butterworth_lowpass',
    'get_butterworth_highpass',
    'get_freq_laplacian',
]


from typing import Literal
import torch

from ..utils.math import _check_ksize


def get_gaussian_lowpass(
    img_size: int | tuple[int, int] | torch.Tensor,
    sigma: float | tuple[float, float],
    d: float | None = None,
    scale: bool = False,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a 2D Gaussian lowpass filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. Shape `(size_y, size_x)`. Or, the rfft image
        with shape `(..., H, W)`.
    sigma : float | tuple[float, float]
        The width of Gaussian function. Shape `[sigma_y, sigma_x]`.
    d : float | None, default=None
        The sampling length scale. If None, uses 1 / img_size. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    scale : bool, default=False
        Scale the filter by `1 / (2 * torch.pi * )`.
    dtype : torch.dtype, default=None
        The Data type of the filter.
    device : torch.device, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Gaussian lowpass filter.

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> lowpass = get_gaussian_lowpass(img_f, 2)
    >>> blurred_f = img_f * lowpass
    >>> blurred = torch.fft.irfft2(blurred_f)
    """
    if isinstance(img_size, torch.Tensor):
        _ksize = img_size.shape[-2:]
    else:
        _ksize = _check_ksize(img_size, True)
    _ksize = _ksize[0], 2 * _ksize[1] - 2
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
        raise TypeError(f'Invalid type of `gamma`: {type(sigma)}')

    freq_y = (
        torch.fft
        .fftfreq(
            _ksize[0],
            d=1 / _ksize[0] if d is None else d,
            dtype=dtype,
            device=device,
        )
        .square_()
        .div_(-2 * _sigma[0] ** 2)
        .view(-1, 1)
    )
    freq_x = (
        torch.fft
        .rfftfreq(
            _ksize[1],
            d=1 / _ksize[1] if d is None else d,
            dtype=dtype,
            device=device,
        )
        .square_()
        .div_(-2 * _sigma[1] ** 2)
        .view(1, -1)
    )
    kernel2d = (freq_y + freq_x).exp_()
    if scale:
        c = 2 * torch.pi * _sigma[0] * _sigma[1]
        kernel2d.div_(c)
    return kernel2d


def get_gaussian_highpass(
    img_size: int | tuple[int, int] | torch.Tensor,
    sigma: float | tuple[float, float],
    d: float | None = None,
    scale: bool = False,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a 2D Gaussian highpass filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. Shape `(size_y, size_x)`. Or, the rfft image
        with shape `(..., H, W)`.
    sigma : float | tuple[float, float]
        The width of Gaussian function. Shape `[sigma_y, sigma_x]`.
    d : float | None, default=None
        The sampling length scale. If None, uses 1 / img_size. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    scale : bool, default=False
        Scale the filter by `1 / (2 * torch.pi * )`.
    dtype : torch.dtype, default=None
        The Data type of the filter.
    device : torch.device, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Gaussian highpass filter.

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> highpass = get_gaussian_highpass(img_f, 2)
    >>> edge_f = img_f * highpass
    >>> edge = torch.fft.irfft2(edge_f)
    """
    kernel = get_gaussian_lowpass(img_size, sigma, d, scale, dtype, device)
    kernel = kernel[..., 0, 0] - kernel  # highpass
    return kernel


def get_butterworth_lowpass(
    img_size: int | tuple[int, int] | torch.Tensor,
    cutoff: float,
    order: float = 1.0,
    d: float | None = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a 2D Butterworth lowpass filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. Shape `(size_y, size_x)`. Or, the rfft image
        with shape `(..., H, W)`.
    cutoff : float
        The cutoff frequency of Butterworth filter.
    order : float, default=1.0
        The order of Butterworth filter.
    d : float | None, default=None
        The sampling length scale. If None, uses 1 / img_size. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    dtype : torch.dtype, default=None
        The Data type of the filter.
    device : torch.device, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Butterworth lowpass filter.

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> lowpass = get_butterworth_lowpass(img_f, 10)
    >>> blurred_f = img_f * lowpass
    >>> blurred = torch.fft.irfft2(blurred_f)
    """
    if isinstance(img_size, torch.Tensor):
        _ksize = img_size.shape[-2:]
    else:
        _ksize = _check_ksize(img_size, True)
    _ksize = _ksize[0], 2 * _ksize[1] - 2
    if not isinstance(cutoff, (int, float)):
        raise TypeError(f'Invalid type of `cutoff`: {type(cutoff)}')
    elif cutoff <= 0:
        raise ValueError(f'`cutoff` must be positive: {cutoff}')
    if not isinstance(order, (int, float)):
        raise TypeError(f'Invalid type of `order`: {type(order)}')
    elif order <= 0:
        raise ValueError(f'`order` must be positive: {order}')

    freq_y = torch.fft.fftfreq(
        _ksize[0],
        d=1 / _ksize[0] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(-1, 1)
    freq_x = torch.fft.rfftfreq(
        _ksize[1],
        d=1 / _ksize[1] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(1, -1)
    kernel2d = freq_y.square_() + freq_x.square_()  # D(u, v)**2
    # 1 / [1 + (D(u, v) / cutoff) ** (2 * order)]
    kernel2d.div_(cutoff**2).pow_(order).add_(1.0).reciprocal_()
    return kernel2d


def get_butterworth_highpass(
    img_size: int | tuple[int, int] | torch.Tensor,
    cutoff: float,
    order: float = 1.0,
    d: float | None = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a 2D Butterworth highpass filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. Shape `(size_y, size_x)`. Or, the rfft image
        with shape `(..., H, W)`.
    cutoff : float
        The cutoff frequency of Butterworth filter.
    order : float, default=1.0
        The order of Butterworth filter.
    d : float | None, default=None
        The sampling length scale. If None, uses 1 / img_size. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    dtype : torch.dtype, default=None
        The Data type of the filter.
    device : torch.device, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Butterworth highpass filter.

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> highpass = get_butterworth_highpass(img_f, 10)
    >>> edge_f = img_f * highpass
    >>> edge = torch.fft.irfft2(edge_f)
    """
    kernel = get_butterworth_lowpass(img_size, cutoff, order, d, dtype, device)
    kernel = 1 - kernel  # highpass
    return kernel


def get_freq_laplacian(
    img_size: int | tuple[int, int] | torch.Tensor,
    form: Literal['continuous', '4-neighbor', '8-neighbor'] = 'continuous',
    d: float | None = 1.0,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a frequency domain Laplacian filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. Shape `[size_y, size_x]`.
    form : {'continuous', '4-neighbor'}, default='continuous'
        The form of approximation of discrete Laplacian filter in frequency
        domain.

        - `'continuous'`: Discretize the Fourier transform of the continuous
        Laplacian operator. Better
        - `'4-neighbor'`: Computes the Fourier transform of the discrete
        4-neighbor Laplacian operator. This can be used to solve the PDE.
        - `'8-neighbor'`: Computes the Fourier transform of the discrete
        8-neighbor Laplacian operator.
    d : float | None, default=1.0
        The sampling length scale. If None, uses 1 / img_size. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    dtype : torch.dtype, default=None
        The Data type of the filter.
    device : torch.device, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Laplacian filter.

    Notes
    -----
    The results of 4-neighbor and 8-neighbor (in frequency domain) are similar
    to the result by using the convolution. Theese options are present for
    solving PDE in the frequency domain.

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> highpass = get_freq_laplacian(img_f, 10)
    >>> edge_f = img_f * highpass
    >>> edge = torch.fft.irfft2(edge_f)
    """
    if isinstance(img_size, torch.Tensor):
        _ksize = img_size.shape[-2:]
    else:
        _ksize = _check_ksize(img_size, True)
    _ksize = _ksize[0], 2 * _ksize[1] - 2

    freq_y = torch.fft.fftfreq(
        _ksize[0],
        d=1 / _ksize[0] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(-1, 1)  # type: torch.Tensor
    freq_x = torch.fft.rfftfreq(
        _ksize[1],
        d=1 / _ksize[1] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(1, -1)  # type: torch.Tensor

    if form == 'continuous':
        # Discretize the Fourier transform of the continuous Laplacian operator
        freq_y.mul_(2 * torch.pi).square_()
        freq_x.mul_(2 * torch.pi).square_()
        fft_laplacian = -freq_y - freq_x
    elif form == '4-neighbor':
        # The Fourier transform of the discrete 4-neighbor Laplacian
        freq_y.mul_(2 * torch.pi).cos_().mul_(-2.0).add_(4.0)
        freq_x.mul_(2 * torch.pi).cos_().mul_(-2.0)
        fft_laplacian = freq_y + freq_x
    elif form == '8-neighbor':
        # The Fourier transform of the discrete 8-neighbor Laplacian
        freq_y.mul_(2 * torch.pi).cos_().mul_(-2.0)
        freq_x.mul_(2 * torch.pi).cos_().mul_(-2.0)
        fft_laplacian = (freq_y + freq_x).sub_(freq_y * freq_x).add_(8.0)
    else:
        raise ValueError(
            f'`form` must be one of "continuous", "4-neighbor", or "8-neighbor": {form}'
        )
    return fft_laplacian
