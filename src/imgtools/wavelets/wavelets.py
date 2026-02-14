__all__ = [
    'dwt2',
    'dwt2_partial',
    'idwt2',
]

import torch
from torch.nn.functional import conv2d, pad

from ..utils.helpers import align_device_type, check_valid_image_ndim


def dwt2(
    img: torch.Tensor,
    scaling: torch.Tensor,
    wavelet: torch.Tensor,
    mode: str = 'reflect',
) -> torch.Tensor:
    """Performs discrete wavelet transform (DWT) of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    scaling : torch.Tensor
        The scaling filter (father wavelet) with shape `(K,)`.
    wavelet : torch.Tensor
        The wavelet filter (mother wavelet) with shape `(K,)`.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        The DWT components of the image. Shape `(*, C, 4, Ho, Wo)`, where
        `Ho = (H + 1) // 2` and `Wo = (W + 1) // 2`.
        The components has the following order: `[LL, LH, HL, HH]`, or
        equivalently, `[cA, cV, cH, cD]`
    """
    assert scaling.ndim == 1 and wavelet.ndim == 1, (
        'Kernels must be 1 dimension.'
    )
    assert mode in ('constant', 'reflect', 'replicate', 'circular'), (
        '`mode` must be in {"constant", "reflect", "replicate", "circular"}: '
        f'{mode}'
    )
    is_not_batch = check_valid_image_ndim(img)
    if is_not_batch:
        img = img.unsqueeze(0)
    num_batch = img.size(0)
    num_ch = img.size(1)
    length = scaling.numel()
    #
    scaling = align_device_type(scaling, img)
    wavelet = align_device_type(wavelet, img)
    kernels_1d = (scaling, wavelet)
    # prepare kernels
    kernel_2d = []  # type: list[torch.Tensor]
    for row_vector in kernels_1d:  # x-direction
        for col_vector in kernels_1d:  # y-direction
            kernel = torch.outer(col_vector, row_vector)
            kernel = kernel.view(1, 1, length, length)
            kernel_2d.append(kernel)
    kernels = torch.cat(kernel_2d * num_ch)
    #
    p = (length - 1) // 2
    padding = (p, length - 1 - p, p, length - 1 - p)
    _img = pad(img, padding, mode)
    #
    dec = conv2d(_img, weight=kernels, stride=(2, 2), groups=num_ch)
    dec = dec.view(num_batch, num_ch, 4, dec.shape[-2], dec.shape[-1])
    if is_not_batch:
        dec.squeeze_(0)
    return dec


def dwt2_partial(
    img: torch.Tensor,
    scaling: torch.Tensor | None,
    wavelet: torch.Tensor | None,
    target: str,
    mode: str = 'reflect',
) -> torch.Tensor:
    """Returns a component of the discrete wavelet transform (DWT).

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    scaling : torch.Tensor | None
        The scaling filter (father wavelet) with shape `(K,)`.
    wavelet : torch.Tensor | None
        The wavelet filter (mother wavelet) with shape `(K,)`.
    target : {'LL', 'LH', 'HL', 'HH'}
        The component of discrete wavelet transform. The argument is case
        insensitive. The first letter represents the filter (lowpass/highpass)
        in x-direction.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        The component of the DWT with shape `(*, C, Ho, Wo)`, where
        `Ho = (H + 1) // 2` and `Wo = (W + 1) // 2`.
    """
    assert target in ('LL', 'LH', 'HL', 'HH'), (
        f'`target` must be in {{"LL", "LH", "HL", "HH"}}: {target}'
    )
    assert mode in ('constant', 'reflect', 'replicate', 'circular'), (
        '`mode` must be in {"constant", "reflect", "replicate", "circular"}: '
        f'{mode}'
    )
    is_not_batch = check_valid_image_ndim(img)
    num_ch = img.size(-3)
    length = scaling.numel()
    # prepare kernel
    target = target.upper()
    row_vector = scaling if target[0] == 'L' else wavelet
    col_vector = scaling if target[1] == 'L' else wavelet
    row_vector = align_device_type(row_vector, img)
    col_vector = align_device_type(col_vector, img)
    kernel_2d = torch.outer(scaling, scaling)
    kernel_2d = kernel_2d.view(1, 1, length, length).repeat(num_ch, 1, 1, 1)
    #
    p = (length - 1) // 2
    padding = (p, length - 1 - p, p, length - 1 - p)
    _img = pad(img, padding, mode)
    #
    dec = conv2d(_img, weight=kernel_2d, stride=(2, 2), groups=num_ch)
    if is_not_batch:
        dec = dec.squeeze(0)
    return dec


def idwt2(
    img: torch.Tensor,
    scaling: torch.Tensor,
    wavelet: torch.Tensor,
    mode: str = 'reflect',
) -> torch.Tensor:
    """Performs inverse discrete wavelet transform (IDWT) of wavelet.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, 4, H, W)`.
    scaling : torch.Tensor
        The scaling filter (father wavelet) with shape `(K,)`.
    wavelet : torch.Tensor
        The wavelet filter (mother wavelet) with shape `(K,)`.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
        Padding mode. Same as the argument `mode` in `torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        The reconstructed image. Shape `(*, C, 2*H, 2*W)`.

    Notes
    -----
        The value may not correctly recontructed at the boundary of the image.
    """
    assert scaling.ndim == 1 and wavelet.ndim == 1, (
        'Kernels must be 1 dimension.'
    )
    assert mode in ('constant', 'reflect', 'replicate', 'circular'), (
        '`mode` must be in {"constant", "reflect", "replicate", "circular"}: '
        f'{mode}'
    )
    if not (4 <= img.ndim <= 5):
        raise ValueError(f'4 <= img.ndim <= 5: {img.ndim}')
    is_not_batch = img.ndim == 4
    if is_not_batch:
        img = img.unsqueeze(0)
    num_ch = img.size(-4)
    length = scaling.numel()
    # Insert zeros between elements.
    shape = [s for s in img.size()]
    shape[-1] *= 2
    shape[-2] *= 2
    new_x = img.new_zeros(shape)
    new_x[..., ::2, ::2] = img
    #
    scaling = (
        align_device_type(scaling, img)
        .view(1, 1, length, 1)
        .repeat(num_ch, 1, 1, 1)
    )
    wavelet = (
        align_device_type(wavelet, img)
        .view(1, 1, length, 1)
        .repeat(num_ch, 1, 1, 1)
    )
    # padding size
    p_end = (length - 1) // 2
    p_start = length - 1 - p_end
    padding = (0, 0, p_start, p_end)
    #
    components = [
        conv2d(
            pad(c.squeeze_(-3), padding, mode),
            weight=wavelet if i % 2 else scaling,
            groups=3,
        )
        for i, c in enumerate(torch.split(new_x, 1, -3))
    ]
    low = components[0] + components[1]
    high = components[2] + components[3]
    components.clear()
    #
    scaling = scaling.moveaxis(-1, -2).contiguous()
    wavelet = wavelet.moveaxis(-1, -2).contiguous()
    #
    padding = (p_start, p_end, 0, 0)
    low = pad(low, padding, mode)
    high = pad(high, padding, mode)
    #
    rec = conv2d(low, weight=scaling, groups=num_ch) + conv2d(
        high, weight=wavelet, groups=num_ch
    )
    if is_not_batch:
        rec.squeeze_(0)
    return rec
