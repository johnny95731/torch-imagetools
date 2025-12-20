__all__ = [
    'dwt',
    'dwt_partial',
]

import torch

from ..utils.helpers import align_device_type, check_valid_image_ndim


def dwt(
    img: torch.Tensor,
    scaling: torch.Tensor,
    wavelet: torch.Tensor,
) -> list[torch.Tensor]:
    """Discrete wavelet transform of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    scaling : torch.Tensor
        The scaling filter (father wavelet) with shape (K,).
    wavelet : torch.Tensor
        The wavelet filter (mother wavelet) with shape (K,).

    Returns
    -------
    list[torch.Tensor]
        The wavelet decomposition components with the following order:
        - `[LL, LH, HL, HH]`
        The first letter represents the filter (lowpass/highpass)
        in x-direction.

    Raises
    ------
    ValueError
        When img.ndim is neither 3 nor 4.
    """
    length = scaling.numel()
    assert length == wavelet.numel(), 'Kernels must be the same size.'
    is_not_batch = check_valid_image_ndim(img)

    if is_not_batch:
        img = img.unsqueeze(0)
    num_ch = img.size(-3)

    p = (length - 1) // 2
    padding = (p, p)

    scaling = align_device_type(scaling, img)
    wavelet = align_device_type(wavelet, img)
    kernels = (scaling, wavelet)
    # [LL, LH, HL, HH]
    decomps = []  # type: list[torch.Tensor]
    for row_vector in kernels:  # x-direction
        for col_vector in kernels:  # y-direction
            kernel_2d = torch.outer(col_vector, row_vector)
            kernel_2d = kernel_2d.repeat(num_ch, 1, 1, 1)
            dec = torch.nn.functional.conv2d(
                img,
                weight=kernel_2d,
                stride=(2, 2),
                padding=padding,
                groups=num_ch,
            )
            if is_not_batch:
                dec = dec.squeeze(0)
            decomps.append(dec)
    return decomps


def dwt_partial(
    img: torch.Tensor,
    scaling: torch.Tensor | None,
    wavelet: torch.Tensor | None,
    target: str,
) -> torch.Tensor:
    """Returns a component of the discrete wavelet transform (DWT).

    Parameters
    ----------
    img : torch.Tensor
        An image with shape (*, C, H, W).
    scaling : torch.Tensor | None
        The scaling filter (father wavelet) with shape (K,).
    wavelet : torch.Tensor | None
        The wavelet filter (mother wavelet) with shape (K,).
    target : {'LL', 'LH', 'HL', 'HH'}
        The component of discrete wavelet transform. The argument is case
        insensitive. The first letter represents the filter (lowpass/highpass)
        in x-direction.

    Returns
    -------
    torch.Tensor
        The component of the DWT with shape (*, C, H // 2, W // 2).

    Raises
    ------
    ValueError
        If `target` is not one of the valid values: {'LL', 'LH', 'HL', 'HH'}.
    ValueError
        When `img.ndim` is neither 3 nor 4.
    """
    length = scaling.numel()
    assert length == wavelet.numel(), 'Kernels must be the same size.'
    is_not_batch = check_valid_image_ndim(img)

    scaling = align_device_type(scaling, img)
    wavelet = align_device_type(wavelet, img)

    target = target.upper()
    if target == 'LL':
        kernel_2d = torch.outer(scaling, scaling)
    elif target == 'HH':
        kernel_2d = torch.outer(wavelet, wavelet)
    elif target == 'LH':
        kernel_2d = torch.outer(wavelet, scaling)
    elif target == 'HL':
        kernel_2d = torch.outer(scaling, wavelet)
    else:
        raise ValueError(
            f"`target` must be one of 'LL', 'LH', 'HL', 'HH', not {target}"
        )

    num_ch = img.size(-3)
    p = (length - 1) // 2
    padding = (p, p)

    kernel_2d = kernel_2d.repeat(num_ch, 1, 1, 1)
    res = torch.nn.functional.conv2d(
        img,
        weight=kernel_2d,
        stride=(2, 2),
        padding=padding,
        groups=num_ch,
    )
    if is_not_batch:
        res = res.squeeze(0)
    return res
