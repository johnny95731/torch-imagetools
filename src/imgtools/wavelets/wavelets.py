import torch

from ..utils.helpers import align_device_type, check_valid_image_ndim
from ..utils.math import filter2d


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
        ```[LL, LH, HL, HH]

    Raises
    ------
    ValueError
        When img.ndim is neither 3 nor 4.
    """
    check_valid_image_ndim(img)

    single_image = img.ndim == 3
    if single_image:
        img = img.unsqueeze(0)
    num_ch = img.size(-3)

    scaling = align_device_type(scaling, img)
    length_scaling = scaling.numel()
    scaling = scaling.view(length_scaling)

    wavelet = align_device_type(wavelet, img)
    length_wavelet = wavelet.numel()
    wavelet = wavelet.view(length_wavelet)

    # [LL, LH, HL, HH]
    decomps = []  # type: list[torch.Tensor]
    for row_vector in (scaling, wavelet):
        for col_vector in (scaling, wavelet):
            kernel_2d = torch.outer(col_vector, row_vector)
            padding = [(length - 1) // 2 for length in kernel_2d.shape]
            kernel_2d = kernel_2d[None, None, :, :].repeat(num_ch, 1, 1, 1)
            dec = torch.nn.functional.conv2d(  # y-direction
                img,
                weight=kernel_2d,
                stride=(2, 2),
                padding=padding,
                groups=num_ch,
            )
            if single_image:
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
        insensitive.

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
    check_valid_image_ndim(img)

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

    res = filter2d(img, kernel_2d)
    return res
