__all__ = [
    'scaling_coeffs_to_wavelet_coeffs',
    'wavelet_hh',
]

import numpy as np
import torch


def scaling_coeffs_to_wavelet_coeffs(
    scaling: torch.Tensor,
    *_,
    device: torch.DeviceObjType | str | None = None,
) -> torch.Tensor:
    """Calculate coefficients of the wavelet function from given coefficients of
    the scaling function.

    wavelet[i] = (-1)**i * scaling[N - k], where N = scaling.numel() - 1 and
    i = 0, 1, ..., N.

    Parameters
    ----------
    scaling : torch.Tensor
        coefficients of the scaling function.

    Returns
    -------
    torch.Tensor
        _description_
    """
    if isinstance(scaling, np.ndarray):
        scaling = torch.from_numpy(scaling)
    device = scaling.device if device is None else device
    if scaling.device != device:
        scaling = scaling.to(device)
    # Reverse the order and then multiply -1 to odd order elements.
    # wavelet[i] = (-1)**i * scaling[N - k], where N = scaling.numel() - 1
    wavelet = torch.flip(scaling, dims=(0,))
    wavelet[1::2].mul_(-1)
    return wavelet


def wavelet_hh(
    img: torch.Tensor,
    wavelet: torch.Tensor,
) -> torch.Tensor:
    """Calculates the highpass-highpass component of the wavelet decomposition
    by a given wavelet coefficient.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape (*, C, H, W).
    wavelet : torch.Tensor
        The wavelet coefficients (mother wavelet) with shape (N,).

    Returns
    -------
    torch.Tensor
        The highpass-highpass component of the wavelet decomposition with shape
        (*, C, H // 2, W // 2).

    Raises
    ------
    ValueError
        img.ndim should be 3 or 4.
    """
    if (ndim := img.ndim) != 4 and ndim != 3:
        raise ValueError(
            f'Dimention of the image should be 3 or 4, but found {ndim}.'
        )

    single_image = img.ndim == 3
    if single_image:
        img = img.unsqueeze(0)
    num_ch = img.size(-3)

    if wavelet.device != img.device:
        wavelet = wavelet.to(img.device)
    length = wavelet.numel()
    wavelet = wavelet.view(1, 1, 1, length).repeat(num_ch, 1, 1, 1).contiguous()

    padding = (length - 1) // 2
    h = torch.nn.functional.conv2d(  # x-direction
        img,
        weight=wavelet,
        stride=(1, 2),
        padding=(0, padding),
        groups=num_ch,
    )
    hh = torch.nn.functional.conv2d(  # y-direction
        h,
        weight=wavelet.movedim(-2, -1),
        stride=(2, 1),
        padding=(padding, 0),
        groups=num_ch,
    )
    if single_image:
        hh = hh.squeeze(0)
    return hh
