__all__ = [
    'scaling_coeffs_to_wavelet_coeffs',
]

import torch


def scaling_coeffs_to_wavelet_coeffs(scaling: torch.Tensor) -> torch.Tensor:
    """Convert the scaling filter (father wavelet) to the wavelet
    filter (mother wavelet).

    wavelet[i] = (-1)**i * scaling[N - k], where N = scaling.numel() - 1 and
    i = 0, 1, ..., N.

    Parameters
    ----------
    scaling : torch.Tensor
        Coefficients of the scaling filter.

    Returns
    -------
    torch.Tensor
        The wavelet filter (mother wavelet).
    """
    # Reverse the order and then multiply -1 to odd order elements.
    # wavelet[i] = (-1)**i * scaling[N - k], where N = scaling.numel() - 1
    wavelet = torch.flip(scaling, dims=(0,))
    wavelet[1::2].mul(-1)
    return wavelet
