__all__ = [
    'scaling_coeffs_to_wavelet_coeffs',
    'wavelet_hh',
]

import torch

def scaling_coeffs_to_wavelet_coeffs(
    scaling: torch.Tensor,
    *_,
    device: torch.DeviceObjType | str | None = None,
) -> torch.Tensor: ...

#
def wavelet_hh(img: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor: ...
