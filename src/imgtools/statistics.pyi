__all__ = [
    'combine_mean_std',
    'estimate_noise_from_wavelet',
    'estimate_noise_from_wavelet_2',
]

import torch

#
def combine_mean_std(
    *stats: tuple[torch.Tensor, torch.Tensor, int],
) -> tuple[torch.Tensor, torch.Tensor, int]: ...

#
def estimate_noise_from_wavelet(hh: torch.Tensor) -> float: ...
def estimate_noise_from_wavelet_2(
    hh: torch.Tensor, maximum: float | int = 1.0
) -> float: ...
