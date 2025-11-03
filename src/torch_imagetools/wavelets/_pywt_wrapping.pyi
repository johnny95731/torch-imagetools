__all__ = [
    'get_families',
    'get_wavelets',
    'Wavelet',
]

from typing import Literal

import torch

def get_families(short: bool = False) -> list[str]: ...
def get_wavelets(family: str | None = None) -> list[str]: ...

#
class Wavelet:
    family_name: str
    short_family_name: str
    name: str

    orthogonal: bool
    biorthogonal: bool
    symmetry: Literal['asymmetric', 'near symmetric', 'symmetric']

    dec_low: torch.Tensor
    dec_high: torch.Tensor
    rec_low: torch.Tensor
    rec_high: torch.Tensor

    @property
    def dec_len(self) -> int: ...
    @property
    def rec_len(self) -> int: ...

    #
    @property
    def filter_bank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    #
    def to(
        self,
        device: torch.Device = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    #
    @property
    def device(self) -> torch.device: ...
    @property
    def dtype(self) -> torch.dtype: ...

    #
    def dwt(self, img: torch.Tensor) -> list[torch.Tensor]: ...
    def dwt_ll(self, img: torch.Tensor) -> torch.Tensor: ...
    def dwt_hh(self, img: torch.Tensor) -> torch.Tensor: ...
