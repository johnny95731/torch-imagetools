"""Wrap PyWavelets functions and classes.

https://github.com/PyWavelets/pywt
"""

# PyWavelets's License

# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-     The PyWavelets Developers <https://github.com/PyWavelets/pywt>

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__all__ = [
    'get_families',
    'get_wavelets',
    'Wavelet',
]

from typing import Literal

import torch
from pywt import (
    Wavelet as PyWavelet,
)
from pywt import wavelist

from .wavelets import dwt2, dwt2_partial, idwt2


def get_families(short: bool = False) -> list[str]:
    """Return a list of available discrete wavelet families.

    Parameters
    ----------
    short : bool, default=False
        Use short name.

    Returns
    -------
    list[str]
        List of available wavelet families.
    """
    if short:
        res = ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey']
    else:
        res = [
            'Haar',
            'Daubechies',
            'Symlets',
            'Coiflets',
            'Biorthogonal',
            'Reverse biorthogonal',
            'Discrete Meyer (FIR Approximation)',
        ]
    return res


def get_wavelets(family: str | None = None) -> list[str]:
    """Return list of available discrete wavelet names for the given
    family name.

    Parameters
    ----------
    family : str | None, default=None
        Short family name. If the family name is None (default), then all
        support wavelets are returned.

    Returns
    -------
    list[str]
        List of available wavelet names.
    """
    res = wavelist(family, 'discrete')  # type: list[str]
    return res


class Wavelet:
    """Discrete Wavelet Object."""

    family_name: str
    """Wavelet family name."""
    short_family_name: str
    """Wavelet short family name."""
    name: str
    """Wavelet name."""

    orthogonal: bool
    """Whether the wavelet is orthogonal."""
    biorthogonal: bool
    """Whether the wavelet is biorthogonal."""
    symmetry: Literal['asymmetric', 'near symmetric', 'symmetric']
    """`asymmetric`, `near symmetric`, `symmetric`"""

    dec_low: torch.Tensor
    """Scaling coefficients of the decomposition filter."""
    dec_high: torch.Tensor
    """Wavelet coefficients of the decomposition filter."""
    rec_low: torch.Tensor
    """Scaling coefficients of the reconstruction filter."""
    rec_high: torch.Tensor
    """Wavelet coefficients of the reconstruction filter."""

    def __init__(
        self,
        wavelet_name: str,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        normalize: bool = True,
    ):
        """An wavelet object, containing informations of the wavelet.

        Parameters
        ----------
        wavelet_name : str
            Name of the wavelet, e.g., 'haar', 'sym4', 'db8'.
        device : torch.device | str | None, default=None
            The device of the coefficients.
        dtype : torch.dtype | None, default=None
            The dtype of the coefficients. Must be floating point type.
            By default float32.
        normalize : bool, default=True
            Normalize the coefficients by dividing the absolute sum
            of `dec_lo`.
        """
        pywavelet = PyWavelet(wavelet_name)
        attrs = (
            'family_name',
            'short_family_name',
            'name',
            'orthogonal',
            'biorthogonal',
            'symmetry',
        )
        for attr in attrs:
            setattr(self, attr, getattr(pywavelet, attr))

        value_attrs = ('dec_low', 'dec_high', 'rec_low', 'rec_high')
        for attr in value_attrs:
            _filter = getattr(pywavelet, attr[:6])
            _filter = torch.tensor(_filter, dtype=dtype, device=device)
            if attr == 'dec_low':
                s = _filter.sum().item()
            if normalize and attr.startswith('dec'):
                _filter /= s
            elif normalize and attr.startswith('rec'):
                _filter *= s
            setattr(self, attr, _filter)

        self._s = s

        self._verify_dtype()

        self._device = self.dec_low.device
        self._dtype = self.dec_low.dtype

    def _verify_dtype(self):
        value_attrs = ('dec_low', 'dec_high', 'rec_low', 'rec_high')
        for attr in value_attrs:
            tensor = getattr(self, attr)
            if not torch.is_floating_point(tensor):
                raise ValueError(
                    f'self.{attr} must be one of the floating types: '
                    '`torch.float16`/`torch.half`, '
                    '`torch.float32`/`torch.float`,'
                    '`torch.float64`/`torch.double`.'
                    f'Received {tensor.dtype}.'
                )

    @property
    def dec_len(self) -> int:
        """Decomposition filter length."""
        length = len(self.dec_low)
        return length

    @property
    def rec_len(self) -> int:
        """Reconstruction filter length."""
        length = len(self.rec_low)
        return length

    @property
    def filter_bank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns filters tuple for the current wavelet in the following
        order:

        `(dec_lo, dec_hi, rec_lo, rec_hi)`
        """
        bank = (self.dec_low, self.dec_high, self.rec_low, self.rec_high)
        return bank

    def to(
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies `torch.Tensor.to` to decomposition and reconstruction
        filter valuess. Returns the filter bank.
        """
        attrs = ('dec_low', 'dec_high', 'rec_low', 'rec_high')
        for attr in attrs:
            values = getattr(self, attr)  # type: torch.Tensor
            values = values.to(device, dtype)
            setattr(self, attr, values)

        self._verify_dtype()
        return self.filter_bank

    @property
    def device(self) -> torch.device:
        """The device of filters."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The data type of filters."""
        return self._dtype

    @property
    def s(self) -> float:
        """The normalization factor."""
        return self._s

    def dwt2(
        self,
        img: torch.Tensor,
        mode: Literal[
            'constant', 'reflect', 'replicate', 'circular'
        ] = 'reflect',
    ) -> torch.Tensor:
        """Discrete wavelet transform of an image.

        Parameters
        ----------
        img : torch.Tensor
            An image with shape `(*, C, H, W)`.
        mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
            Padding mode. Same as the argument `mode` in
            `torch.nn.functional.pad`.

        Returns
        -------
        torch.Tensor
            Shape `(*, C, 4, Ho, Wo)`, where `Ho = (H + 1) // 2` and `Wo = (W + 1) // 2`.
            The wavelet decomposition components with the following order:

            `[LL, LH, HL, HH]`
        """
        res = dwt2(img, self.dec_low, self.dec_high, mode)
        return res

    def idwt2(
        self,
        img: torch.Tensor,
        mode: Literal[
            'constant', 'reflect', 'replicate', 'circular'
        ] = 'reflect',
    ) -> torch.Tensor:
        """Discrete wavelet transform of an image.

        Parameters
        ----------
        img : torch.Tensor
            An image with shape `(*, C, H, W)`.
        mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
            Padding mode. Same as the argument `mode` in
            `torch.nn.functional.pad`.

        Returns
        -------
        torch.Tensor
            The reconstructed image. Shape `(*, C, 2*H, 2*W)`. Note that the
            val
        """
        res = idwt2(img, self.rec_low, self.rec_high, mode)
        return res

    def dwt2_ll(
        self,
        img: torch.Tensor,
        mode: Literal[
            'constant', 'reflect', 'replicate', 'circular'
        ] = 'reflect',
    ) -> torch.Tensor:
        """Return the low-low component of the discrete wavelet transform.

        Parameters
        ----------
        img : torch.Tensor
            An image with shape `(*, C, H, W)`.
        mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
            Padding mode. Same as the argument `mode` in
            `torch.nn.functional.pad`.

        Returns
        -------
        torch.Tensor
            The low-low component of image. Shape `(*, C, 4, Ho, Wo)`, where
            `Ho = (H + 1) // 2` and `Wo = (W + 1) // 2`.
        """
        res = dwt2_partial(img, self.dec_low, self.dec_high, 'LL', mode)
        return res

    def dwt2_hh(
        self,
        img: torch.Tensor,
        mode: Literal[
            'constant', 'reflect', 'replicate', 'circular'
        ] = 'reflect',
    ) -> torch.Tensor:
        """Return the high-high component of the discrete wavelet transform.

        Parameters
        ----------
        img : torch.Tensor
            An image with shape `(*, C, H, W)`.
        mode : {'constant', 'reflect', 'replicate', 'circular'}, default='reflect'
            Padding mode. Same as the argument `mode` in
            `torch.nn.functional.pad`.

        Returns
        -------
        torch.Tensor
            The high-high component of image. Shape `(*, C, 4, Ho, Wo)`, where
            `Ho = (H + 1) // 2` and `Wo = (W + 1) // 2`.
        """
        res = dwt2_partial(img, self.dec_low, self.dec_high, 'HH', mode)
        return res
