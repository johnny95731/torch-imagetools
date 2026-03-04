"""Core math tools

- Wavelet transform and its inverse transform.
- Decomposition or matrix decomposition, such as PCA.
-
"""

__all__ = [
    'Wavelet',
    'get_families',
    'get_wavelets',
    'pca',
    'calc_padding',
    'deg_to_rad',
    'filter2d',
    'matrix_transform',
    'rad_to_deg',
    'dwt2',
    'dwt2_partial',
    'idwt2',
]

from ._pywt_wrapping import (
    Wavelet,
    get_families,
    get_wavelets,
)
from .decomposition import (
    pca,
)
from .math import (
    calc_padding,
    deg_to_rad,
    filter2d,
    matrix_transform,
    rad_to_deg,
)
from .wavelets import (
    dwt2,
    dwt2_partial,
    idwt2,
)
