"""This module including
- Chromatic adaptation
- White balance
- Low-light compensation
- correlate color temperature estimation

Color Balance
-------------

- von Kries transform :
    Chromatic adaptation by scaling the LMS channels from a source white
    point to a target white point.
- Balance by scaling :
    Multiplies each channel of an image by
        ``coeff_channel = maximum / maximum_of_channel.``
- Gray-world balance :
    Multiplies each channel of an image by
        ``coeff_channel = mean / mean_of_channel.``
- White patch balance :
    A generalized version of white patch balance for specified percentiles
    instead of maximums. Multiplies each channel of an RGB image by
    ``coeff_channel = 1 / qtile_of_channel.``
    When q = 1.0, it is the standard white patch balance and equivalent to
    balance by scaling for maximum = 1.
"""

__all__ = [
    'balance_by_scaling',
    'cheng_pca_balance',
    'estimate_illuminant_cheng',
    'get_von_kries_transform_matrix',
    'gray_edge_balance',
    'gray_world_balance',
    'hernandez_andre_approximation',
    'light_compensation_htchen',
    'mccamy_approximation',
    'von_kries_transform',
    'white_patch_balance',
]

from ._balance import (
    balance_by_scaling,
    cheng_pca_balance,
    get_von_kries_transform_matrix,
    gray_edge_balance,
    gray_world_balance,
    von_kries_transform,
    white_patch_balance,
)
from ._cct import (
    hernandez_andre_approximation,
    mccamy_approximation,
)
from .est_illuminant import (
    estimate_illuminant_cheng,
)
from .light import (
    light_compensation_htchen,
)
