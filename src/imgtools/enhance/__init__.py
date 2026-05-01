"""This module including

- Basic intensity mapping, such as linear, gamma correction, log
  transformation, etc.
- Edge enhancement, e.g., sharpening, unsharp masking.
- Contrast enhancement
- High dynamic range, e.g., Durand2002 [1]_ and Reinhard2002 [2]_.
- Low-light enhancement, including MSRCR [3]_ and MSRCP [3]_.
- Style transfer by Reinhard2002 [4]_.

References
----------
.. [1] F. Durand and J. Dorsey, "Fast bilateral filtering for the display of
    high-dynamic-range images," SIGGRAPH '02', pp. 257-266, Jul. 2002
    doi: 10.1145/566570.566574.
.. [2] Erik Reinhard, Michael Stark, Peter Shirley, and James Ferwerda. 2002.
    Photographic tone reproduction for digital images.
    ACM Trans. Graph. 21, 3 (July 2002), 267-276.
    https://doi.org/10.1145/566654.566575
.. [3] Ana Belén Petro, Catalina Sbert, and Jean-Michel Morel,
    Multiscale Retinex, Image Processing On Line, (2014), pp. 71-88.
    https://doi.org/10.5201/ipol.2014.107
.. [4] E. Reinhard, M. Adhikhmin, B. Gooch and P. Shirley, "Color
    transfer between images," in IEEE Computer Graphics and Applications,
    vol. 21, no. 5, pp. 34-41, July-Aug. 2001, doi: 10.1109/38.946629
"""

__all__ = [
    'adjust_gamma',
    'adjust_inverse',
    'adjust_linear',
    'adjust_log',
    'adjust_sigmoid',
    'high_frequency_emphasis_filter',
    'unsharp_mask',
    'auto_gamma_correction',
    'lide',
    'local_gamma_correction',
    'hist_equalize',
    'match_historgram',
    'match_mean_std',
    'bilateral_hdr',
    'reinhard2002',
    'faster_lime',
    'msrcp',
    'msrcr',
    'retinex',
    'transfer_reinhard',
]

from .basic import (
    adjust_gamma,
    adjust_inverse,
    adjust_linear,
    adjust_log,
    adjust_sigmoid,
    high_frequency_emphasis_filter,
    unsharp_mask,
)
from .contrast import (
    auto_gamma_correction,
    lide,
    local_gamma_correction,
)
from .equlization import (
    hist_equalize,
    match_historgram,
    match_mean_std,
)
from .hdr import (
    bilateral_hdr,
    reinhard2002,
)
from .lowlight import (
    faster_lime,
    msrcp,
    msrcr,
    retinex,
)
from .stylization import (
    transfer_reinhard,
)
