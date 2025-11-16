__all__ = [
    'gradient_magnitude',
    'kirsch',
    'laplacian',
    'prewitt',
    'robinson',
    'scharr',
    'sobel',
]

from ._edges import (
    gradient_magnitude,
    kirsch,
    laplacian,
    robinson,
)
from ._prewitt import (
    prewitt,
)
from ._scharr import (
    scharr,
)
from ._sobel import (
    sobel,
)
