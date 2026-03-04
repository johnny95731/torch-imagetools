"""Basic spatial and frequency domain filters.

- Spatial domain filters
    * Edge: Laplacian, Sobel, Kirsch, etc.
    * Blurring filters: box blur, Gaussian, guided filter
- Frequency domain filters: based on rfft2
    * Laplacian
    * Gaussian highpass/lowpass
    * Butterworthhighpass/lowpass.
"""

__all__ = [
    'gradient_magnitude',
    'kirsch',
    'laplacian',
    'prewitt',
    'robinson',
    'scharr',
    'sobel',
    'box_blur',
    'gaussian_blur',
    'get_gaussian_kernel',
    'guided_filter',
    'get_butterworth_highpass',
    'get_butterworth_lowpass',
    'get_freq_laplacian',
    'get_gaussian_highpass',
    'get_gaussian_lowpass',
]

from ._edges import (
    gradient_magnitude,
    kirsch,
    laplacian,
    prewitt,
    robinson,
    scharr,
    sobel,
)
from .blur import (
    box_blur,
    gaussian_blur,
    get_gaussian_kernel,
    guided_filter,
)
from .rfft import (
    get_butterworth_highpass,
    get_butterworth_lowpass,
    get_freq_laplacian,
    get_gaussian_highpass,
    get_gaussian_lowpass,
)
