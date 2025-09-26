"""Operations about color models.

Color Space Transformations
---------------------------

Provides color space transformations between color spaces. Except LMS space,
all the color spaces can be converted from/to RGB color space.

- RGB based color spaces : RGB, YUV, HSV, HSL, HSI, HWB
    * RGB : Red Green Blue\\
        Most color space can be transformed from this space.
        Supports the linearize and the gamma-ize of several RGB models: sRGB,
        Display P3, Adobe RGB, wide-gamut, ProPhoto RGB, REC. 2020.
    * YUV : Luminance u v\\
        The color space used in the television.
        The coefficients of Y channel follows the SDTV standard (BT.470).
        And the U, V channels are scaled to the range of [-0.5, 0.5].
    * HSV : Hue Saturation Value\\
        A cylindrical-coordinate representation in an RGB color model.
        The resulting solid is a cone.
    * HSL : Hue Saturation Lightness\\
        A cylindrical-coordinate representation in an RGB color model.
        The resulting solid is a bicone.
    * HSI : Hue Saturation Intensity\\
        A cylindrical-coordinate representation in an RGB color model.
        The resulting solid is a bicone.
    * HWB : Hue Whitness Blackness\\
        A cylindrical-coordinate of RGB color model.
    * Gray : Grayscale\\
        Same as the Y channel of YUV color space.

- CIE based color spaces: CIE XYZ, LMS, CIE LAB, CIE LUV
    * CIE XYZ : X Y Z\\
        CIE 1931 color space, which define the relationship between the
        visible spectrum and human color vision.
    * LMS : long medium short\\
        The cone response space, which is based on the human visual system.
        This space can only be converted from/to CIE XYZ space.
    * CIE LAB : Lightness a* b*\\
        L*a*b* color space, which was intended as a perceptually uniform space.
        a* represents green-red opponent colors, and b* represents
        blue-yellow opponent colors.
    * CIE LUV : Lightness u* v*\\
        L*u*v* color space, which attempted perceptual uniformity, like L*a*b*.

- Other color spaces: HED
    * HED : Haematoxylin Eosin DAB
            Haematoxylin-Eosin-DAB. See [Ruifrok] for details.

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

Reference
---------

[Ruifrok] : A. C. Ruifrok and D. A. Johnston, “Quantification of
    histochemical staining by color deconvolution.,” Analytical and
    quantitative cytology and histology / the International Academy of
    Cytology [and] American Society of Cytology,
    vol. 23, no. 4, pp. 291-9, Aug. 2001.
"""

from .balance import *
from .cielab import *
from .cieluv import *
from .ciexyz import *
from .grayscale import *
from .hed import *
from .hsi import *
from .hsl import *
from .hsv import *
from .hwb import *
from .lms import *
from .rgb import *
from .yuv import *
