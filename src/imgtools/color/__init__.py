"""Transformations about color models.

.. currentmodule:: imgtools.color

Color Space Transformations
---------------------------

Provides color space transformations between color spaces. Except LMS space,
all the color spaces can be converted from/to RGB color space.

- RGB based color spaces : RGB, YUV, HSV, HSL, HSI, HWB
    * RGB : Red Green Blue<br>
        Most color space can be transformed from this space.
        Supports the linearize and the gamma-ize of several RGB models: sRGB,
        Display P3, Adobe RGB, wide-gamut, ProPhoto RGB, REC. 2020.
    * YUV : Luminance u v<br>
        The color space used in the television.
        The coefficients of Y channel follows the SDTV standard (BT.470).
        And the U, V channels are scaled to the range of [-0.5, 0.5].
    * HSV : Hue Saturation Value<br>
        A cylindrical-coordinate representation in an RGB color model.
        The resulting solid is a cone.
    * HSL : Hue Saturation Lightness<br>
        A cylindrical-coordinate representation in an RGB color model.
        The resulting solid is a bicone.
    * HSI : Hue Saturation Intensity<br>
        A cylindrical-coordinate representation in an RGB color model.
        The resulting solid is a bicone.
    * HWB : Hue Whitness Blackness<br>
        A cylindrical-coordinate of RGB color model.
    * Gray : Grayscale<br>
        Same as the Y channel of YUV color space.

- CIE based color spaces: CIE XYZ, LMS, CIE LAB, CIE LUV
    * CIE XYZ : X Y Z<br>
        CIE 1931 color space, which define the relationship between the
        visible spectrum and human color vision.
    * LMS : long medium short<br>
        The cone response space, which is based on the human visual system.
        This space can only be converted from/to CIE XYZ space.
    * CIE LAB : Lightness a* b*<br>
        L*a*b* color space, which was intended as a perceptually uniform space.
        a* represents green-red opponent colors, and b* represents
        blue-yellow opponent colors.
    * CIE LUV : Lightness u* v*<br>
        L*u*v* color space, which attempted perceptual uniformity, like L*a*b*.

- Other color spaces: HED
    * HED : Haematoxylin Eosin DAB
            Haematoxylin-Eosin-DAB. See [Ruifrok] for details.

Reference
---------

[Ruifrok] : A. C. Ruifrok and D. A. Johnston, "Quantification of
    histochemical staining by color deconvolution.," Analytical and
    quantitative cytology and histology / the International Academy of
    Cytology [and] American Society of Cytology,
    vol. 23, no. 4, pp. 291-9, Aug. 2001.
"""

__all__ = [
    'gammaize_adobe_rgb',
    'gammaize_prophoto_rgb',
    'gammaize_rec2020',
    'gammaize_rgb',
    'gammaize_srgb',
    'get_chromatic_adaptation',
    'get_lms_to_xyz_matrix',
    'get_rgb_model',
    'get_rgb_names',
    'get_rgb_to_xyz_matrix',
    'get_white_point',
    'get_white_point_names',
    'get_xyz_to_lms_matrix',
    'get_xyz_to_rgb_matrix',
    'gray_to_rgb',
    'hed_to_rgb',
    'hsi_to_rgb',
    'hsl_to_rgb',
    'hsv_to_rgb',
    'hwb_to_rgb',
    'lab_to_rgb',
    'lab_to_xyz',
    'linearize_adobe_rgb',
    'linearize_prophoto_rgb',
    'linearize_rec2020',
    'linearize_rgb',
    'linearize_srgb',
    'lms_to_xyz',
    'luv_to_rgb',
    'luv_to_xyz',
    'normalize_xyz',
    'rgb_to_gray',
    'rgb_to_hed',
    'rgb_to_hsi',
    'rgb_to_hsl',
    'rgb_to_hsv',
    'rgb_to_hwb',
    'rgb_to_lab',
    'rgb_to_luv',
    'rgb_to_xyz',
    'rgb_to_yuv',
    'unnormalize_xyz',
    'xyz_to_lab',
    'xyz_to_lms',
    'xyz_to_luv',
    'xyz_to_rgb',
    'yuv_to_rgb',
]

from ._cielab import (
    lab_to_rgb,
    lab_to_xyz,
    rgb_to_lab,
    xyz_to_lab,
)
from ._cieluv import (
    luv_to_rgb,
    luv_to_xyz,
    rgb_to_luv,
    xyz_to_luv,
)
from ._ciexyz import (
    get_rgb_model,
    get_rgb_names,
    get_rgb_to_xyz_matrix,
    get_white_point,
    get_white_point_names,
    get_xyz_to_rgb_matrix,
    normalize_xyz,
    rgb_to_xyz,
    unnormalize_xyz,
    xyz_to_rgb,
)
from ._grayscale import (
    gray_to_rgb,
    rgb_to_gray,
)
from ._hed import (
    hed_to_rgb,
    rgb_to_hed,
)
from ._hsi import (
    hsi_to_rgb,
    rgb_to_hsi,
)
from ._hsl import (
    hsl_to_rgb,
    rgb_to_hsl,
)
from ._hsv import (
    hsv_to_rgb,
    rgb_to_hsv,
)
from ._hwb import (
    hwb_to_rgb,
    rgb_to_hwb,
)
from ._lms import (
    get_chromatic_adaptation,
    get_lms_to_xyz_matrix,
    get_xyz_to_lms_matrix,
    lms_to_xyz,
    xyz_to_lms,
)
from ._rgb import (
    gammaize_adobe_rgb,
    gammaize_prophoto_rgb,
    gammaize_rec2020,
    gammaize_rgb,
    gammaize_srgb,
    linearize_adobe_rgb,
    linearize_prophoto_rgb,
    linearize_rec2020,
    linearize_rgb,
    linearize_srgb,
)
from ._yuv import (
    rgb_to_yuv,
    yuv_to_rgb,
)
