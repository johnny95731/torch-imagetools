__all__ = [
    'rgb_to_hed',
    'hed_to_rgb',
]

import torch

from ..utils.math import matrix_transform


# Haematoxylin-Eosin-DAB colorspace
# From original Ruifrok's paper:
# A. C. Ruifrok and D. A. Johnston,
# "Quantification of histochemical staining by color deconvolution,"
# Analytical and quantitative cytology and histology / the International
# Academy of Cytology [and] American Society of Cytology,
# vol. 23, no. 4, pp. 291-9, Aug. 2001.
_MAT_HED_TO_RGB = torch.tensor(
    # (
    #     (0.65, 0.70, 0.29),
    #     (0.07 / 1.17, 0.99 / 1.17, 0.11 / 1.17),
    #     (0.27 / 1.62, 0.57 / 1.62, 0.78 / 1.62),
    # ),
    (  # Normalize to sum of rows to 1
        (0.396342, 0.426829, 0.176829),
        (0.059829, 0.846154, 0.094017),
        (0.166667, 0.351852, 0.481481),
    ),
    dtype=torch.float32,
)
_MAT_RGB_TO_HED = torch.linalg.inv(_MAT_HED_TO_RGB)


def rgb_to_hed(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to HED space [1].

    Parameters
    ----------
    rgb : torch.Tensor
        An image in RGB space with shape `(*, 3, H, W)`.

    Returns
    -------
    torch.Tensor
        An image in HED space. The the coefficients of the transformation
        matrix is scaled. Hence the maximum is the same as the input.

    References
    ----------
    [1] A. C. Ruifrok and D. A. Johnston,
        "Quantification of histochemical staining by color deconvolution,"
        Analytical and quantitative cytology and histology / the International
        Academy of Cytology [and] American Society of Cytology,
        vol. 23, no. 4, pp. 291-9, Aug. 2001.
    """
    matrix = (
        _MAT_HED_TO_RGB
        if rgb.device == 'cpu'
        else _MAT_HED_TO_RGB.to(rgb.device)
    )
    hed = matrix_transform(rgb, matrix)
    return hed


def hed_to_rgb(hed: torch.Tensor) -> torch.Tensor:
    """Converts an image from HED space [1] to RGB space.

    Parameters
    ----------
    hex : torch.Tensor
        An image in HED space with shape `(*, 3, H, W)`.

    Returns
    -------
    torch.Tensor
        An image in RGB space in the range of [0, 1] with shape `(*, 3, H, W)`.

    References
    ----------
    [1] A. C. Ruifrok and D. A. Johnston,
        "Quantification of histochemical staining by color deconvolution,"
        Analytical and quantitative cytology and histology / the International
        Academy of Cytology [and] American Society of Cytology,
        vol. 23, no. 4, pp. 291-9, Aug. 2001.
    """
    matrix = (
        _MAT_RGB_TO_HED
        if hed.device == 'cpu'
        else _MAT_RGB_TO_HED.to(hed.device)
    )
    rgb = matrix_transform(hed, matrix).clip(0.0, 1.0)
    return rgb
