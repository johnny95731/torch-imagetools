import torch

from ..utils.helpers import matrix_transform


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
    #     (0.07 / 1.17, 0.99, 0.11),
    #     (0.27, 0.57, 0.78),
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
    """Converts an image from RGB space to HED space.

    For details about HED, see:
        A. C. Ruifrok and D. A. Johnston,
        "Quantification of histochemical staining by color deconvolution,"
        Analytical and quantitative cytology and histology / the International
        Academy of Cytology [and] American Society of Cytology,
        vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Parameters
    ----------
    rgb : torch.Tensor
        An image in RGB space with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An image in HED space. The the coefficients of the transformation
        matrix is scaled. Hence the maximum is the same as the input.
    """
    matrix = (
        _MAT_HED_TO_RGB
        if rgb.device == 'cpu'
        else _MAT_HED_TO_RGB.to(rgb.device)
    )
    hed = matrix_transform(rgb, matrix)
    return hed


def hed_to_rgb(hed: torch.Tensor) -> torch.Tensor:
    """Converts an image from HED space to RGB space.

    For details about HED, see:
        A. C. Ruifrok and D. A. Johnston,
        "Quantification of histochemical staining by color deconvolution,"
        Analytical and quantitative cytology and histology / the International
        Academy of Cytology [and] American Society of Cytology,
        vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Parameters
    ----------
    hex : torch.Tensor
        An image in HED space with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An image in RGB space.
    """
    matrix = (
        _MAT_RGB_TO_HED
        if hed.device == 'cpu'
        else _MAT_RGB_TO_HED.to(hed.device)
    )
    rgb = matrix_transform(hed, matrix).clip_(0.0, 1.0)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = torch.randint(0, 256, (24, 3, 512, 512)).type(torch.float32) / 255
    num = 50

    hed = rgb_to_hed(img)
    ret = hed_to_rgb(hed)

    d = torch.abs(ret - img)
    print(torch.max(d))
    print(timeit('rgb_to_hed(img)', number=num, globals=locals()))
    print(timeit('hed_to_rgb(hed)', number=num, globals=locals()))
