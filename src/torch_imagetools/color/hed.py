import torch
import numpy as np

from ..utils.helpers import matrix_transform, tensorlize


# Haematoxylin-Eosin-DAB colorspace
# From original Ruifrok's paper: A. C. Ruifrok and D. A. Johnston,
# "Quantification of histochemical staining by color deconvolution,"
# Analytical and quantitative cytology and histology / the International
# Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4,
# pp. 291-9, Aug. 2001.
_MAT_HED_TO_RGB = torch.tensor(
    [
        [0.65, 0.07, 0.27],
        [0.70, 0.99, 0.57],
        [0.29, 0.11, 0.78],
    ],
    dtype=torch.float32,
)
_MAT_RGB_TO_HED = torch.linalg.inv(_MAT_HED_TO_RGB)


def rgb_to_hed(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    rgb = tensorlize(rgb)

    matrix = (
        _MAT_HED_TO_RGB
        if rgb.device == 'cpu'
        else _MAT_HED_TO_RGB.to(rgb.device)
    )
    hed = matrix_transform(rgb, matrix)
    return hed


def hed_to_rgb(hed: np.ndarray | torch.Tensor) -> torch.Tensor:
    hed = tensorlize(hed)

    matrix = (
        _MAT_RGB_TO_HED
        if hed.device == 'cpu'
        else _MAT_RGB_TO_HED.to(hed.device)
    )
    rgb = matrix_transform(hed, matrix)
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
