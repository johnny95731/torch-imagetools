import torch
import numpy as np

from .rgb import srgb_linear_to_srgb, srgb_to_srgb_linear
from ..utils.helpers import matrix_transform, matrix_transform_, tensorlize


# def get_xyz_matrix(): ...


def rgb_to_xyz(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a CIE XYZ image.

    The input is assumed to be in the range of [0, 1].
    The minimum of XYZ is 0, and the maximum of X, Y, Z channels are
    0.9505, 1.0000, 1.0888, respectively.

    The matrix is based on sRGB model and the reference white is D65.

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        _description_
    """
    rgb = tensorlize(rgb)
    rgb = srgb_to_srgb_linear(rgb)

    dtype = rgb.dtype if torch.is_floating_point(rgb) else torch.float32
    # fmt: off
    matrix = torch.tensor(
        (( 0.4124564, 0.2126729, 0.0193339),
         ( 0.3575761, 0.7151522, 0.1191920),
         ( 0.1804375, 0.0721750, 0.9503041)),
        dtype=dtype,
        device=rgb.device,
    )
    # fmt: on
    xyz = matrix_transform_(rgb, matrix)
    return xyz


rgb_to_xyz.max = (0.9505, 1.0000, 1.0888)


def xyz_to_rgb(
    xyz: np.ndarray | torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conver a CIE XYZ image to an RGB image.

    The input is assumed to be in the range of [0, 1].
    The minimum of XYZ is 0, and the maximum of X, Y, Z channels are
    0.9505, 1.0000, 1.0888, respectively.

    The matrix is based on sRGB model and the reference white is D65.

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        _description_
    """
    xyz = tensorlize(xyz)

    dtype = xyz.dtype if torch.is_floating_point(xyz) else torch.float32
    # fmt: off
    matrix = torch.tensor(
        (( 3.2404542,-0.9692660, 0.0556434),
         (-1.5371385, 1.8760108,-0.2040259),
         (-0.4985314, 0.0415560, 1.0572252)),
        dtype=dtype,
        device=xyz.device,
    )
    # fmt: on
    rgb = matrix_transform(xyz, matrix)
    rgb = srgb_linear_to_srgb(rgb, out=rgb)
    return rgb


def normalize_xyz(xyz: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Normalize the CIE XYZ image to [0, 1] by dividing the sum of
    coefficients of the matrix when converting from RGB.

    Parameters
    ----------
    xyz : torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        _description_
    """
    max = rgb_to_xyz.max

    out = xyz if inplace else xyz.clone()
    out[..., 0, :, :].mul_(1 / max[0])
    # xyz[..., 1, :, :].mul_(1 / max[1])  # Ignore this line since max[1] = 1
    out[..., 2, :, :].mul_(1 / max[2])
    torch.clip(out, 0.0, 1.0, out=out)
    return out


def unnormalize_xyz(xyz: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Un-normalize the CIE XYZ image by multiplying the sum of coefficients
    of the matrix when converting from RGB.

    Parameters
    ----------
    xyz : torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        _description_
    """
    max = rgb_to_xyz.max
    out = xyz if inplace else xyz.clone()
    out[..., 0, :, :].mul_(max[0])
    # xyz[..., 1, :, :].mul_(max[1])  # Ignore this line since max[1] = 1
    out[..., 2, :, :].mul_(max[2])
    return out


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (3, 256, 256)).type(torch.float32) / 255
    num = 70

    xyz = rgb_to_xyz(img)
    ret = xyz_to_rgb(xyz)

    # d = torch.abs(ret - img)
    # print(torch.max(d))
    # print(timeit('rgb_to_xyz(img)', number=num, globals=locals()))
    # print(timeit('xyz_to_rgb(xyz)', number=num, globals=locals()))

    xyz2 = xyz.clone()
    xyz_un = unnormalize_xyz(normalize_xyz(xyz))

    d = torch.abs(xyz - xyz_un)
    print(torch.max(d))
    print(
        timeit(
            'unnormalize_xyz(normalize_xyz(xyz))',
            number=num,
            globals=locals(),
        )
    )
    print(
        timeit(
            'unnormalize_xyz(normalize_xyz(xyz, True), True)',
            number=num,
            globals=locals(),
        )
    )
