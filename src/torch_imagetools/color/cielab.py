import torch
import numpy as np

from ..utils.helpers import tensorlize
from .ciexyz import rgb_to_xyz, xyz_to_rgb

_6_29 = 6 / 29  # threshold for _lab_helper_inv
_6_29_pow3 = _6_29**3  # threshold for _lab_helper
_scaling = 1 / (3 * _6_29**2)  # = 1 / (3 * _6_29**2)
_bias = 4 / 29  # = 16 / 116


def _lab_helper(value: torch.Tensor):
    """Function that be used in the transformation from CIE XYZ to CIE LAB and
    to CIE LUV.
    The function maps [0, 1] into [4/29, 1] and is continuous.

    Parameters
    ----------
    value : torch.Tensor
        _description_
    """
    output = torch.empty_like(value)
    mask_gt = value > _6_29_pow3
    output[mask_gt] = torch.pow(value[mask_gt], 1 / 3)

    mask_leq = torch.bitwise_not(mask_gt)
    lower = value[mask_leq] * _scaling
    torch.add(lower, _bias, out=lower)
    output[mask_leq] = lower

    return output


def _lab_helper_inv(value: torch.Tensor):
    """Function that be used in the transformation from CIE LAB to CIE XYZ and
    from CIE LUV to CIE XYZ.
    The function maps [4/29, 1] into [0, 1].

    Parameters
    ----------
    value : torch.Tensor
        _description_
    """
    output = torch.empty_like(value)
    mask_gt = value > _6_29
    output[mask_gt] = torch.pow(value[mask_gt], 3)

    mask_leq = torch.bitwise_not(mask_gt)
    lower = value[mask_leq] - _bias
    torch.mul(lower, 1 / _scaling, out=lower)
    output[mask_leq] = lower
    return output


def xyz_to_lab(xyz: np.ndarray | torch.Tensor) -> torch.Tensor:
    xyz = tensorlize(xyz)

    max = rgb_to_xyz.max
    x: torch.Tensor = xyz[..., 0, :, :] * (1 / max[0])
    y: torch.Tensor = xyz[..., 1, :, :] * (1 / max[1])
    z: torch.Tensor = xyz[..., 2, :, :] * (1 / max[2])
    fx = _lab_helper(x)
    fy = _lab_helper(y)
    fz = _lab_helper(z)
    l = 1.16 * fy - 0.16
    a = 5.0 * (fx - fy)
    b = 2.0 * (fy - fz)
    lab = torch.stack((l, a, b), dim=-3)
    return lab


def lab_to_xyz(lab: np.ndarray | torch.Tensor) -> torch.Tensor:
    lab = tensorlize(lab)

    max = rgb_to_xyz.max
    l: torch.Tensor = lab[..., 0, :, :]
    a: torch.Tensor = lab[..., 1, :, :]
    b: torch.Tensor = lab[..., 2, :, :]

    torch.add(l, 0.16, out=l)
    torch.mul(l, 1 / 1.16, out=l)
    x = max[0] * _lab_helper_inv(l + a * 0.2)
    y = max[1] * _lab_helper_inv(l)  # inplace operation
    z = max[2] * _lab_helper_inv(l - b * 0.5)

    xyz = torch.stack((x, y, z), dim=-3)
    return xyz


def rgb_to_lab(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab


def lab_to_rgb(lab: np.ndarray | torch.Tensor) -> torch.Tensor:
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 10

    xyz = rgb_to_xyz(img)
    lab = xyz_to_lab(xyz)
    ret = lab_to_xyz(lab)

    d = torch.abs(ret - xyz)
    print(torch.max(d))
    print(timeit('xyz_to_lab(xyz)', number=num, globals=locals()))
    print(timeit('lab_to_xyz(lab)', number=num, globals=locals()))

    lab = rgb_to_lab(img)
    ret = lab_to_rgb(lab)

    d = torch.abs(ret - img)
    print(torch.max(d))
    print(timeit('rgb_to_lab(img)', number=num, globals=locals()))
    print(timeit('lab_to_rgb(lab)', number=num, globals=locals()))
