import torch
import numpy as np

from ..utils.helpers import tensorlize
from .ciexyz import rgb_to_xyz, xyz_to_rgb

_6_29 = 6 / 29  # threshold for _lab_helper_inv
_6_29_pow3 = _6_29**3  # threshold for _lab_helper
_scaling = 0.01 / ((_6_29 / 2) ** 3)  # = (29 / 3)**3 / 100


def _luv_helper(value: torch.Tensor):
    """Function that be used in the transformation from CIE XYZ to CIE LUV.

    This is an in-place operation.

    Parameters
    ----------
    value : torch.Tensor
        _description_
    """
    output = torch.empty_like(value)
    mask_gt = value > _6_29_pow3
    output[mask_gt] = 1.16 * torch.pow(value[mask_gt], 1 / 3) - 0.16

    mask_leq = torch.bitwise_not(mask_gt)
    output[mask_leq] = value[mask_leq] * _scaling

    return output


def _luv_helper_inv(value: torch.Tensor):
    """Function that be used in the transformation from CIE LUV to CIE XYZ.

    Parameters
    ----------
    value : torch.Tensor
        _description_
    """
    output = torch.empty_like(value)
    mask_gt = value > 0.08
    higher = value[mask_gt] + 0.16
    higher *= 1 / 1.16
    higher = higher**3
    output[mask_gt] = higher

    mask_leq = torch.bitwise_not(mask_gt)
    output[mask_leq] = value[mask_leq] * (1 / _scaling)
    return output


_U_CONST = 0.19784668703068656
_V_CONST = 0.4683377651962596


def xyz_to_luv(xyz: np.ndarray | torch.Tensor) -> torch.Tensor:
    xyz = tensorlize(xyz)

    max = rgb_to_xyz.max
    x: torch.Tensor = xyz[..., 0, :, :]
    y: torch.Tensor = xyz[..., 1, :, :]
    z: torch.Tensor = xyz[..., 2, :, :]

    coeff = 4.0 / (x + 15 * y + 3 * z)
    u_prime = x * coeff
    v_prime = 2.25 * y * coeff

    l = _luv_helper(y / max[1])  # inplace operation
    l_13 = 0.13 * l
    u = l_13 * (u_prime - _U_CONST)
    v = l_13 * (v_prime - _V_CONST)
    luv = torch.stack((l, u, v), dim=-3)
    return luv


def luv_to_xyz(luv: np.ndarray | torch.Tensor) -> torch.Tensor:
    luv = tensorlize(luv)

    max = rgb_to_xyz.max
    l: torch.Tensor = luv[..., 0, :, :]
    u: torch.Tensor = luv[..., 1, :, :]
    v: torch.Tensor = luv[..., 2, :, :]

    l_13 = 0.13 * l
    u_prime = (u / l_13) + _U_CONST
    v_prime = (v / l_13) + _V_CONST

    y = max[1] * _luv_helper_inv(l)  # in-place operation
    x = 2.25 * u_prime / v_prime * y
    z = (3.0 - 0.75 * u_prime - 5.0 * v_prime) / v_prime * y

    xyz = torch.stack((x, y, z), dim=-3)
    return xyz


def rgb_to_luv(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    xyz = rgb_to_xyz(rgb)
    luv = xyz_to_luv(xyz)
    return luv


def luv_to_rgb(luv: np.ndarray | torch.Tensor) -> torch.Tensor:
    xyz = luv_to_xyz(luv)
    rgb = xyz_to_rgb(xyz)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 10

    xyz = rgb_to_xyz(img)
    luv = xyz_to_luv(xyz)
    ret = luv_to_xyz(luv)

    d = torch.abs(ret - xyz)
    print(torch.max(d))
    print(timeit('xyz_to_luv(xyz)', number=num, globals=locals()))
    print(timeit('luv_to_xyz(luv)', number=num, globals=locals()))

    luv = rgb_to_luv(img)
    ret = luv_to_rgb(luv)

    print(torch.max(torch.abs(ret - img)))
    print(timeit('rgb_to_luv(img)', number=num, globals=locals()))
    print(timeit('luv_to_rgb(luv)', number=num, globals=locals()))
