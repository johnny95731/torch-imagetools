__all__ = [
    'StandardIlluminants',
    'RGBModel',
    'WhitePoint',
    'get_white_point',
    'get_rgb_model',
    'get_rgb_to_xyz_matrix',
    'get_xyz_to_rgb_matrix',
    'rgb_to_xyz',
    'xyz_to_rgb',
    'normalize_xyz',
    'unnormalize_xyz',
]

from typing import Literal, TypedDict, overload

import numpy as np
import torch

from .rgb import RGBSpec, gammaize_rgb, linearize_rgb
from ..utils.helpers import matrix_transform


StandardIlluminants = Literal[
    'A',
    'B',
    'C',
    'D50',
    'D55',
    'D65',
    'D75',
    'E',
    'F1',
    'F2',
    'F3',
    'F4',
    'F5',
    'F6',
    'F7',
    'F8',
    'F9',
    'F10',
    'F11',
    'F12',
]


class RGBModel(TypedDict):
    """Informations about the RGB specification."""

    name: str
    """The name of the color space."""
    r: tuple[float, float]
    """(x, y) value of red in xyY space."""
    g: tuple[float, float]
    """(x, y) value of green in xyY space."""
    b: tuple[float, float]
    """(x, y) value of blue in xyY space."""
    w: StandardIlluminants
    """The name of the white point."""


class WhitePoint(TypedDict):
    """Informations about the white point / standard illuminant."""

    name: StandardIlluminants
    """The name of the standard illuminant."""
    xy: float
    """The (x, y) values in xyY space."""
    cct: int
    """The correlated color temperature."""
    obs: Literal[2, 10]
    """The degree of observer."""


def get_white_point(
    white: StandardIlluminants,
    obs: Literal[2, '2', 10, '10'] = 10,
) -> WhitePoint:
    """Returns the name, x, y, CCT, and degree of observer of the standard
    illuminant.

    Parameters
    ----------
    white : Literal[W]
        Name of the standard illuminant, such as D65, D50, and F1-F12. The
        input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        Degree of observer, by default None. None and invalid value will be
        regarded as 10 degree. The input is case-insensitive.
    """
    obs_deg_2 = str(obs) == '2'
    obs = 2 if obs_deg_2 else 10  # degree of oberver.
    if obs == 2:
        data = {
            'A': (0.44757, 0.40745, 2856),
            'B': (0.34842, 0.35161, 4874),
            'C': (0.31006, 0.31616, 6774),
            'D50': (0.34567, 0.35850, 5003),
            'D55': (0.33242, 0.34743, 5503),
            'D65': (0.31271, 0.32902, 6504),
            'D75': (0.29902, 0.31485, 7504),
            'D93': (0.28315, 0.29711, 9305),
            'E': (1 / 3, 1 / 3, 5454),
            'F1': (0.31310, 0.33727, 6430),
            'F2': (0.37208, 0.37529, 4230),
            'F3': (0.40910, 0.39430, 3450),
            'F4': (0.44018, 0.40329, 2940),
            'F5': (0.31379, 0.34531, 6350),
            'F6': (0.37790, 0.38835, 4150),
            'F7': (0.31292, 0.32933, 6500),
            'F8': (0.34588, 0.35875, 5000),
            'F9': (0.37417, 0.37281, 4150),
            'F10': (0.34609, 0.35986, 5000),
            'F11': (0.38052, 0.37713, 4000),
            'F12': (0.43695, 0.40441, 3000),
        }
    elif obs == 10:
        data = {
            'A': (0.45117, 0.40594, 2856),
            'B': (0.34980, 0.35270, 4874),
            'C': (0.31039, 0.31905, 6774),
            'D50': (0.34773, 0.35952, 5003),
            'D55': (0.33411, 0.34877, 5503),
            'D65': (0.31382, 0.33100, 6504),
            'D75': (0.29968, 0.31740, 7504),
            'D93': (0.28327, 0.30043, 9305),
            'E': (1 / 3, 1 / 3, 5454),
            'F1': (0.31811, 0.33559, 6430),
            'F2': (0.37925, 0.36733, 4230),
            'F3': (0.41761, 0.38324, 3450),
            'F4': (0.44920, 0.39074, 2940),
            'F5': (0.31975, 0.34246, 6350),
            'F6': (0.38660, 0.37847, 4150),
            'F7': (0.31569, 0.32960, 6500),
            'F8': (0.34902, 0.35939, 5000),
            'F9': (0.37829, 0.37045, 4150),
            'F10': (0.35090, 0.35444, 5000),
            'F11': (0.38541, 0.37123, 4000),
            'F12': (0.44256, 0.39717, 3000),
        }

    key = white.upper()
    if key not in data:
        key = 'D65'

    x, y, cct = data[key]
    white_point: WhitePoint = {
        'name': white,
        'xy': (x, y),
        'cct': cct,
        'obs': obs,
    }
    return white_point


def get_rgb_model(rgb_spec: RGBSpec) -> RGBModel:
    """Returns a dict containing informations about a RGB specification.

    Parameters
    ----------
    rgb_spec : RGBSpec
        Name of the specification, such as sRGB, displayP3, and adobeRGB....
        The input is case-insensitive.
    """
    spaces: dict[str, RGBModel] = {
        'adobergb': {
            'r': (0.6400, 0.3300),
            'g': (0.2100, 0.7100),
            'b': (0.1500, 0.0600),
            'w': 'D65',
        },
        'ciergb': {
            'r': (0.73474284, 0.26525716),
            'g': (0.27377903, 0.7174777),
            'b': (0.16655563, 0.00891073),
            'w': 'E',
        },
        'displayp3': {
            'r': (0.6800, 0.3200),
            'g': (0.2650, 0.6900),
            'b': (0.1500, 0.0600),
            'w': 'D65',
        },
        'prophotorgb': {
            'r': (0.734699, 0.265301),
            'g': (0.159597, 0.840403),
            'b': (0.036598, 0.000105),
            'w': 'D50',
        },
        'rec2020': {
            'r': (0.7080, 0.2920),
            'g': (0.1700, 0.7970),
            'b': (0.1310, 0.0460),
            'w': 'D65',
        },
        'srgb': {
            'r': (0.6400, 0.3300),
            'g': (0.3000, 0.6000),
            'b': (0.1500, 0.0600),
            'w': 'D65',
        },
        'widegamut': {
            'r': (0.7347, 0.2653),
            'g': (0.1152, 0.8264),
            'b': (0.1566, 0.0177),
            'w': 'D50',
        },
    }

    rgb_spec = rgb_spec.lower()
    if rgb_spec not in spaces:
        rgb_spec = 'srgb'
    rgb = spaces[rgb_spec]
    rgb['name'] = rgb_spec
    return rgb


def get_rgb_to_xyz_matrix(
    rgb_spec: RGBSpec,
    white: StandardIlluminants,
    obs: Literal[2, '2', 10, '10'] = 10,
) -> torch.Tensor:
    """Evaluate the matrix for converting RGB to CIE XYZ by the given RGB
    model, white point, and degree of observer.

    Parameters
    ----------
    rgb_spec : RGBSpec, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    """
    white_ = get_white_point(white, obs)
    rgb_ = get_rgb_model(rgb_spec)
    rx, ry = rgb_['r']
    gx, gy = rgb_['g']
    bx, by = rgb_['b']
    wx, wy = white_['xy']

    Xr = rx / ry  # noqa
    Zr = (1.0 - rx - ry) / ry  # noqa
    Xg = gx / gy  # noqa
    Zg = (1.0 - gx - gy) / gy  # noqa
    Xb = bx / by  # noqa
    Zb = (1.0 - bx - by) / by  # noqa

    matrix = np.array(
        [
            [Xr, Xg, Xb],
            [1.0, 1.0, 1.0],
            [Zr, Zg, Zb],
        ],
        dtype=np.float32,
    )
    s_mat = np.linalg.inv(matrix)
    s_vec = np.dot(s_mat, [wx / wy, 1.0, (1 - wx - wy) / wy])  # type: np.ndarray

    np.multiply(matrix, s_vec, out=matrix)
    matrix = torch.from_numpy(matrix)
    return matrix


def get_xyz_to_rgb_matrix(
    rgb_spec: RGBSpec,
    white: StandardIlluminants,
    obs: Literal[2, 10] = 10,
) -> torch.Tensor:
    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)
    matrix = torch.linalg.inv(matrix)
    return matrix


@overload
def rgb_to_xyz(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def rgb_to_xyz(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def rgb_to_xyz(
    rgb: torch.Tensor,
    matrix: torch.Tensor,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def rgb_to_xyz(
    rgb: torch.Tensor,
    matrix: torch.Tensor,
    *,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def rgb_to_xyz(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from RGB space to CIE XYZ space.

    The input is assumed to be in the range of [0, 1]. If rgb_spec is a
    tensor, then the input rgb is assumed to be linear RGB.

    The minimum of channels of XYZ is 0, and the maxima of channels depend
    on the RGB model and the white point (Y channel always 1).

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type. If rgb_spec is a
        tensor, then the input rgb is assumed to be linear RGB.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    ret_matrix : bool, optional
        If True, returns image and conversion matrix (rgb -> xyz).
        By default False.

    Returns
    -------
    torch.Tensor
        An image in CIE XYZ space with the shape (*, 3, H, W). If ret_matrix
        is True, returns image and the transformation matrix.
    """
    if torch.is_tensor(rgb_spec):
        matrix = rgb_spec
    else:
        rgb = linearize_rgb(rgb, rgb_spec)
        matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)

    xyz = matrix_transform(rgb, matrix)
    if ret_matrix:
        return xyz, matrix
    return xyz


@overload
def xyz_to_rgb(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def xyz_to_rgb(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def xyz_to_rgb(
    xyz: torch.Tensor,
    matrix: torch.Tensor,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def xyz_to_rgb(
    xyz: torch.Tensor,
    matrix: torch.Tensor,
    *,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def xyz_to_rgb(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to RGB space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type. If rgb_spec is a
        tensor, then the output is a linear RGB.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    ret_matrix : bool, optional
        If True, returns image and transformation matrix (xyz -> rgb).
        By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An RGB image in the range of [0, 1] with the shape (*, 3, H, W). If
        rgb_spec is a tensor, then the image is in linear RGB space. If
        ret_matrix is True, returns image and the transformation matrix.
    """
    matrix = (
        get_xyz_to_rgb_matrix(rgb_spec, white, obs)
        if not torch.is_tensor(rgb_spec)
        else rgb_spec
    )

    linear = matrix_transform(xyz, matrix)
    linear.clip_(0.0, 1.0)
    if not torch.is_tensor(rgb_spec):
        gammaize_rgb(linear, rgb_spec, out=linear)
    if ret_matrix:
        return linear, matrix
    return linear


def normalize_xyz(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    inplace: bool = False,
) -> torch.Tensor:
    """Normalize the image in CIE XYZ to [0, 1] by evaluting
    xyz / rgb_to_xyz(white_rgb) and clips to [0, 1].

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    inplace : bool, optional
        In-place operation or not, by default False.
    """
    matrix = (
        get_rgb_to_xyz_matrix(rgb_spec, white, obs)
        if not torch.is_tensor(rgb_spec)
        else rgb_spec
    )
    max_ = matrix.sum(dim=1)

    out = xyz if inplace else xyz.clone()
    out[..., 0, :, :].mul_(1 / max_[0])
    # xyz[..., 1, :, :].mul_(1 / max[1])  # Ignore this line since max[1] = 1
    out[..., 2, :, :].mul_(1 / max_[2])
    out.clip_(0.0, 1.0)
    return out


def unnormalize_xyz(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    inplace: bool = False,
) -> torch.Tensor:
    """The inverse function of normalize_xyz.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec | torch.Tensor, optional
        The RGB specification or a conversion matrix, by default 'srgb'.
        The input is case-insensitive if it is str type.
    white : STANDARD_ILLUMINANTS, optional
        White point, by default 'D65'. The input is case-insensitive.
    obs : Literal[2, '2', 10, '10'], optional
        The degree of oberver, by default 10.
    inplace : bool, optional
        In-place operation or not, by default False.
    """
    matrix = (
        get_rgb_to_xyz_matrix(rgb_spec, white, obs)
        if not torch.is_tensor(rgb_spec)
        else rgb_spec
    )
    max_ = matrix.sum(dim=1)

    out = xyz if inplace else xyz.clone()
    out[..., 0, :, :].mul_(max_[0])
    # xyz[..., 1, :, :].mul_(max[1])  # Ignore this line since max[1] = 1
    out[..., 2, :, :].mul_(max_[2])
    return out


if __name__ == '__main__':
    from timeit import timeit

    img = torch.randint(0, 256, (8, 3, 512, 512), dtype=torch.float32).mul_(
        1 / 255
    )
    num = 30

    spec: RGBSpec = 'adobergb'
    white: StandardIlluminants = 'D65'
    obs = 2
    mat1 = get_rgb_to_xyz_matrix(spec, white, obs)
    print(mat1.sum(dim=1))
    print(
        timeit(
            "get_rgb_to_xyz_matrix('srgb', 'D65')",
            number=10000,
            globals=locals(),
        )
    )

    xyz = rgb_to_xyz(img, spec, white, obs)
    ret = xyz_to_rgb(xyz, spec, white, obs)

    d = torch.abs(ret - img)
    print('Error:', torch.max(d).item())
    # print(timeit('rgb_to_xyz(img)', number=num, globals=locals()))
    # print(timeit('xyz_to_rgb(xyz)', number=num, globals=locals()))

    # xyz_n = normalize_xyz(xyz, spec, white, obs)
    # xyz_un = unnormalize_xyz(xyz_n, spec, white, obs)

    # d = torch.abs(xyz - xyz_un)
    # print('Error:', torch.max(d).item())
    # print(
    #     timeit(
    #         'unnormalize_xyz(normalize_xyz(xyz))',
    #         number=num,
    #         globals=locals(),
    #     )
    # )
    # print(
    #     timeit(
    #         'unnormalize_xyz(normalize_xyz(xyz, inplace=True), inplace=True)',
    #         number=num,
    #         globals=locals(),
    #     )
    # )

    w = [
        'A',
        'B',
        'C',
        'D50',
        'D55',
        'D65',
        'D75',
        'E',
        'F1',
        'F2',
        'F3',
        'F4',
        'F5',
        'F6',
        'F7',
        'F8',
        'F9',
        'F10',
        'F11',
        'F12',
    ]
    rgbs = [
        'srgb',
        'adobergb',
        'prophotorgb',
        'rec2020',
        'displayp3',
        'widegamut',
        'ciergb',
    ]
    # for rgb in rgbs:
    #     for white in w:
    #         xyz = rgb_to_xyz(img.clone(), rgb, white, 2)
    #         ret = xyz_to_rgb(xyz, rgb, white, 2)
    #         d2 = torch.abs(ret - img)

    #         xyz = rgb_to_xyz(img.clone(), rgb, white, 10)
    #         ret = xyz_to_rgb(xyz, rgb, white, 10)
    #         d10 = torch.abs(ret - img)
    #         title = f'{rgb}/{white}'
    #         print(f'{title:<16}', torch.max(d2).item(), torch.max(d10).item())
