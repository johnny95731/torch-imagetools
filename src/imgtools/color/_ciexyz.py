__all__ = [
    'get_white_point_names',
    'get_white_point',
    'get_rgb_names',
    'get_rgb_model',
    'get_rgb_to_xyz_matrix',
    'get_xyz_to_rgb_matrix',
    'rgb_to_xyz',
    'xyz_to_rgb',
    'normalize_xyz',
    'unnormalize_xyz',
]

import numpy as np
import torch

from ..utils.helpers import align_device_type, to_channel_coeff
from ..utils.math import matrix_transform
from ._rgb import gammaize_rgb, linearize_rgb


def get_white_point_names() -> tuple[str, ...]:
    """Support standard illuminants.

    Returns
    -------
    tuple[str, ...]
        Names of standard illuminants.
    """
    res = (
        'A',
        'B',
        'C',
        'D50',
        'D55',
        'D65',
        'D75',
        'D93',
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
    )
    return res


def get_white_point(
    white: str,
    obs: str | int = 10,
):
    """Returns a dictionary that contains name, x, y, CCT, and degree of
    observer of the standard illuminant.

    Parameters
    ----------
    white : StandardIlluminants
        Name of the standard illuminant, such as D65, D50, and F1-F12. The
        input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of observer. None and invalid value will be regarded as
        10 degree. The input is case-insensitive.

    Returns
    -------
    WhitePoint
        Dict with the folloing keys:
        - name: The name of the standard illuminant.
        - xy: The (x, y) values in xyY space.
        - cct: The correlated color temperature.
        - obs: The degree of observer.
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
    white_point = {
        'name': white,
        'xy': (x, y),
        'cct': cct,
        'obs': obs,
    }
    return white_point


def get_rgb_names() -> tuple[str, ...]:
    """Names of RGB models, such as srgb or displayp3.

    Returns
    -------
    tuple[str, ...]
        Names of RGB models.
    """
    res = (
        'adobergb',
        'ciergb',
        'displayp3',
        'prophotorgb',
        'rec2020',
        'srgb',
        'widegamut',
    )
    return res


def get_rgb_model(rgb_spec: str):
    """Returns a dict containing informations about a RGB specification.

    Parameters
    ----------
    rgb_spec : RGBSpec
        Name of the RGB specification, such as sRGB, displayP3, etc.
        The input is case-insensitive.

    Returns
    -------
    RGBModel
        Dict with the folloing keys:
        - name: The name of the color space.
        - r: (x, y) value of red in xyY space.
        - g: (x, y) value of green in xyY space.
        - b: (x, y) value of blue in xyY space.
        - w: The name of the white point.
    """
    spaces = {
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
    rgb_spec: str,
    white: str,
    obs: int | str = 10,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
) -> torch.Tensor:
    """Evaluate the matrix for converting RGB to CIE XYZ by the given RGB
    model, white point, and degree of observer.

    Parameters
    ----------
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        White point. The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        The degree of oberver
    dtype : torch.dtype, default=torch.float32
        The data type of the tensor. Must be a floating point.
    device: torch.device | str, default='cpu'
        The device of the tensor.

    Returns
    -------
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.

    Raises
    ------
    ValueError
        When `dtype` is not a floating point.
    """
    if not dtype.is_floating_point:
        raise ValueError(f'dtype must be floating point.')

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
        (
            (Xr, Xg, Xb),
            (1.0, 1.0, 1.0),
            (Zr, Zg, Zb),
        ),
        dtype=np.float64,
    )
    s_mat = np.linalg.inv(matrix)
    s_vec = np.dot(s_mat, (wx / wy, 1.0, (1 - wx - wy) / wy))  # type: np.ndarray

    np.multiply(matrix, s_vec, out=matrix)
    matrix = torch.as_tensor(matrix, dtype=dtype, device=device)
    return matrix


def get_xyz_to_rgb_matrix(
    rgb_spec: str,
    white: str,
    obs: str | int = 10,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
) -> torch.Tensor:
    """Evaluate the matrix for converting CIE XYZ to RGB by the given RGB
    model, white point, and degree of observer.

    Parameters
    ----------
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        White point. The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        The degree of oberver.
    dtype : torch.dtype, default=torch.float32
        The data type of the tensor. Must be a floating point.
    device: torch.device | str, default='cpu'
        The device of the tensor.

    Returns
    -------
    mat : torch.Tensor
        A transformation matrix used to convert CIE XYZ to RGB.

    Raises
    ------
    ValueError
        When `dtype` is not a floating point.
    """
    if not dtype.is_floating_point:
        raise ValueError(f'dtype must be floating point.')

    _dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs, _dtype, device)
    matrix = matrix.inverse()
    if _dtype is not dtype:
        matrix = matrix.type(dtype)
    return matrix


def rgb_to_xyz(
    rgb: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
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
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    xyz : torch.Tensor
        An image in CIE XYZ space with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert RGB to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    rgb = linearize_rgb(rgb, rgb_spec)
    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)

    xyz = matrix_transform(rgb, matrix)
    if ret_matrix:
        return xyz, matrix
    return xyz


def xyz_to_rgb(
    xyz: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to RGB space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        Reference white point for the rgb to xyz conversion.
        The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        Degree of the standard observer (2° or 10°).
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert CIE XYZ to RGB.
        `mat` is returned only if `ret_matrix` is true.
    """
    matrix = get_xyz_to_rgb_matrix(rgb_spec, white, obs)

    nonlinear_rgb = matrix_transform(xyz, matrix)
    nonlinear_rgb = nonlinear_rgb.clip(0.0, 1.0)
    linear = gammaize_rgb(nonlinear_rgb, rgb_spec)
    if ret_matrix:
        return linear, matrix
    return linear


def normalize_xyz(
    xyz: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
) -> torch.Tensor:
    """Normalize the image in CIE XYZ to [0, 1] by evaluting
    xyz / rgb_to_xyz(white_rgb) and clips to [0, 1].

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        White point. The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        The degree of oberver.
    """
    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)
    max_ = matrix.sum(dim=1)
    max_ = align_device_type(max_, xyz)
    max_ = to_channel_coeff(max_, 3)

    res = xyz / max_
    return res


def unnormalize_xyz(
    xyz: torch.Tensor,
    rgb_spec: str = 'srgb',
    white: str = 'D65',
    obs: str | int = 10,
) -> torch.Tensor:
    """The inverse function of normalize_xyz.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    rgb_spec : RGBSpec, default='srgb'
        The name of RGB specification. The argument is case-insensitive.
    white : StandardIlluminants, default='D65'
        White point. The input is case-insensitive.
    obs : {2, '2', 10, '10'}, default=10
        The degree of oberver.
    inplace : bool, default=False
        In-place operation or not.
    """
    matrix = get_rgb_to_xyz_matrix(rgb_spec, white, obs)
    max_ = matrix.sum(dim=1)
    max_ = align_device_type(max_, xyz)
    max_ = to_channel_coeff(max_, 3)

    res = xyz * max_
    return res
