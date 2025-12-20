__all__ = [
    'get_chromatic_adaptation',
    'get_xyz_to_lms_matrix',
    'get_lms_to_xyz_matrix',
    'xyz_to_lms',
    'lms_to_xyz',
]

import torch

from ..utils.math import matrix_transform


def get_chromatic_adaptation() -> tuple[str, ...]:
    """Returns the support chromatic adaptation methods.

    Returns
    -------
    list[str]
        A list of chromatic adaptation names.
    """
    res = (
        'bradford',
        'cat02',
        'cat97s',
        'cam16',
        'hpe',
        'vonkries',
        'xyz',
    )
    return res


def get_xyz_to_lms_matrix(
    method: str = 'bradford',
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
):
    """Returns a transformation matrix for the conversion from CIE XYZ space
    to LMS space.

    Parameters
    ----------
    method : CATMethod, default='bradford'
        The method of conversion. The argument is case-insensitive. Check
        valid values by calling `get_chromatic_adaptation()`.
    dtype : torch.dtype, default=torch.float32
        The data type of the tensor. Must be a floating point.
    device: torch.device | str, default='cpu'
        The device of the tensor.

    Returns
    -------
    torch.Tensor
        The transformation matrix with shape (3, 3).

    Raises
    ------
    ValueError
        When `dtype` is not a floating point.
    """
    if not dtype.is_floating_point:
        raise ValueError(f'dtype must be floating point.')

    method = method.lower()
    data = {
        'bradford': (
            (0.8951, 0.2664, -0.1614),
            (-0.7502, 1.7135, 0.0367),
            (0.0389, -0.0685, 1.0296),
        ),
        'cat02': (
            (0.7328, 0.4296, -0.1624),
            (-0.7036, 1.6975, 0.0061),
            (0.0030, 0.0136, 0.9834),
        ),
        'cat97s': (
            (0.8562, 0.3372, -0.1934),
            (-0.8360, 1.8327, 0.0033),
            (0.0357, -0.0469, 1.0112),
        ),
        'cam16': (
            (0.401288, 0.650173, -0.051461),
            (-0.250268, 1.204414, 0.045854),
            (-0.002079, 0.048952, 0.953127),
        ),
        'hpe': (
            (0.38971, 0.68898, -0.07868),
            (-0.22981, 1.18340, 0.04641),
            (0.00000, 0.00000, 1.00000),
        ),
        'vonkries': (
            (0.38971, 0.68898, -0.07868),
            (-0.22981, 1.18340, 0.04641),
            (0.00000, 0.00000, 1.00000),
        ),
        'xyz': (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    }

    matrix = torch.tensor(data[method], dtype=dtype, device=device)
    return matrix


def get_lms_to_xyz_matrix(
    method: str = 'bradford',
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
):
    """Returns a transformation matrix for the conversion from LMS space to
    CIE XYZ space.

    Parameters
    ----------
    method : CATMethod, default='bradford'
        The method of conversion. The argument is case-insensitive.
    dtype : torch.dtype, default=torch.float32
        The data type of the tensor. Must be a floating point.
    device: torch.device | str, default='cpu'
        The device of the tensor.

    Returns
    -------
    torch.Tensor
        The transformation matrix with shape (3, 3).

    Raises
    ------
    ValueError
        When `dtype` is not a floating point.
    """
    if not dtype.is_floating_point:
        raise ValueError(f'dtype must be floating point.')
    _dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    matrix = get_xyz_to_lms_matrix(method, _dtype, device)
    matrix = matrix.inverse()
    if _dtype is not dtype:
        matrix = matrix.type(dtype)
    return matrix


def xyz_to_lms(
    xyz: torch.Tensor,
    method: str = 'bradford',
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to LMS space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    method : CATMethod, default='bradford'
        The method of conversion. The argument is case-insensitive.
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    lms : torch.Tensor
        An image in LMS space with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert CIE XYZ to LMS.
        `mat` is returned only if `ret_matrix` is true.
    """
    mat = get_xyz_to_lms_matrix(method, xyz.dtype, xyz.device)
    lms = matrix_transform(xyz, mat)
    if ret_matrix:
        return lms, mat
    return lms


def lms_to_xyz(
    lms: torch.Tensor,
    method: str = 'bradford',
    ret_matrix: bool = False,
):
    """Converts an image from LMS space to CIE XYZ space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in LMS space with shape (*, 3, H, W).
    method : CATMethod, default='bradford'
        The method of conversion. The argument is case-insensitive.
    ret_matrix : bool, default=False
        If false, only the image is returned.
        If true, also returns the transformation matrix.

    Returns
    -------
    xyz : torch.Tensor
        An image in CIE XYZ space with the shape (*, 3, H, W).
    mat : torch.Tensor
        A transformation matrix used to convert LMS to CIE XYZ.
        `mat` is returned only if `ret_matrix` is true.
    """
    mat = get_lms_to_xyz_matrix(method, lms.dtype, lms.device)
    xyz = matrix_transform(lms, mat)
    if ret_matrix:
        return xyz, mat
    return xyz
