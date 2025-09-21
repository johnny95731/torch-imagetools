from typing import Literal, overload

import torch

from ..utils.helpers import matrix_transform


CATMethod = Literal[
    'bradford',
    'hpe',
    'vonkries',
    'cat97s',
    'cat02',
    'cam16',
    'xyz',
]
"""The methods about chromatic adaptation transform (CAT).

- bradford (Bradford) : The recommended method.
- hpe (Hunt–Pointer–Estevez) : The traditional transformation method.
- vonkries (von Kries) : Same as hpe.
- cat97s : CIECAM97s.
- cat02 : CIECAM02.
- cam16 : CAM16. Not a CIE standard
- xyz : Regards XYZ as the cone response of the human eyes.
"""


def get_xyz_to_lms_matrix(method: CATMethod = 'bradford'):
    """Returns a transformation matrix for the conversion from CIE XYZ space
    to LMS space.

    Parameters
    ----------
    method : CATMethod, optional
        The method of conversion, by default 'bradford'.
        The argument is case-insensitive.

    Returns
    -------
    torch.Tensor
        The transformation matrix with shape (3, 3).
    """
    method = method.lower()
    data: dict[CATMethod] = {
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

    matrix = torch.tensor(data[method])
    return matrix


def get_lms_to_xyz_matrix(method: CATMethod = 'bradford'):
    """Returns a transformation matrix for the conversion from LMS space to
    CIE XYZ space.

    Parameters
    ----------
    method : CATMethod, optional
        The method of conversion, by default 'bradford'.
        The argument is case-insensitive.

    Returns
    -------
    torch.Tensor
        The transformation matrix with shape (3, 3).
    """
    matrix = get_xyz_to_lms_matrix(method)
    matrix = matrix.inverse()
    return matrix


@overload
def xyz_to_lms(
    xyz: torch.Tensor,
    method: CATMethod = 'bradford',
    *,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def xyz_to_lms(
    xyz: torch.Tensor,
    method: CATMethod = 'bradford',
    *,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def xyz_to_lms(
    xyz: torch.Tensor,
    method: CATMethod = 'bradford',
    *,
    ret_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Converts an image from CIE XYZ space to LMS space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in CIE XYZ space with shape (*, 3, H, W).
    method : CATMethod, optional
        The method of conversion, by default 'bradford'.
        The argument is case-insensitive.
    ret_matrix : bool, optional
        If True, returns image and conversion matrix (XYZ -> LMS).
        By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An image in LMS space with the shape (*, 3, H, W). If ret_matrix
        is True, returns image and the transformation matrix.
    """
    matrix = get_xyz_to_lms_matrix(method)
    lms = matrix_transform(xyz, matrix)
    if ret_matrix:
        return lms, matrix
    return lms


@overload
def lms_to_xyz(
    lms: torch.Tensor,
    method: CATMethod = 'bradford',
    *,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def lms_to_xyz(
    lms: torch.Tensor,
    method: CATMethod = 'bradford',
    *,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def lms_to_xyz(
    lms: torch.Tensor,
    method: CATMethod = 'bradford',
    *,
    ret_matrix: bool = False,
):
    """Converts an image from LMS space to CIE XYZ space.

    Parameters
    ----------
    xyz : torch.Tensor
        An image in LMS space with shape (*, 3, H, W).
    method : CATMethod, optional
        The method of conversion, by default 'bradford'.
        The argument is case-insensitive.
    ret_matrix : bool, optional
        If True, returns image and conversion matrix (LMS -> XYZ).
        By default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        An image in CIE XYZ space with the shape (*, 3, H, W). If ret_matrix
        is True, returns image and the transformation matrix.
    """
    matrix = get_lms_to_xyz_matrix(method)
    xyz = matrix_transform(lms, matrix)
    if ret_matrix:
        return xyz, matrix
    return xyz
