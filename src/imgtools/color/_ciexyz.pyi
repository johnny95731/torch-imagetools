__all__ = [
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

import torch

from ._rgb import RGBSpec

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

#
def get_white_point(
    white: str | StandardIlluminants,
    obs: Literal[2, '2', 10, '10'] = 10,
) -> WhitePoint: ...

#
def get_rgb_model(rgb_spec: RGBSpec) -> RGBModel: ...

#
def get_rgb_to_xyz_matrix(
    rgb_spec: str | RGBSpec,
    white: str | StandardIlluminants,
    obs: Literal[2, '2', 10, '10'] = 10,
) -> torch.Tensor: ...
def get_xyz_to_rgb_matrix(
    rgb_spec: str | RGBSpec,
    white: str | StandardIlluminants,
    obs: Literal[2, 10] = 10,
) -> torch.Tensor: ...

#
@overload
def rgb_to_xyz(
    rgb: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def rgb_to_xyz(
    rgb: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def xyz_to_rgb(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def xyz_to_rgb(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#

def normalize_xyz(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    inplace: bool = False,
) -> torch.Tensor: ...
def unnormalize_xyz(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    inplace: bool = False,
) -> torch.Tensor: ...
