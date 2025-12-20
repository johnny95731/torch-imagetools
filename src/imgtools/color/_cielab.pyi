__all__ = [
    'xyz_to_lab',
    'lab_to_xyz',
    'rgb_to_lab',
    'lab_to_rgb',
]

from typing import Literal, overload

import torch

from ._ciexyz import StandardIlluminants
from ._rgb import RGBSpec

@overload
def xyz_to_lab(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def xyz_to_lab(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def lab_to_xyz(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def lab_to_xyz(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def rgb_to_lab(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def rgb_to_lab(
    xyz: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def lab_to_rgb(
    lab: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def lab_to_rgb(
    lab: torch.Tensor,
    rgb_spec: str | RGBSpec = 'srgb',
    white: str | StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
