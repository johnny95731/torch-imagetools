__all__ = [
    'xyz_to_luv',
    'luv_to_xyz',
    'rgb_to_luv',
    'luv_to_rgb',
]

from typing import Literal, overload

import torch

from ._ciexyz import StandardIlluminants
from ._rgb import RGBSpec

@overload
def xyz_to_luv(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def xyz_to_luv(
    xyz: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def luv_to_xyz(
    luv: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def luv_to_xyz(
    luv: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def rgb_to_luv(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def rgb_to_luv(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def luv_to_rgb(
    luv: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def luv_to_rgb(
    luv: torch.Tensor,
    rgb_spec: RGBSpec | torch.Tensor = 'srgb',
    white: StandardIlluminants = 'D65',
    obs: Literal[2, '2', 10, '10'] = 10,
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
