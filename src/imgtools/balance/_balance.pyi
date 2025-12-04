__all__ = [
    'get_von_kries_transform_matrix',
    'von_kries_transform',
    'balance_by_scaling',
    'gray_world_balance',
    'gray_edge_balance',
    'white_patch_balance',
    'linear_regression_balance',
]

from typing import Literal, overload

import torch

from ..color._lms import CATMethod

def get_von_kries_transform_matrix(
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: str | CATMethod = 'bradford',
) -> torch.Tensor: ...

#
@overload
def von_kries_transform(
    xyz: torch.Tensor,
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: str | CATMethod | torch.Tensor = 'bradford',
    ret_matrix: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def von_kries_transform(
    xyz: torch.Tensor,
    xyz_white: torch.Tensor,
    xyz_target_white: torch.Tensor,
    method: str | CATMethod | torch.Tensor = 'bradford',
    ret_matrix: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def balance_by_scaling(
    img: torch.Tensor,
    scaled_max: int | float | torch.Tensor,
    ret_factors: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def balance_by_scaling(
    img: torch.Tensor,
    scaled_max: int | float | torch.Tensor,
    ret_factors: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def gray_world_balance(
    rgb: torch.Tensor,
    ret_factors: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def gray_world_balance(
    rgb: torch.Tensor,
    ret_factors: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def gray_edge_balance(
    rgb: torch.Tensor,
    edge: torch.Tensor,
    ret_factors: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def gray_edge_balance(
    rgb: torch.Tensor,
    edge: torch.Tensor,
    ret_factors: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def white_patch_balance(
    rgb: torch.Tensor,
    q: int | float | torch.Tensor = 1.0,
    ret_factors: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def white_patch_balance(
    rgb: torch.Tensor,
    q: int | float | torch.Tensor = 1.0,
    ret_factors: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
def linear_regression_balance(rgb: torch.Tensor) -> torch.Tensor: ...

#
def cheng_pca_balance(
    rgb: torch.Tensor,
    adaptation: Literal['rgb', 'von kries'] = 'von kries',
) -> torch.Tensor: ...
