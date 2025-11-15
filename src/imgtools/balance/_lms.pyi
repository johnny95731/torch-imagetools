from typing import Literal, overload

import torch

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
- hpe (Hunt-Pointer-Estevez) : The traditional transformation method.
- vonkries (von Kries) : Same as hpe.
- cat97s : CIECAM97s.
- cat02 : CIECAM02.
- cam16 : CAM16. Not a CIE standard
- xyz : Regards XYZ as the cone response of the human eyes.
"""

def get_xyz_to_lms_matrix(method: CATMethod = 'bradford') -> torch.Tensor: ...
def get_lms_to_xyz_matrix(method: CATMethod = 'bradford') -> torch.Tensor: ...

#
@overload
def xyz_to_lms(
    xyz: torch.Tensor,
    method: CATMethod = 'bradford',
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def xyz_to_lms(
    xyz: torch.Tensor,
    method: CATMethod = 'bradford',
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...

#
@overload
def lms_to_xyz(
    lms: torch.Tensor,
    method: CATMethod = 'bradford',
    ret_matrix: bool = False,
) -> torch.Tensor: ...
@overload
def lms_to_xyz(
    lms: torch.Tensor,
    method: CATMethod = 'bradford',
    ret_matrix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
