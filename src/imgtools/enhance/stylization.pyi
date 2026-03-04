from typing import Literal

import torch

from ..color._ciexyz import StandardIlluminants
from ..color._lms import CATMethod
from ..color._rgb import RGBSpec

def transfer_reinhard(
    src: torch.Tensor,
    tar: torch.Tensor,
    rgb_spec: str | RGBSpec,
    white: str | StandardIlluminants,
    obs: Literal[2, '2', 10, '10'] | str | int = 10,
    adap_method: CATMethod = 'bradford',
) -> torch.Tensor: ...
