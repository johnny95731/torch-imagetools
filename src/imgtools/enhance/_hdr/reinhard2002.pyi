__all__ = ['reinhard2002']

from typing import Literal

import torch

def scale_luminance(
    lum: torch.Tensor,
    exposure: float = 1.7,
) -> torch.Tensor: ...

#
def global_tone_mapping(
    lum: torch.Tensor,
    l_white: float | None = 0.9,
) -> torch.Tensor: ...
def local_tone_mapping(
    lum: torch.Tensor,
    mid_gray: float = 0.72,
    num_scale: int = 4,
    alpha: float = 0.35355,
    ratio: float = 1.6,
    phi: float = 8.0,
    thresh: float = 0.05,
) -> torch.Tensor: ...

#
def reinhard2002(
    img: torch.Tensor,
    exposure: float = 1.7,
    l_white: float | None = None,
    num_scale: int = 4,
    alpha: float = 0.35355,
    ratio: float = 1.6,
    phi: float = 8.0,
    thresh: float = 0.05,
    tone: Literal['local', 'global'] = 'local',
) -> torch.Tensor: ...
