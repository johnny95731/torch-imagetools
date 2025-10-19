__all__ = [
    'linearize_srgb',
    'gammaize_srgb',
    'linearize_adobe_rgb',
    'gammaize_adobe_rgb',
    'linearize_prophoto_rgb',
    'gammaize_prophoto_rgb',
    'linearize_rec2020',
    'gammaize_rec2020',
    'linearize_rgb',
    'gammaize_rgb',
]

from typing import Literal

import torch

RGBSpec = Literal[
    'srgb',
    'adobergb',
    'prophotorgb',
    'rec2020',
    'displayp3',
    'widegamut',
    'ciergb',
]
"""RGB specifications."""

#
def linearize_srgb(
    srgb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...
def gammaize_srgb(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...

#
def linearize_adobe_rgb(
    adobe_rgb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...
def gammaize_adobe_rgb(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...

#
def linearize_prophoto_rgb(
    prophoto_rgb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...
def gammaize_prophoto_rgb(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...

#
def linearize_rec2020(
    rec2020: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...
def gammaize_rec2020(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...

#
def linearize_rgb(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec = 'srgb',
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...
def gammaize_rgb(
    rgb: torch.Tensor,
    rgb_spec: RGBSpec = 'srgb',
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...
