"""Corelated color temperature (CCT) module."""

__all__ = [
    'mccamy_approximation',
    'hernandez_andre_approximation',
]

import torch

def mccamy_approximation(xy: torch.Tensor) -> torch.Tensor: ...

#
def hernandez_andre_approximation(xy: torch.Tensor) -> torch.Tensor: ...
