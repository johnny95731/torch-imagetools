"""Corelated color temperature (CCT) module."""

__all__ = [
    'mccamy_approximation',
    'hernandez_andre_approximation',
]

import torch


def mccamy_approximation(
    xy: torch.Tensor,
) -> torch.Tensor:
    """Calculates CCT from xy chromaticity coordinates by McCamy's
    approximation.

    Parameters
    ----------
    xy : torch.Tensor
        Chromaticity coordinates, a tensor with shape `(2, *)`.

    Returns
    -------
    torch.Tensor
        Correlated color temperature in Kelvin.
    """
    x, y = xy.unbind(0) if isinstance(xy, torch.Tensor) else xy

    n = (x - 0.3320) / (y - 0.1858)
    p = n * n
    cct = 437.0 * (p * n) + 3601.0 * p + 6861.0 * n + 5517.0
    return cct


def hernandez_andre_approximation(
    xy: torch.Tensor,
) -> torch.Tensor:
    """Calculates CCT from xy chromaticity coordinates by Hernández-André's
    approximation.

    Parameters
    ----------
    xy : torch.Tensor
        Chromaticity coordinates, a tensor with shape `(2, *)`.

    Returns
    -------
    torch.Tensor
        Correlated color temperature in Kelvin.
    """
    x, y = xy.unbind(0)

    # Low-temperature formula 3-50 kilo Kelvin
    n = (x - 0.3366).div(y - 0.1735)
    cct = (n * (1 / 0.92159)).exp().mul(6253.80338).sub(949.86315)
    cct += (n * (1 / 0.20039)).exp().mul(28.70599)
    cct += (n * (1 / 0.07125)).exp().mul(0.00004)

    over_50k_kelvin = cct > 50000.0
    if torch.any(over_50k_kelvin).item():
        # high-temperature formula 50-800 kilo Kelvin
        n = (x - 0.3356) / (y - 0.1691)
        high_temp = (n * (1 / 0.07861)).exp().mul(0.00228).add(36284.48953)
        high_temp += (n * (1 / 0.01543)).exp().mul(5.4535e-36)

        cct = torch.where(over_50k_kelvin, high_temp, cct)
    return cct
