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

import torch


def linearize_srgb(
    srgb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    linear = torch.empty_like(srgb) if out is None else out

    torch.where(
        srgb <= 0.04045,
        srgb.mul(1 / 12.92),
        srgb.add(0.055).mul_(1 / 1.055).pow_(2.4),
        out=linear,
    )
    return linear


def gammaize_srgb(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    srgb = torch.empty_like(linear) if out is None else out

    torch.where(
        linear <= 0.0031308,
        linear.mul(12.92),
        torch.pow(linear, 1 / 2.4).mul_(1.055).sub_(0.055),
        out=srgb,
    )
    return srgb


def linearize_adobe_rgb(
    adobe_rgb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    linear = torch.empty_like(adobe_rgb) if out is None else out

    torch.pow(adobe_rgb, 2.19921875, out=linear)
    return linear


def gammaize_adobe_rgb(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    adobe_rgb = torch.empty_like(linear) if out is None else out

    torch.pow(linear, 1 / 2.19921875, out=adobe_rgb)
    return adobe_rgb


def linearize_prophoto_rgb(
    prophoto_rgb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    linear = torch.empty_like(prophoto_rgb) if out is None else out

    torch.where(
        prophoto_rgb <= 0.5,
        prophoto_rgb.mul(1 / 16.0),
        prophoto_rgb.pow(1.8),
        out=linear,
    )
    return linear


def gammaize_prophoto_rgb(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    prophoto_rgb = torch.empty_like(linear) if out is None else out

    torch.where(
        linear <= 0.03125,  # 16/512
        linear.mul(16.0),
        linear.pow_(1 / 1.8),
        out=prophoto_rgb,
    )
    return prophoto_rgb


def linearize_rec2020(
    rec2020: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    linear = torch.empty_like(rec2020) if out is None else out

    alpha = 1.09929682680944
    torch.where(
        rec2020 < 0.0812428582986315,
        rec2020.mul(1 / 4.5),
        rec2020.add(alpha - 1.0).mul_(1 / alpha).pow_(1 / 0.45),
        out=linear,
    )
    return linear


def gammaize_rec2020(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    rec2020 = torch.empty_like(linear) if out is None else out

    alpha = 1.09929682680944
    torch.where(
        linear < 0.018053968510807,
        linear.mul(4.5),
        linear.pow(0.45).mul_(alpha).sub_(alpha - 1.0),
        out=rec2020,
    )
    return rec2020


def linearize_rgb(
    rgb: torch.Tensor,
    rgb_spec: str = 'srgb',
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    rgb_spec = rgb_spec.lower()
    table = {
        'srgb': linearize_srgb,
        'displayp3': linearize_srgb,
        'adobergb': linearize_adobe_rgb,
        # TODO: Check whether wide gamut is linear.
        'widegamut': linearize_adobe_rgb,
        'prophotorgb': linearize_prophoto_rgb,
        'rec2020': linearize_rec2020,
        'ciergb': lambda x, out=None: x,
    }
    linear = table[rgb_spec](rgb, out=out)  # type: torch.Tensor
    return linear


def gammaize_rgb(
    rgb: torch.Tensor,
    rgb_spec: str = 'srgb',
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    rgb_spec = rgb_spec.lower()
    table = {
        'srgb': gammaize_srgb,
        'displayp3': gammaize_srgb,
        'adobergb': gammaize_adobe_rgb,
        'widegamut': gammaize_adobe_rgb,
        'prophotorgb': gammaize_prophoto_rgb,
        'rec2020': gammaize_rec2020,
        'ciergb': lambda x, out=None: x,
    }
    gamma = table[rgb_spec](rgb, out=out)  # type: torch.Tensor
    return gamma
