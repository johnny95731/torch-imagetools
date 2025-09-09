import torch
from torch_imagetools.utils.helpers import tensorlize


def srgb_to_srgb_linear(
    srgb: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    srgb = tensorlize(srgb)
    linear = torch.empty_like(srgb) if out is None else out

    mask_leq = srgb <= 0.04045
    lower = srgb[mask_leq] * (1 / 12.92)

    mask_gt = torch.bitwise_not(mask_leq)
    # ((rgb + 0.055) / 1.055) ** 2.4
    higher = torch.add(srgb[mask_gt], 0.055).mul_(1 / 1.055).pow_(2.4)

    linear[mask_leq] = lower
    linear[mask_gt] = higher
    return linear


def srgb_linear_to_srgb(
    linear: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    linear = tensorlize(linear)
    srgb = torch.empty_like(linear) if out is None else out

    mask_leq = linear <= 0.0031308
    lower = linear[mask_leq] * 12.92

    mask_gt = torch.bitwise_not(mask_leq)
    higher = torch.pow(linear[mask_gt], 1 / 2.4).mul_(1.055).sub_(0.055)

    srgb[mask_leq] = lower
    srgb[mask_gt] = higher
    return srgb


def adobe_rgb_to_linear(
    adobe_rgb: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    adobe_rgb = tensorlize(adobe_rgb)
    linear = torch.empty_like(adobe_rgb) if out is None else out

    linear = torch.pow(adobe_rgb, 2.19921875, out=linear)
    return linear
