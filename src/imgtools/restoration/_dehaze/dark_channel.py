__all__ = [
    'dark_channel_dehaze',
]

import torch

from ...filters.blur import guided_filter, min_filter
from ...utils.helpers import check_valid_image_ndim


def calc_dark_channel(rgb: torch.Tensor, patch_size: int = 3):
    """Computes dark channel of an RGB image.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    patch_size : int, default=3
        The kernel size.

    Returns
    -------
    torch.Tensor
        Dark channel. Shape `(*, 1, H, W)`.
    """
    mini_ch = rgb.amin(-3, keepdim=True)
    dark_channel = min_filter(mini_ch, patch_size)
    return dark_channel


def estimate_atmospheric_light(
    rgb: torch.Tensor,
    dark: torch.Tensor,
    percent=0.1,
):
    """Estimates atmospheric light of an RGB image.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    dark : torch.Tensor
        The dark channel of an image. Shape `(*, 1, H, W)`.
    percent : float, default=0.1
        The percentage for selecting data.

    Returns
    -------
    torch.Tensor
        Estimated atmospheric light. Shape `(*, 1, H, W)`.
    """
    flatted = rgb.flatten(-2)
    flatted_dark = dark.flatten(-2)
    top_k = max(int(flatted.size(-1) * percent / 100), 1)
    data, _ = torch.sort(flatted_dark, descending=True)
    atmo_light = data[..., :top_k].amax(-1, keepdim=True).unsqueeze_(-1)
    return atmo_light


def calc_transmission(
    rgb: torch.Tensor,
    atmo: torch.Tensor,
    patch_size: int = 3,
    omega: float = 0.95,
    t_min: float = 0.4,
    ksize: int = 43,
    eps: float = 0.1,
):
    """Computes the transmission map.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    atmo : torch.Tensor
        The atmospheric light. Shape `(*, 1, H, W)`.
    patch_size : int, default=3
        Kernel size for computing dark channel.
    omega : float, default=0.95
        Coefficient for preserving the haze. An lower value means more haze.
    t_min : float, default=0.4
        The minimum of the transmission.
    ksize : int, default=43
        Kernel size for guided filetr to refine transmission.
    eps : float, default=0.1
        Epsilon for guided filetr to refine transmission.

    Returns
    -------
    torch.Tensor
        The transmission map. Shape `(*, 1, H, W)`.
    """
    dark = calc_dark_channel(rgb / atmo, patch_size)
    trans = 1 - omega * dark
    trans = trans.clip(t_min)
    # Refine transmission
    trans = guided_filter(trans, rgb, ksize, eps)
    return trans


def dark_channel_dehaze(
    rgb: torch.Tensor,
    patch_size: int = 3,
    percent: float = 0.1,
    omega: float = 0.95,
    t_min: float = 0.4,
    guide_ksize: int = 43,
    guide_eps: float = 0.1,
):
    """Image dehazing by dark channel prior.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of `[0, 1]` with shape `(*, 3, H, W)`.
    patch_size : int, default=3
        Kernel size for computing dark channel.
    percent : float, default=0.1
        The percentage for selecting data to estimate the atmospheric light.
    omega : float, default=0.95
        Coefficient for preserving the haze. An lower value means more haze.
    t_min : float, default=0.4
        Minimum of the transmission.
    guide_ksize : int, default=43
        Kernel size for guided filetr to refine transmission.
    guide_eps : float, default=0.1
        Epsilon for guided filetr to refine transmission.

    Returns
    -------
    torch.Tensor
        Dehazed RGB image with shape `(*, 3, H, W)`.

    Examples
    --------

    >>> from imgtools import enhance
    >>> res = enhance.dark_channel_dehaze(img)
    """
    check_valid_image_ndim(rgb, 3)
    dark_channel = calc_dark_channel(rgb, patch_size)
    atmo_light = estimate_atmospheric_light(rgb, dark_channel, percent)
    trans = calc_transmission(
        rgb,
        atmo_light,
        patch_size,
        omega,
        t_min,
        guide_ksize,
        guide_eps,
    )
    dehazed = (rgb - atmo_light) / trans + atmo_light
    dehazed = dehazed.clip(0.0, 1.0)
    return dehazed
