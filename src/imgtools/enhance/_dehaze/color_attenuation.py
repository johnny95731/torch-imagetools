__all__ = [
    'color_attenuation_dehaze',
]
import torch

from ...color._grayscale import rgb_to_gray
from ...color._hsv import rgb_to_hsv
from ...filters.blur import guided_filter, min_filter


def calc_depth_map(
    rgb: torch.Tensor,
    patch_size: int = 3,
    guide_ksize: int = 43,
    guide_eps: float = 0.1,
):
    """Computes depth map of an RGB image.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    patch_size : int, default=3
        The kernel size.
    guide_ksize : int, default=43
        Kernel size for guided filetr to refine transmission.
    guide_eps : float, default=0.1
        Epsilon for guided filetr to refine transmission.

    Returns
    -------
    torch.Tensor
        Depth map. Shape `(*, 1, H, W)`.
    """
    hsv = rgb_to_hsv(rgb)
    _, sat, val = torch.unbind(hsv, -3)
    epsilon = torch.randn(rgb.shape[-2:], device=rgb.device).mul_(0.041337)
    depth = 0.121779 + 0.959710 * val - 0.780245 * sat + epsilon
    depth.unsqueeze_(-3)
    depth_mini = min_filter(depth, patch_size)
    gray = rgb_to_gray(rgb)
    depth_guide = guided_filter(depth_mini, gray, guide_ksize, guide_eps)
    return depth_guide, depth_mini, depth


def estimate_atmospheric_light(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    percent: float = 0.1,
):
    """Estimates atmospheric light of an RGB image.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    depth : torch.Tensor
        The depth map of `rgb`. Shape `(*, 1, H, W)`.
    percent : float, default=0.1
        The percentage for selecting data.

    Returns
    -------
    torch.Tensor
        Estimated atmospheric light. Shape `(*, 1, H, W)`.
    """
    flatted = rgb.flatten(-2)
    flatted_depth = depth.flatten(-2)
    n_sample = max(int(flatted.size(-1) * percent / 100), 1)
    data, _ = flatted_depth.sort(-1, descending=True)
    # Sample top-n deepest points
    sample = data[..., :n_sample]
    norm = sample.norm(dim=-2, keepdim=True)
    # lightest points in deepest points
    data, _ = norm.sort(descending=True)
    select_num = min(n_sample, 20)
    selected = data[..., :select_num]
    atmo = selected.amax(-1, keepdim=True).unsqueeze_(-1)
    return atmo


def color_attenuation_dehaze(
    rgb: torch.Tensor,
    patch_size: int = 3,
    beta: float = 1.0,
    t_min: float = 0.05,
    t_max: float = 1.0,
    percent: float = 0.1,
    guide_ksize: int = 43,
    guide_eps: float = 0.1,
):
    """Image dehazing by using color attenuation prior.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, C, H, W)`
    patch_size : int, default=3
        The neighborhood size for estimating the depth.
    beta : float, default=1.0
        _description_, by default 1.0
    t_min : float, default=0.05
        The minimum of the transmission.
    t_max : float, default=1.0
        The maximum of the transmission.
    percent : float, default=0.1
        The percentage for selecting data to estimate the atmospheric light.
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
    depth, _, _ = calc_depth_map(rgb, patch_size, guide_ksize, guide_eps)
    trans = depth.mul(-beta).exp()
    trans = trans.clip(t_min, t_max)
    atmo = estimate_atmospheric_light(rgb, depth, percent)
    dehazed = (rgb - atmo) / trans + atmo
    dehazed = dehazed.clip(0.0, 1.0)
    return dehazed
