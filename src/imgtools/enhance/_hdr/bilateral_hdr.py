__all__ = ['bilateral_hdr']

from math import sqrt

import torch
from torch.nn.functional import interpolate

from ...filters.blur import get_gaussian_kernel
from ...utils.helpers import check_valid_image_ndim
from ...utils.math import filter2d


def _edge_stopping_huber(diff: torch.Tensor, coeff: float):
    dist = diff.abs()
    response = 1.0 / dist.clip(coeff)
    return response


def _edge_stopping_lorentz(diff: torch.Tensor, coeff: float):
    dist2 = diff.square()
    response = coeff / (dist2).add(coeff)
    return response


def _edge_stopping_turkey(diff: torch.Tensor, coeff: float):
    dist2 = diff.div(coeff * sqrt(5)).square()
    response = (1.0 - dist2).square().mul(0.5)
    return response


def _edge_stopping_gaussian(diff: torch.Tensor, coeff: float):
    dist2 = diff.square()
    response = dist2.div(-coeff).exp()
    return response


def bilateral_hdr(
    img: torch.Tensor,
    sigma_c: float,
    sigma_s: float | None = None,
    ksize: int = 0,
    downsample: float = 1,
    edge_stopping: str = 'gaussian',
):
    """Applies high dynamic range to an image by Durand's work [1] (
    modified fast bilateral filter).

    Parameters
    ----------
    img : torch.Tensor
        An RGB image in the range of [0, 1] with shape `(*, C, H, W)`.
    sigma_c : float
        Sigma in the color intensity. A larger value means that the
        dissimilar intensity will cause more effect.
    sigma_s : float | None, default=None
        Sigma in the space/coordinate. The value influence the gaussian
        kernel which is applied on spatial domain. If None is provided,
        `sigma_s` is set to be `min(H, W) * 0.02`.
    ksize : int | tuple[int, int], default=5
        Gaussian kernel size. If ksize is non-positive, the value will be
        computed from `sigma_s`:
        `ksize = odd(max(6 * sigma_s * downsample + 1, 3))`, where
        `odd(x)` returns the smallest odd integer that `odd(x) >= x`.
    downsample : float, default=1
        Downsample rate. A smaller value means small size in iteration.
    edge_stopping : {'huber', 'lorentz', 'turkey', 'gaussian'}, default='gaussian'
        Edge-stopping function. A function for preventing diffusion between
        dissimilar intensity.

    Returns
    -------
    torch.Tensor
        High dynamic image with shape `(*, C, H, W)`.

    References
    ----------
    [1] F. Durand and J. Dorsey, "Fast bilateral filtering for the display of
        high-dynamic-range images," SIGGRAPH '02', pp. 257-266, Jul. 2002
        doi: 10.1145/566570.566574.
    """
    is_not_batch = check_valid_image_ndim(img)
    if is_not_batch:
        img = img.unsqueeze(0)
    # Init args
    edge_stopping = edge_stopping.lower()
    if edge_stopping == 'huber':
        coeff_c = sigma_c
        color_fn = _edge_stopping_huber
    elif edge_stopping == 'lorentz':
        coeff_c = sigma_c**2
        color_fn = _edge_stopping_lorentz
    elif edge_stopping == 'turkey':
        coeff_c = sigma_c
        color_fn = _edge_stopping_turkey
    elif edge_stopping == 'gaussian':
        coeff_c = 2 * sigma_c**2
        color_fn = _edge_stopping_gaussian
    else:
        raise ValueError(
            "`edge_stopping` must be one of 'huber', 'lorentz', 'turkey', "
            f"of 'gaussian': {edge_stopping}"
        )
    if sigma_s is None:
        sigma_s = min(img.shape[-2:]) * 0.02
    #
    ori_size = (img.size(-2), img.size(-1))
    down_size = (int(ori_size[0] * downsample), int(ori_size[1] * downsample))
    img_down = interpolate(img, down_size, mode='bilinear', align_corners=True)

    mini = torch.amin(img_down, dim=(-3, -2, -1), keepdim=True)
    maxi = torch.amax(img_down, dim=(-3, -2, -1), keepdim=True)
    delta = maxi - mini
    num_seg = int(torch.amax((delta / sigma_c)).round().item())
    delta = delta / num_seg

    res = torch.zeros_like(img)
    space_kernel = get_gaussian_kernel(
        ksize,
        sigma_s * downsample,
        normalize=True,
    )
    for j in range(num_seg):
        i_j = j * delta + mini

        g_j = color_fn(img_down - i_j, coeff_c)
        k_j = filter2d(g_j, space_kernel)
        h_j = g_j * img_down
        h_star_j = filter2d(h_j, space_kernel)
        j_j = h_star_j / (k_j + 1e-7)
        j_j = interpolate(j_j, ori_size, mode='bilinear', align_corners=True)
        # Get interpolation weights
        diff = (img - i_j).abs()
        mask = (diff < delta).float()
        weight = mask * (1 - (diff / delta))
        res = res + j_j * weight
    res = res.clip(0.0, 1.0)
    if is_not_batch:
        res = res.squeeze(0)
    return res
