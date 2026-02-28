__all__ = ['bilateral_hdr']

import torch
from torch.nn.functional import interpolate

from ...color._yuv import rgb_to_yuv, yuv_to_rgb
from ...filters.rfft import get_gaussian_lowpass
from ...utils.helpers import align_device_type, check_valid_image_ndim


# Note that the stopping functions are normalize to the range [c, 1]
# for som constant 0 < c < 1
def _edge_stopping_huber(diff: torch.Tensor, coeff: float):
    # f(x) = 1           if |x| <= coeff,
    #        coeff / |x| if |x| > coeff.
    # Range: [coeff / max(|x|), 1]
    dist = diff.abs()
    response = torch.div(coeff, dist.clip_(coeff), out=dist)
    return response


def _edge_stopping_lorentz(diff: torch.Tensor, coeff: float):
    # f(x) = coeff / (coeff + x**2)
    dist2 = diff.div(coeff).square_()
    response = torch.div(1.0, dist2.add_(1.0), out=dist2)
    return response


def _edge_stopping_turkey(diff: torch.Tensor, coeff: float):
    # f(x) = (1 - (x/coeff)**2)**2 if |x| <= coeff
    #        0                     if |x| > coeff
    # Range: [0, 1]
    dist2 = diff.div(coeff).square_()
    response = dist2.sub_(1.0).clip_(None, 0.0).square_()
    return response


def _edge_stopping_gaussian(diff: torch.Tensor, coeff: float):
    # f(x) = exp(- x**2 / coeff)
    dist2 = diff.square()
    response = dist2.div_(-coeff).exp_()
    return response


def normalize01(x: torch.Tensor, channelwise: bool = True):
    if channelwise:
        maxi = x.amax((-1, -2), keepdim=True)
        mini = x.amin((-1, -2), keepdim=True)
    else:
        maxi = x.amax(keepdim=True)
        mini = x.amin(keepdim=True)
    delta = maxi.sub_(mini)
    res = x.sub(mini).div_(delta)
    return res


def _weight(
    diff: torch.Tensor,
    g_j: torch.Tensor,
    delta: torch.Tensor,
    strategy: str = 'soft',
):
    """Computes the interpolation weight for fast bilateral filter.

    Parameters
    ----------
    diff : torch.Tensor
        The difference of input image and a color: image -
    g_j : torch.Tensor
        The influence in intensity domain.
    delta : torch.Tensor
        The delta between two intensity center.
    strategy : str, default='soft'
        The strategy of computing weights.

    Returns
    -------
    torch.Tensor
        The interpolation weight.
    """
    # Note: `g_j` can be use to compute weights.
    diff = diff.abs_()
    if strategy == 'soft':
        weight = g_j.pow_(3.0)
    elif strategy == 'lighter':
        weight = (1.0 - diff).clip_(0.0, 1.0).pow_(3.0)
    elif strategy == 'linear':
        weight = (1.0 - diff).clip_(0.0, 1.0)
    elif strategy == 'std':
        weight = 1.0 - diff / delta
    else:
        raise ValueError(
            '`inter_weight` must be one of ("soft", "lighter", "linear", '
            f'"std"), but received `{strategy}`.'
        )
    mask = (diff < delta).float()
    weight *= mask
    return weight


def bilateral_hdr(
    img: torch.Tensor,
    sigma_c: float = 0.15,
    sigma_s: float | None = 1.0,
    contrast: float = 1.5,
    downsample: float = 1,
    edge_stopping: str = 'gaussian',
    tone: str = 'soft',
):
    """Applies high dynamic range to an image by using Durand's work [1]. (
    modified fast bilateral filter).

    Parameters
    ----------
    img : torch.Tensor
        An RGB image in the range of [0, 1] with shape `(*, C, H, W)`.
    sigma_c : float
        Sigma in the color intensity. A larger value means that the
        dissimilar intensity will cause more effect.
    sigma_s : float | None, default=1.0
        Sigma in the space/coordinate. A larger value means that the
        farther points will cause more effect. If None, `sigma_s` is set
        to be `min(H, W) * 0.02`.
    contrast: float, default=1.5
        The contrast factor. When `tone == 'std'`, the `contrast` will be
        clipped to [1, inf).
    downsample : float, default=1
        Downsample rate. A smaller value means small size in iteration.
    edge_stopping : {'huber', 'lorentz', 'turkey', 'gaussian'}, default='gaussian'
        Edge-stopping function. A function for preventing diffusion between
        dissimilar intensity. The value `'huber'` is only recommended
        with `tone == 'std'`.
    tone : {'soft', 'lighter', 'linear', 'std'}, default='soft'
        Tone mapping strategy. The `'std'` is closed to the standard strategy.
        See notes section for details.

    Returns
    -------
    torch.Tensor
        High dynamic image with shape `(*, C, H, W)`.

    Notes
    -----
    The argument `tone` mainly affects the interpolation of weights when
    computing fast bilateral filter.
    The options of `tone` other than `'std'` give larger weights, thus, is
    brighter. And, theese options are faster since the computation of
    tone mapping is simpler.

    References
    ----------
    [1] F. Durand and J. Dorsey, "Fast bilateral filtering for the display of
        high-dynamic-range images," SIGGRAPH '02', pp. 257-266, Jul. 2002
        doi: 10.1145/566570.566574.

    Examples
    --------

    >>> from imgtools.enhance import bilateral_hdr
    >>>
    >>> hdr = bilateral_hdr(rgb, 0.25, 0.1)
    >>> hdr2 = bilateral_hdr(rgb, 0.25, None, tone='std')
    >>> hdr3 = bilateral_hdr(dark_rgb, 0.25, None, contrast=0.5)
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
        coeff_c = sigma_c
        color_fn = _edge_stopping_lorentz
    elif edge_stopping == 'turkey':
        coeff_c = sigma_c * 5**0.5
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
    sigma_s *= downsample * 2 * torch.pi
    # Only using the brightness.
    is_color = img.size(-3) == 3
    if is_color:
        yuv = rgb_to_yuv(img)
        _img_t0 = yuv[..., :1, :, :]
    elif img.size(-3) == 1:
        _img_t0 = img
    else:
        raise ValueError(f'The channel of `img` must be 1 or 3: {img.size(-3)}')
    if tone == 'std':
        lum = _img_t0
        _img_t0 = _img_t0.add(1e-7).log_()
    # Downsample
    if not isinstance(downsample, (int, float)):
        raise TypeError(f'`downsample` must be a number: {type(downsample)}')
    elif downsample < 0 or downsample > 1:
        raise TypeError(
            f'`downsample` must be a number in (0, 1]: {downsample}'
        )
    ori_size = (_img_t0.size(-2), _img_t0.size(-1))
    down_size = (int(ori_size[0] * downsample), int(ori_size[1] * downsample))
    should_downsample = down_size != ori_size
    if should_downsample:
        img_t0 = interpolate(_img_t0, down_size, mode='area')
    else:
        img_t0 = _img_t0
    # Pre-computed constant
    mini = torch.amin(img_t0, dim=(-2, -1), keepdim=True)
    maxi = torch.amax(img_t0, dim=(-2, -1), keepdim=True)
    delta = maxi.sub_(mini)
    num_seg = max(int((torch.amax(delta) / sigma_c).round().item()), 2)
    delta /= num_seg - 1
    # Fast bilateral filter
    base = torch.zeros_like(_img_t0)
    for j in range(num_seg):
        i_j = j * delta + mini

        g_j = color_fn(img_t0 - i_j, coeff_c)
        g_j_f = torch.fft.rfft2(g_j)  # type: torch.Tensor
        if j == 0:
            space_kernel = get_gaussian_lowpass(
                g_j_f,
                1 / sigma_s,
                d=1.0,
                device=_img_t0.device,
            )
            space_kernel = align_device_type(space_kernel, _img_t0)
        k_j_f = g_j_f * space_kernel
        h_j = g_j * img_t0
        h_j_f = torch.fft.rfft2(h_j)  # type: torch.Tensor
        h_star_j_f = h_j_f.mul_(space_kernel)

        k_j = torch.fft.irfft2(k_j_f, s=g_j.shape[-2:])  # type: torch.Tensor
        h_star_j = torch.fft.irfft2(h_star_j_f, s=h_j.shape[-2:])  # type: torch.Tensor
        j_j = h_star_j / (k_j + 1e-7)
        # Get interpolation weights
        diff = img_t0 - i_j
        weight = _weight(diff, g_j, delta, tone)
        delta_base = j_j.mul_(weight)
        if should_downsample:
            delta_base = interpolate(
                delta_base, ori_size, mode='bilinear', align_corners=True
            )
        base += delta_base
    # Tone mapping
    if tone == 'std':
        contrast = max(contrast, 1.0)
        detail = _img_t0.sub_(base)
        tone_mapping = (
            base.mul_(contrast).add_(detail).exp_().div_(lum.add(1e-8))
        )
        maxi_tone = tone_mapping.amax((-1, -2), keepdim=True)
        res = (img * tone_mapping.div_(maxi_tone)).clip_(0.0, 1.0)
    elif is_color:
        tone_mapping = base
        yuv[..., :1, :, :] = tone_mapping.clip_(0.0, 1.0).pow_(contrast)
        res = yuv_to_rgb(yuv)
    else:
        res = base.clip_(0.0, 1.0).pow_(contrast)
    if is_not_batch:
        res = res.squeeze(0)
    return res
