__all__ = ['reinhard2002']

import torch

from ...color._grayscale import rgb_to_gray
from ...filters.blur import gaussian_blur
from ...utils.helpers import align_device_type


def scale_luminance(
    lum: torch.Tensor,
    mid_gray: float = 1.0,
    where: torch.Tensor | None = None,
):
    if where is None:
        geo_mean = lum.add(1e-7).log().mean(dim=(-1, -2), keepdim=True).exp()
    else:
        where = align_device_type(where, lum)
        count_valid = torch.count_nonzero(where.flatten(-2, -1), -1)
        log = lum.add(1e-7).log()
        masked_mean = (log * where).sum((-1, -2), keepdim=True) / count_valid
        geo_mean = masked_mean.exp()
    scaled_lum = (mid_gray / (geo_mean + 1e-7)) * lum
    return scaled_lum


def global_tone_mapping(
    lum: torch.Tensor,
    l_white: float | None = 0.9,
):
    tone = lum / (lum + 1)
    if l_white is not None and (l_white < float('inf')):
        tone = tone * (1 + lum / l_white**2)
    return tone


def local_tone_mapping(
    lum: torch.Tensor,
    mid_gray: float = 1.0,
    num_scale: int = 4,
    alpha: float = 0.35355,
    ratio: float = 1.6,
    phi: float = 8.0,
    thresh: float = 0.05,
):
    bias = mid_gray * (2**phi)
    v1s = [gaussian_blur(lum, 0, alpha) * (torch.pi * alpha**2)]
    vs = []
    for i in range(1, num_scale):
        sigma = alpha * (ratio**i) * (i + 1)
        # A constant from gaussian kernel
        multiplier = torch.pi * (sigma) ** 2
        # relation between multiplication and fft
        # fft(x / a) = a * fft(x)
        v2 = gaussian_blur(lum, 0, sigma) * (multiplier)
        v1s.append(v2)
        v1 = v1s[i]
        nume = (v2 - v1).abs()
        deno = (bias / i**2) + v1
        vs.append(nume / deno)
    vsm = v1s.pop(-1)
    for _ in range(num_scale - 1):
        mask = vs.pop() < thresh
        torch.where(mask, v1s.pop(), vsm, out=vsm)
    tone = lum / (1.0 + vsm)
    return tone


def reinhard2002(
    img: torch.Tensor,
    mid_gray: float = 1.0,
    l_white: float | None = 0.9,
    num_scale: int = 4,
    alpha: float = 0.35355,
    ratio: float = 1.6,
    phi: float = 8.0,
    thresh: float = 0.05,
    tone: str = 'local',
    where: torch.Tensor | None = None,
):
    """Applies high dynamic range to an image by E. Reinhard's work [1].

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image in the range of [0, 1] with
        shape `(*, C, H, W)`.
    mid_gray : float, default=1.0
        Key value. Affects significantly the luminance of output.
    l_white : float | None, default=0.9
        Maximum luminance. An argument for `tone == 'global'`.
        See the equation (4) in [1].
    num_scale : int, default=4
        Number of scale. An argument for `tone == 'local'`.
    alpha : float, default=0.35355
        Basic sigma for gaussian kernel. An argument for `tone == 'local'`.
    ratio : float, default=1.6
        Sigma groth rate when scale increase. An argument for `tone == 'local'`.
    phi : float, default=8.0
        Sharpening parameter
    thresh : float, default=0.05
        Threshold value for scale selection.
    tone : {'local', 'global'}, default='local'
        Tone mapping method.
    where : torch.Tensor | None, default=None
        Mask to estimate the initial luminance scaling. The argument will
        NOT mask the output.

    Returns
    -------
    torch.Tensor
        High dynamic image with shape `(*, C, H, W)`.

    References
    ----------
    [1] Erik Reinhard, Michael Stark, Peter Shirley, and James Ferwerda. 2002.
        Photographic tone reproduction for digital images.
        ACM Trans. Graph. 21, 3 (July 2002), 267-276.
        https://doi.org/10.1145/566654.566575
    """
    if img.size(-3) == 1:
        gray = img
    elif img.size(-3) == 3:
        gray = rgb_to_gray(img)
    else:
        raise ValueError(
            f'`img` must be 1 or 3 channels, but got {img.size(-3)}.'
        )
    scaled_lum = scale_luminance(gray, mid_gray, where)
    tone = tone.lower()
    if tone == 'local':
        tone_mapping = local_tone_mapping(
            scaled_lum,
            mid_gray,
            num_scale,
            alpha,
            ratio,
            phi,
            thresh,
        )
    elif tone == 'global':
        tone_mapping = global_tone_mapping(scaled_lum, l_white)
    else:
        raise ValueError(f'tone must be "global" or "local".')
    res = (tone_mapping * img).clip(0.0, 1.0)
    return res
