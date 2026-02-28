__all__ = ['reinhard2002']

import torch

from ...color._grayscale import rgb_to_gray
from ...filters.rfft import get_gaussian_lowpass
from ...utils.helpers import align_device_type, check_valid_image_ndim


def scale_luminance(
    lum: torch.Tensor,
    exposure: float = 1.7,
    where: torch.Tensor | None = None,
):
    """Scales the image by the ratio of the `mid_gray` to the geometric mean
    of `lum`.

    Parameters
    ----------
    lum : torch.Tensor
        An image with shape `(*, C, H, W)`
    exposure : float, default=1.0
        Scaling coefficient.
    where : torch.Tensor | None, default=None
        Masking the lum when computing geometric mean.

    Returns
    -------
    torch.Tensor
        Scaled image with shape `(*, C, H, W)`.
    """
    log_lum = lum.add(1e-8).log_()
    if where is None:
        geo_mean = log_lum.mean(dim=(-1, -2, -3), keepdim=True).exp_()
    else:
        where = align_device_type(where, lum)
        count_valid = torch.count_nonzero(where.flatten(-2, -1), -1)
        log = lum.add(1e-7).log()
        masked_mean = (log * where).sum((-1, -2), keepdim=True) / count_valid
        geo_mean = masked_mean.exp_()
    scaled_lum = exposure * lum
    mid_gray = exposure * geo_mean
    return scaled_lum, mid_gray


def global_tone_mapping(
    lum: torch.Tensor,
    l_white: float | None = 0.9,
):
    """Computes the global tone mapping by E. Reinhard's work [1].

    Parameters
    ----------
    lum : torch.Tensor
        An image with shape `(*, C, H, W)`
    l_white : float | None, default=None
        Maximum luminance.

    Returns
    -------
    torch.Tensor
        The tone mapping. Shape `(*, C, H, W)`.

    References
    ----------
    [1] Erik Reinhard, Michael Stark, Peter Shirley, and James Ferwerda. 2002.
        Photographic tone reproduction for digital images.
        ACM Trans. Graph. 21, 3 (July 2002), 267-276.
        https://doi.org/10.1145/566654.566575
    """
    tone = lum / (lum + 1.0)
    if l_white is not None:
        tone *= 1 + lum / l_white**2
    return tone


def local_tone_mapping(
    lum: torch.Tensor,
    mid_gray: float = 0.72,
    l_white: float = 0.9,
    num_scale: int = 4,
    alpha: float = 0.35355,
    ratio: float = 1.6,
    phi: float = 8.0,
    thresh: float = 0.05,
) -> torch.Tensor:
    """Computes the local tone mapping by E. Reinhard's work [1].

    Parameters
    ----------
    lum : torch.Tensor
        An image with shape `(*, C, H, W)`
    mid_gray : float, default=0.72
        Key value.
    num_scale : int, default=4
        The number of scales.
    alpha : float, default=0.35355
        Basic blurring strength.
    ratio : float, default=1.6
        The ratio between two blurring strength.
    phi : float, default=8.0
        Sharpening parameter
    thresh : float, default=0.05
        Threshold value for scale selection.

    Returns
    -------
    torch.Tensor
        The tone mapping. Shape `(*, C, H, W)`.

    References
    ----------
    [1] Erik Reinhard, Michael Stark, Peter Shirley, and James Ferwerda. 2002.
        Photographic tone reproduction for digital images.
        ACM Trans. Graph. 21, 3 (July 2002), 267-276.
        https://doi.org/10.1145/566654.566575
    """
    bias = mid_gray * (2**phi)
    lum_f = torch.fft.rfft2(lum)  # type: torch.Tensor
    rec_size = lum.shape[-2:]
    sigma0 = 2 * torch.pi * alpha
    for scale in range(num_scale, 0, -1):
        sigma = sigma0 * ratio ** (scale - 1)
        lowpass = get_gaussian_lowpass(
            lum_f, 1 / sigma, d=1.0, device=lum_f.device
        )
        lowpass = align_device_type(lowpass, lum)
        v1 = torch.fft.irfft2(lum_f * lowpass, s=rec_size)  # type: torch.Tensor
        if scale == num_scale:
            v1sm = v1
        else:
            nume = (v2 - v1).abs_()
            deno = bias / 2 ** (scale - 1) + v1
            vs = nume.div_(deno)
            mask = vs < thresh
            torch.where(mask, v1, v1sm, out=v1sm)
        v2 = v1  # noqa: F841
    tone = (lum / (1.0 + v1sm)).mul_(v1sm.div_(l_white**2))
    return tone


def reinhard2002(
    img: torch.Tensor,
    exposure: float = 1.7,
    l_white: float | None = None,
    num_scale: int = 4,
    alpha: float = 0.35355,
    ratio: float = 1.6,
    phi: float = 8.0,
    thresh: float = 0.05,
    tone: str = 'local',
    where: torch.Tensor | None = None,
):
    """Applies high dynamic range to an image by using the modified
    version of E. Reinhard's work [1].

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image in the range of [0, 1] with
        shape `(*, C, H, W)`.
    exposure : float, default=1.7
        An mutiplier for scaling the luminance. Affects significantly
        the luminance of output. The relation between exposure and mid
        gray (`a` in equation (2) of [1]) is
        `exposure = a / geometric_mean(lum)`
    l_white : float | None, default=None
        Maximum luminance. If None, the value will be set to the mean
        of luminance.
    num_scale : int, default=4
        The number of scales.
    alpha : float, default=0.35355
        Basic blurring strength.
    ratio : float, default=1.6
        The ratio between two blurring strength.
    phi : float, default=8.0
        Sharpening parameter
    thresh : float, default=0.05
        Threshold value for scale selection.
    tone : {'local', 'global'}, default='local'
        Tone mapping strategy.
    where : torch.Tensor | None, default=None
        Mask to estimate the initial luminance scaling. The argument will
        NOT mask the output.

    Returns
    -------
    torch.Tensor
        High dynamic image in the range of [0, 1] with shape `(*, C, H, W)`.

    Notes
    -----
    When `tone == 'global'`, only the arguments `exposure`, `l_white`, and
    `where` affect the result.
    When `tone == 'local'`, all arguments affect the result.

    References
    ----------
    [1] Erik Reinhard, Michael Stark, Peter Shirley, and James Ferwerda. 2002.
        Photographic tone reproduction for digital images.
        ACM Trans. Graph. 21, 3 (July 2002), 267-276.
        https://doi.org/10.1145/566654.566575
    """
    check_valid_image_ndim(img)
    if img.size(-3) == 1:
        gray = img
    elif img.size(-3) == 3:
        gray = rgb_to_gray(img)
    else:
        raise ValueError(
            f'`img` must be 1 or 3 channels, but got {img.size(-3)}.'
        )
    if l_white is None:
        l_white = gray.amax((-1, -2), keepdim=True)
    elif not isinstance(l_white, (int, float)):
        raise TypeError(f'`l_white` must be a positive number: {type(l_white)}')
    elif l_white <= 0:
        raise ValueError(f'`l_white` must be a positive number: {l_white}')
    if not isinstance(where, torch.Tensor) and where is not None:
        raise ValueError(
            f'`where` must be None or a Tensor type: {type(where)}'
        )
    scaled_lum, mid_gray = scale_luminance(gray, exposure, where)
    tone = tone.lower()
    if tone == 'local':
        tone_mapping = local_tone_mapping(
            scaled_lum,
            mid_gray,
            l_white,
            num_scale,
            alpha,
            ratio,
            phi,
            thresh,
        )
        tone_mapping.div_(gray.add(1e-8))
    elif tone == 'global':
        tone_mapping = global_tone_mapping(scaled_lum, l_white)
    else:
        raise ValueError(f'tone must be "global" or "local".')
    res = (img * tone_mapping).clip_(0.0, 1.0)
    return res
