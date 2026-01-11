__all__ = [
    'hist_equalize',
    'match_mean_std',
]

import torch

from ..utils.helpers import align_device_type, check_valid_image_ndim


def hist_equalize(img: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Apply histogram equalization to the image.

    Parameters
    ----------
    img : torch.Tensor
        An image in the range of [0, 1] with shape 2 <= img.ndim <= 4.
    bins : int, default=256
        The number of groups in data range.

    Returns
    -------
    torch.Tensor
        Enhanced image in the range of [0, 1]. Shape=img.shape.
    """
    if not isinstance(bins, int):
        raise TypeError(f'`bins` must be an integer: {type(bins)}.')
    res_dtype = img.dtype
    bins_m1_f = float(bins - 1)
    check_valid_image_ndim(img, 2)
    img = (img * bins_m1_f).type(torch.uint8)
    # Compute histogram
    flat_image = img.flatten(start_dim=-2).long()
    hist = torch.zeros(
        img.shape[:-2] + (bins,),
        dtype=torch.int32,
        device=img.device,
    )
    hist.scatter_add_(
        dim=-1, index=flat_image, src=hist.new_ones(1).expand_as(flat_image)
    )
    cdf = hist.cumsum(dim=-1)
    # Compute table
    maxi = cdf[..., bins - 1].unsqueeze_(-1).float()
    nonzero = cdf.count_nonzero(-1).unsqueeze_(-1)  # cdf is increasing
    mini_idx = torch.sub(bins, nonzero).clip_(0, bins - 1)
    # minimum of non-zeros velues in channel.
    mini = torch.gather(cdf, -1, index=mini_idx).float()
    # Normalize cdf to [0, 255]
    # table(x) = floor((x - min) / (max - min) * 255) / 255
    coeff = torch.div(256.0, maxi.sub_(mini))
    coeff.nan_to_num_(0.0, 0.0, 0.0)
    table = (
        ((cdf - mini).mul_(coeff))
        .floor_()
        .clip_(0.0, bins_m1_f)
        .div_(bins_m1_f)
    )
    table = table.type(res_dtype)

    res = torch.gather(table, -1, index=flat_image).reshape(img.shape)
    res = res.contiguous()
    return res


def match_mean_std(src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    """Match the mean and standard deviation.

    Parameters
    ----------
    src : torch.Tensor
        Source image with shape (*, C, H, W).
    tar : torch.Tensor
        Target image with shape (*, C, H, W).

    Returns
    -------
    torch.Tensor
        Enhanced image. Shape=max(src.shape, tar.dtype).
    """
    tar = align_device_type(tar, src)

    std_src, mean_src = torch.std_mean(src, dim=(-1, -2), keepdim=True)
    std_tar, mean_tar = torch.std_mean(tar, dim=(-1, -2), keepdim=True)

    # res = (src - mean_src) * (std_tar / std_src) + mean_tar
    slope = (std_tar / std_src).nan_to_num_(1.0, 1.0, 1.0)
    bias = mean_tar - slope * mean_src
    res = src * slope + bias
    return res
