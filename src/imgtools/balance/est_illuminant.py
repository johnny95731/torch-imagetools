__all__ = [
    'estimate_illuminant_cheng',
]

import torch

from ..utils.helpers import check_valid_image_ndim
from ..utils.math import pca


def estimate_illuminant_cheng(
    img: torch.Tensor,
    n_selected: int | float = 3.5,
) -> torch.Tensor:
    """Estimate the illuminant in the image by Cheng's PCA method [1].

    Parameters
    ----------
    img : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, C, H, W).
    n_selected : int | float, default=3.5
        Percentage for selecting top-n and bottom-n points.
        Select (2 * n_selected)%  points in total.

    Returns
    -------
    torch.Tensor
        The illuminant of the image. An RGB value with shape=(*, C).

    References
    ----------
    [1] Cheng, Dongliang, Dilip K. Prasad, and Michael S. Brown. "Illuminant
        estimation for color constancy: why spatial-domain methods work and
        the role of the color distribution."
        JOSA A 31.5 (2014): 1049-1058.
    """
    is_not_batch = check_valid_image_ndim(img)
    shape = img.shape

    ch_mean = img.mean(dim=(-1, -2), keepdim=True)
    # Calculate distance
    proj = torch.sum(img * ch_mean, dim=-3)
    norm = torch.norm(img, 2, dim=-3) * torch.norm(ch_mean, 2)
    dist = (proj / norm).nan_to_num(0.0, 0.0)

    # Select top n-percent and bottom n-percent
    flatted = img.flatten(-2)
    flatted_dist = dist.flatten(int(is_not_batch) ^ 1)
    sorted_key = torch.argsort(flatted_dist)
    selected_num = int(n_selected / 100 * shape[-1] * shape[-2])
    if is_not_batch:
        selected_keys = torch.concat((
            sorted_key[..., :selected_num],
            sorted_key[..., -selected_num:],
        ))
        selected = flatted[:, selected_keys]
    else:
        selected_keys = torch.concat(
            (
                sorted_key[..., :selected_num],
                sorted_key[..., -selected_num:],
            ),
            dim=1,
        )
        selected = torch.stack([
            flatted[i, :, selected_keys[i]] for i in range(shape[0])
        ])
    # Find the illuminant by pca.
    _, Vt = pca(selected.unsqueeze(-1))  # noqa: N806
    illuminant = Vt[..., 0]
    illuminant = illuminant.contiguous().abs()
    return illuminant
