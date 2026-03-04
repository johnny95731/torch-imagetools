__all__ = [
    'combine_mean_std',
    'histogram',
]

import torch

from ..utils.helpers import check_valid_image_ndim


def combine_mean_std(
    *stats: tuple[torch.Tensor, torch.Tensor, int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Calculate the mean, standard deviation (std), and dataset size of the
    combination of two datasets. The

    The function is present for evaluating the mean and std of a large dataset
    by computing its sub-datasets. To see the inference of the formula,
    check [1].

    This function is not jit-able.

    Parameters
    ----------
    stats : tuple[torch.Tensor, torch.Tensor, int]
        The [mean, standard deviation, number of samples] of dataset(s).
        np.ndarray type is also acceptable.

    Returns
    -------
    torch.tensor
        The mean value of the combined dataset.
    torch.tensor
        The standard deviation of the combined dataset.
    int
        The number of samples of the combined dataset.

    References
    ----------
    [1] stack exchange - How do I combine standard deviations of two groups?
        https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
    """
    mean_x, std_x, num_x = stats[0][:3]
    if len(stats) == 1:
        return mean_x, std_x, num_x
    for mean_y, std_y, num_y in stats[1:]:
        num_z = num_x + num_y
        mean_z = (num_x * mean_x + num_y * mean_y) / num_z

        var_x = std_x * std_x
        var_y = std_y * std_y

        part_1 = ((num_x - 1) * var_x + (num_y - 1) * var_y) / (num_z - 1)
        part_2 = (mean_x - mean_y) ** 2 * (
            num_x * num_y / (num_z * (num_z - 1))
        )
        std_z = (part_1 + part_2) ** 0.5
        # Set variable to x
        mean_x = mean_z
        std_x = std_z
        num_x = num_z

    return mean_z, std_z, num_z


def histogram(
    img: torch.Tensor,
    bins: int = 256,
    density: bool = False,
) -> torch.Tensor:
    """Compute the histogram of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image in the range of [0, 1] with 2 <= img.ndim <= 4.
    bins : int, default=256
        The number of groups in data range.
    density : bool, default=False
        If true, return the pdf of each channel.

    Returns
    -------
    torch.Tensor
        The histogram or density.

    Raises
    ------
    TypeError
        If bins is not int type.
    """
    if not isinstance(bins, int):
        raise TypeError(f'`bins` must be an integer: {type(bins)}.')
    check_valid_image_ndim(img, 2)
    img = (img * (bins - 1)).type(torch.uint8)

    flat_image = img.flatten(start_dim=-2).long()
    hist = torch.zeros(
        img.shape[:-2] + (bins,),
        dtype=torch.int32,
        device=img.device,
    )
    hist.scatter_add_(
        dim=-1, index=flat_image, src=hist.new_ones(1).expand_as(flat_image)
    )
    if density:
        num_el = flat_image.size(-1)
        hist = hist.float() / num_el
    return hist
