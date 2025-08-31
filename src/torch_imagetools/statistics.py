import torch

from .utils.helpers import tensorlize


def combine_mean_std(
    stats1: tuple[torch.tensor, torch.tensor, int],
    stats2: tuple[torch.tensor, torch.tensor, int],
) -> tuple[torch.tensor, torch.tensor, int]:
    """The combination

    Parameters
    ----------
    stats1 : tuple[torch.tensor, torch.tensor, int]
        The [mean value, standard deviation, number of samples] of dataset 1.
    stats2 : tuple[torch.tensor, torch.tensor, int]
        The [mean value, standard deviation, number of samples] of dataset 2.

    Returns
    -------
    tuple[torch.tensor, torch.tensor, int]
        The [mean value, standard deviation, number of samples] of the
        combination of datasets.
    """
    mean_x, std_x, num_x = stats1[:3]
    mean_y, std_y, num_y = stats2[:3]

    mean_x = tensorlize(mean_x)
    std_x = tensorlize(std_x)
    mean_y = tensorlize(mean_y)
    std_y = tensorlize(std_y)

    num_z = num_x + num_y
    mean_z = (num_x * mean_x + num_y * mean_y) / num_z

    var_x = mean_x * mean_x
    var_y = mean_y * mean_y

    part_1 = ((num_x - 1) * var_x + (num_y - 1) * var_y) / (num_z - 1)
    part_2 = (mean_x - mean_y) ** 2 * (num_x * num_y / (num_z * (num_z - 1)))
    std_z = (part_1 + part_2) ** 0.5

    return mean_z, std_z, num_z
