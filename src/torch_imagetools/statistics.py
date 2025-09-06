import numpy as np
import torch

from torch_imagetools.utils.helpers import tensorlize

TensorLike = torch.Tensor | np.ndarray


def combine_mean_std(
    stats1: tuple[TensorLike, TensorLike, int],
    stats2: tuple[TensorLike, TensorLike, int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Calculate the mean, standard deviation (std), and dataset size of the
    combination of two datasets.

    Parameters
    ----------
    stats1 : tuple[TensorLike, TensorLike, int]
        The [mean value, standard deviation, number of samples] of dataset 1.
    stats2 : tuple[TensorLike, TensorLike, int]
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
    std_z = torch.sqrt(part_1 + part_2)

    return mean_z, std_z, num_z


def estimate_noise_from_wavelet(
    hh: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Estimates the standard deviation of Gaussian noise from the 

     For details, see:
     D. L. Donoho and I. M. Johnstone, “Ideal spatial adaptation by wavelet
     shrinkage,” Biometrika, vol. 81, no. 3, pp. 425-455, Sep. 1994

    Parameters
    ----------
    hh : torch.Tensor | np.ndarray
        The highpass-highpass filtered image.

    Returns
    -------
    torch.Tensor
        _description_
    """
    hh = tensorlize(hh)
    hh = torch.abs(hh)
    if hh.ndim == 3:
        sigma_est = torch.median(hh)
    else:
        hh = hh.view(hh.size(0), -1)
        sigma_est = torch.median(hh, dim=1).values

    sigma_est *= 255 / 0.6745
    return sigma_est


def estimate_noise_from_wavelet_2(
    hh: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    sigma_est = estimate_noise_from_wavelet(hh)
    poly_coeffs = [
        -1.707,
        1.383,
        -2.784e-2,
        8.695e-4,
        -1.092e-5,
        5.404e-8,
    ]

    sigma = 0
    pow = 1.0
    for coeff in poly_coeffs:
        sigma += coeff * pow
        pow *= sigma_est
    return sigma
