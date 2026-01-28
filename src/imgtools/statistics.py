__all__ = [
    'combine_mean_std',
    'estimate_noise_from_wavelet',
    'estimate_noise_from_wavelet_2',
]

import torch

from .utils.helpers import check_valid_image_ndim


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


def estimate_noise_from_wavelet(hh: torch.Tensor) -> float:
    """Estimates the standard deviation of Gaussian noise from the
    highpass-highpass component of the wavelet decomposition.

    Parameters
    ----------
    hh : torch.Tensor
        The highpass-highpass filtered image with shape `(*, 1, H, W)`.

    Returns
    -------
    float
        The standard deviation of Gaussian noise in the image.

    References
    ----------
    [1] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by wavelet
        shrinkage," Biometrika, vol. 81, no. 3, pp. 425-455, Sep. 1994
    """
    hh = torch.abs(hh)
    if hh.ndim <= 3:
        sigma_est = torch.median(hh)
    else:
        num = int(hh.shape[-3] * hh.shape[-2] * hh.shape[-1])
        hh = hh.view(-1, num)
        sigma_est = torch.median(hh, dim=1).values

    sigma_est = sigma_est.item() / 0.6745
    return sigma_est


def estimate_noise_from_wavelet_2(
    hh: torch.Tensor,
    maximum: float | int = 1.0,
) -> float:
    """Estimates the standard deviation of Gaussian noise from the
    highpass-highpass component of the wavelet decomposition.

    An advanced method of `estimate_noise_from_wavelet`, for details, see:


    Parameters
    ----------
    hh : torch.Tensor
        The highpass-highpass filtered image in the range of [0, 1] with shape
        `(*, 1, H, W)`.
    maximum : float | int, default=1.0
        The maximum of the image.

    Returns
    -------
    float
        The standard deviation of Gaussian noise in the image.

    References
    ----------
    [1] V. M. Kamble and K. Bhurchandi, "Noise Estimation and Quality
        Assessment of Gaussian Noise Corrupted Images," IOP Conference
        Series Materials Science and Engineering, vol. 331, p. 012019,
        Mar. 2018, doi: 10.1088/1757-899x/331/1/012019.
    """
    sigma_est = estimate_noise_from_wavelet(hh) * (255.0 / maximum)
    poly_coeffs = (
        -1.707,
        1.383,
        -2.784e-2,
        8.695e-4,
        -1.092e-5,
        5.404e-8,
    )

    sigma = 0.0
    pow = 1.0
    for coeff in poly_coeffs:
        sigma += coeff * pow
        pow *= sigma_est
    sigma = sigma if sigma >= 0 else 0.0
    return sigma
