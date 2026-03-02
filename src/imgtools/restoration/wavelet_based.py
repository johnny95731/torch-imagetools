__all__ = [
    'estimate_noise_from_wavelet',
    'estimate_noise_from_wavelet_2',
]

import torch


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
