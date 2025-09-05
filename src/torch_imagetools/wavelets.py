import torch
import numpy as np

from torch_imagetools.statistics import (
    estimate_noise_from_wavelet,
    estimate_noise_from_wavelet_2,
)
from torch_imagetools.utils.helpers import tensorlize


def scaling_coeffs_to_wavelet_coeffs(
    scaling: np.ndarray | torch.Tensor,
    *_,
    device: torch.DeviceObjType | str | None = None,
) -> torch.Tensor:
    """Calculate coefficients of the wavelet function from given coefficients of
    the scaling function.

    wavelet[i] = (-1)**i * scaling[N - k], where N = scaling.numel() - 1 and
    i = 0, 1, ..., N.

    Parameters
    ----------
    scaling : np.ndarray | torch.Tensor
        coefficients of the scaling function.

    Returns
    -------
    torch.Tensor
        _description_
    """
    if isinstance(scaling, np.ndarray):
        scaling = torch.from_numpy(scaling)
    device = scaling.device if device is None else device
    if scaling.device != device:
        scaling = scaling.to(device)
    # Reverse the order and then multiply -1 to odd order elements.
    # wavelet[i] = (-1)**i * scaling[N - k], where N = scaling.numel() - 1
    wavelet = torch.flip(scaling, dims=(0,))
    odd = wavelet[1::2]
    torch.mul(odd, -1, out=odd)
    return wavelet


def wavelet_hh(
    img: np.ndarray | torch.Tensor,
    wavelet: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Parameters
    ----------
    img : _type_
        _description_
    """
    if (ndim := img.ndim) != 4 and ndim != 3:
        raise ValueError(
            f'Dimention of the image should be 3 or 4, but found {ndim}.'
        )

    img = tensorlize(img)
    single_image = img.ndim == 3
    if single_image:
        img = img.unsqueeze(0)

    wavelet = tensorlize(wavelet)
    if wavelet.device != img.device:
        wavelet = wavelet.to(img.device)
    length = wavelet.numel()
    wavelet = wavelet.view(1, 1, 1, length).repeat(3, 1, 1, 1)

    padding = (length - 1) // 2
    h = torch.nn.functional.conv2d(  # x-direction
        img,
        weight=wavelet,
        stride=(1, 2),
        padding=(0, padding),
        groups=3,
    )
    hh = torch.nn.functional.conv2d(  # y-direction
        h,
        weight=wavelet.transpose(-2, -1),
        stride=(2, 1),
        padding=(padding, 0),
        groups=3,
    )
    if single_image:
        hh = hh.squeeze(0)
    return hh


if __name__ == '__main__':
    from timeit import timeit

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    num = 5

    img = mpimg.imread('C:/Users/johnn/OneDrive/桌面/fig/kimblee 12.png')[
        :, :, :3
    ]
    img = img.astype(np.float32)

    d8 = torch.tensor(
        [
            0.32580343,
            1.01094572,
            0.89220014,
            -0.03957503,
            -0.26450717,
            0.0436163,
            0.0465036,
            -0.01498699,
        ],
        dtype=torch.float32,
    )
    d8_wavelet = scaling_coeffs_to_wavelet_coeffs(d8)

    hh = wavelet_hh(img, d8_wavelet)

    std1 = estimate_noise_from_wavelet(hh)
    std2 = estimate_noise_from_wavelet_2(hh)
    hh = hh.abs().permute(1, 2, 0).numpy()
    print(std1, std2)

    plt.subplot(2, 1, 1)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(hh, cmap='gray')
    plt.axis('off')
    plt.show()
