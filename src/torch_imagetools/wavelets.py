import torch
import numpy as np


def scaling_coeffs_to_wavelet_coeffs(
    scaling: np.ndarray | torch.Tensor,
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
    # Reverse the order and then multiply -1 to odd order elements.
    # wavelet[i] = (-1)**i * scaling[N - k], where N = scaling.numel() - 1
    wavelet = torch.flip(scaling, dims=(0,))
    odd = wavelet[1::2]
    torch.mul(odd, -1, out=odd)
    return wavelet


def wavelet(img: np.ndarray | torch.Tensor) -> torch.Tensor:
    """_summary_

    Parameters
    ----------
    img : _type_
        _description_
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)


if __name__ == '__main__':
    from timeit import timeit

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    num = 5

    img = mpimg.imread('C:/Users/johnn/OneDrive/桌面/fig/kimblee 12.png')[
        :, :, :3
    ]

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
    l = d8.shape[0]
    d8_wavelet = scaling_coeffs_to_wavelet_coeffs(d8)

    d8_wavelet = d8_wavelet.unsqueeze(1).unsqueeze(0).unsqueeze(0)
    d8_wavelet = d8_wavelet.repeat(3, 1, 1, 1)

    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    h = torch.nn.functional.conv2d(img_t, weight=d8_wavelet, stride=2, groups=3)
    h = torch.clip(h, 0, 1)

    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(h)
    plt.axis('off')
    plt.show()
