import torch
import numpy as np

from ..utils.helpers import matrix_transform, tensorlize


def rgb_to_gray(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a grayscale image.

    The input is assumed to be in the range of [0, 1]. The output will be in
    the range of [0, 1]

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        A floating dtype RGB image in the range of [0, 1]. For a ndarray, the
        shape should be (H, W, 3) or (N, H, W, 3). For a Tensor, the shape
        should be (3, H, W) or (N, 3, H, W).

    Returns
    -------
    torch.Tensor
        YUV image with shape (1, H, W) or (N, 1, H, W). The range of Y
        is [0, 1] and the range of U and V are [-0.5, 0.5].
    """
    rgb = tensorlize(rgb)

    # fmt: off
    matrix = torch.tensor(
        [[ 0.299],  [0.587],  [0.114]],
        dtype=torch.float32,
        device=rgb.device
    )
    # fmt: on
    gray = matrix_transform(rgb, matrix)
    return gray


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 30

    g1 = rgb_to_gray(img)

    print(timeit('rgb_to_gray(img)', number=num, globals=locals()))
