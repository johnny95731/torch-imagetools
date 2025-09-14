import torch

from ..utils.helpers import matrix_transform


def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to grayscale.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image. For a ndarray, the shape should be (*, H, W, 3). For a
        Tensor, the shape should be (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An grayscale image with shape (*, 1, H, W). The maximum is the same as
        the input.
    """
    matrix = torch.tensor(
        ((0.299, 0.587, 0.114),),
        device=rgb.device,
    )
    gray = matrix_transform(rgb, matrix)
    return gray


def gray_to_rgb(gray: torch.Tensor) -> torch.Tensor:
    """Converts an image from grayscale to rgb space.

    Parameters
    ----------
    gray : torch.Tensor
        An grayscale image with shape (*, 1, H, W).

    Returns
    -------
    torch.Tensor
        An RGB image with shape (*, 3, H, W).
    """
    matrix = torch.tensor(((1.0,), (1.0,), (1.0,)), device=gray.device)
    rgb = matrix_transform(gray, matrix)
    # rgb = torch.cat((gray, gray, gray), dim=-3)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 30

    g1 = rgb_to_gray(img)
    grgb = gray_to_rgb(g1)

    print(timeit('rgb_to_gray(img)', number=num, globals=locals()))
    print(timeit('gray_to_rgb(g1)', number=num, globals=locals()))
