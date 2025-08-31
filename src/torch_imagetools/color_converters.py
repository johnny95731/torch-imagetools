import torch
import numpy as np


def rgb_to_yuv(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a YUV image.

    The input is assumed to be in the range of [0, 1]. The output will be
    normalized to [0, 1] (for Y channel) and [-0.5, 0.5] (for U and V channels)

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        A floating dtype RGB image in the range of [0, 1]. For a ndarray, the
        shape should be (H, W, 3) or (N, H, W, 3). For a Tensor, the shape
        should be (3, H, W) or (N, 3, H, W).

    Returns
    -------
    torch.Tensor
        YUV image with shape (3, H, W) or (N, 3, H, W). The range of Y
        is [0, 1] and the range of U and V are [-0.5, 0.5].
    """
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)

    # fmt: off
    matrix = torch.tensor(
        [[ 0.299,  0.587,  0.114],
         [-0.169, -0.331,  0.500],
         [ 0.500, -0.419, -0.081]],
        dtype=torch.float32,
        device=rgb.device
    )
    # fmt: on
    if rgb.dim() == 4:
        rgb = rgb.permute(0, 2, 3, 1)
        device = rgb.device
        if device.type == 'cuda':
            yuv = torch.matmul(rgb, matrix.T)
        else:
            yuv = torch.empty_like(rgb)
            for i, b in enumerate(rgb):
                torch.matmul(b, matrix.T, out=yuv[i])

        yuv = yuv.permute(0, 3, 1, 2)
    else:
        rgb = rgb.permute(1, 2, 0)
        yuv = torch.empty_like(rgb)

        torch.matmul(rgb, matrix.T, out=yuv)

        yuv = yuv.permute(2, 0, 1)
    return yuv


def rgb_to_yuv2(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a YUV image.

    The input is assumed to be in the range of [0, 1]. The output will be
    normalized to [0, 1] (for Y channel) and [-0.5, 0.5] (for U and V channels)

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        A floating dtype RGB image in the range of [0, 1]. For a ndarray, the
        shape should be (H, W, 3) or (N, H, W, 3). For a Tensor, the shape
        should be (3, H, W) or (N, 3, H, W).

    Returns
    -------
    torch.Tensor
        YUV image with shape (3, H, W) or (N, 3, H, W). The range of Y
        is [0, 1] and the range of U and V are [-0.5, 0.5].
    """
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(0, 3, 1, 2) if rgb.dim() == 4 else rgb.permute(2, 0, 1)

    r: torch.Tensor = rgb[..., 0, :, :]
    g: torch.Tensor = rgb[..., 1, :, :]
    b: torch.Tensor = rgb[..., 2, :, :]

    y = 0.299 * r
    torch.add(y, 0.587 * g, out=y)
    torch.add(y, 0.114 * b, out=y)
    cb = torch.sub(b, y)
    torch.mul(0.564, cb, out=cb)
    cr = torch.sub(r, y)
    torch.mul(0.713, cr, out=cr)
    # y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    # cb: torch.Tensor = -0.169 * r + -0.331 * g + 0.500 * b + 0.5
    # cr: torch.Tensor = 0.500 * r + -0.419 * g + -0.081 * b + 0.5

    return torch.stack((y, cb, cr), dim=-3)


def yuv_to_rgb(yuv: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver a YUV image to an RGB image.

    The input is assumed to be in the range of [0, 1] (for Y channel) and
    [-0.5, 0.5] (for U and V channels). The output will be normalized to
    [0, 1] (for Y channel) and [-0.5, 0.5] (for U and V channels).

    Parameters
    ----------
    yuv : np.ndarray | torch.Tensor
        A floating dtype YUV image. For a ndarray, the shape should be
        (H, W, 3) or (N, H, W, 3). For a Tensor, the shape should be
        (3, H, W) or (N, 3, H, W).

    Returns
    -------
    torch.Tensor
        RGB image with shape (3, H, W) or (N, 3, H, W). The range of channels
        will be normalized to [0, 1].
    """
    if isinstance(yuv, np.ndarray):
        yuv = torch.from_numpy(yuv)

    # fmt: off
    matrix = torch.tensor(
        [[ 1.0, -0.00093, 1.401687],
         [ 1.0, -0.3437, -0.71417],
         [ 1.0,  1.77216, 0.00099]],
        dtype=torch.float32,
        device=yuv.device
    )
    # fmt: on
    if yuv.dim() == 4:
        yuv = yuv.permute(0, 2, 3, 1)
        device = yuv.device
        if device.type == 'cuda':
            rgb = torch.matmul(yuv, matrix.T)
        else:
            rgb = torch.empty_like(yuv)
            for i, b in enumerate(yuv):
                torch.matmul(b, matrix.T, out=rgb[i])

        rgb = rgb.permute(0, 3, 1, 2)
    else:
        yuv = yuv.permute(1, 2, 0)
        rgb = torch.empty_like(yuv)

        torch.matmul(yuv, matrix.T, out=rgb)

        rgb = rgb.permute(2, 0, 1)
    torch.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb


def srgb_to_srgb_linear(srgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a YUV image.

    The input is assumed to be in the range of [0, 1]. The output will be
    normalized to [0, 1] (for Y channel) and [-0.5, 0.5] (for U and V channels)

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        A floating dtype RGB image in the range of [0, 1]. For a ndarray, the
        shape should be (H, W, 3) or (N, H, W, 3). For a Tensor, the shape
        should be (3, H, W) or (N, 3, H, W).

    Returns
    -------
    torch.Tensor
        YUV image with shape (3, H, W) or (N, 3, H, W). The range of Y
        is [0, 1] and the range of U and V are [-0.5, 0.5].
    """
    if isinstance(srgb, np.ndarray):
        srgb = torch.from_numpy(srgb)

    srgb_linear = torch.zeros_like(srgb)
    mask_leq = srgb <= 0.04045
    srgb_linear[mask_leq] = srgb[mask_leq] * (1 / 12.92)

    mask_gt = torch.bitwise_not(mask_leq)
    higher = srgb[mask_gt] + 0.055
    torch.mul(higher, 1 / 1.055, out=higher)
    torch.pow(higher, 2.4, out=higher)
    srgb_linear[mask_gt]
    print(srgb_linear)
    return srgb_linear


def rgb_to_xyz(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Conver an RGB image to a YUV image.

    The input is assumed to be in the range of [0, 1]. The output will be
    normalized to [0, 1] (for Y channel) and [-0.5, 0.5] (for U and V channels)

    Parameters
    ----------
    rgb : np.ndarray | torch.Tensor
        A floating dtype RGB image in the range of [0, 1]. For a ndarray, the
        shape should be (H, W, 3) or (N, H, W, 3). For a Tensor, the shape
        should be (3, H, W) or (N, 3, H, W).

    Returns
    -------
    torch.Tensor
        YUV image with shape (3, H, W) or (N, 3, H, W). The range of Y
        is [0, 1] and the range of U and V are [-0.5, 0.5].
    """
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)

    # fmt: off
    matrix = torch.tensor(
        [[ 0.299,  0.587,  0.114],
         [-0.169, -0.331,  0.500],
         [ 0.500, -0.419, -0.081]],
        dtype=torch.float32,
        device=rgb.device
    )
    # fmt: on
    if rgb.dim() == 4:
        rgb = rgb.permute(0, 2, 3, 1)
        device = rgb.device
        if device.type == 'cuda':
            xyz = torch.matmul(rgb, matrix.T)
        else:
            xyz = torch.empty_like(rgb)
            for i, b in enumerate(rgb):
                torch.matmul(b, matrix.T, out=xyz[i])

        xyz = xyz.permute(0, 3, 1, 2)
    else:
        rgb = rgb.permute(1, 2, 0)
        xyz = torch.empty_like(rgb)

        torch.matmul(rgb, matrix.T, out=xyz)

        xyz = xyz.permute(2, 0, 1)
    return xyz


if __name__ == '__main__':
    from timeit import timeit

    img = np.random.randint(0, 256, (1024, 1024, 3)).astype(np.float32) / 255
    img = torch.randint(0, 256, (128, 3, 2048, 2048)).type(torch.float32) / 255
    num = 5

    print(srgb_to_srgb_linear(img))

    # yuv1 = rgb_to_yuv(img)
    # yuv2 = rgb_to_yuv2(img)
    # print(torch.max(torch.abs(yuv1 - yuv2)))
    # print(torch.max(torch.abs(img - yuv_to_rgb(yuv1))))

    # print(timeit('rgb_to_yuv(img)', number=num, globals=locals()))
    # print(timeit('rgb_to_yuv2(img)', number=num, globals=locals()))
