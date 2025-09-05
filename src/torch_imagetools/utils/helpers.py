import numpy as np
import torch


def arraylize(item: float | list[float] | np.ndarray | torch.Tensor):
    if isinstance(item, np.ndarray) or torch.is_tensor(item):
        return item
    return np.array(item)


def tensorlize(
    img: float | list[float] | np.ndarray | torch.Tensor,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    """Convert an item to torch.Tensor. If it is a np.ndarray, then the shape
    will be assume to be (H, W, C) or (N, H, W, C) and permute to (C, H, W).

    Parameters
    ----------
    item : float | list[float] | np.ndarray | torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        Tensor with shape (C, H, W) or (N, C, H, W).
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

        img = (
            img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            if img.ndim == 3
            else img.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            if img.ndim == 4
            else img.unsqueeze(0)  # (H, W) -> (1, H, W)
            if img.ndim == 2
            else img
        )
    elif not torch.is_tensor(img):
        img = torch.tensor(img)
    # Check attributes
    if dtype is not None and img.dtype != dtype:
        img = img.type(dtype)
    if device is not None and img.device != device:
        img = img.to(device)
    return img


def matrix_transform(
    img: torch.Tensor, matrix: torch.Tensor, *, out: torch.Tensor | None = None
):
    img = img.movedim(-3, -1)  # To (H, W, C) or (N, H, W, C)
    output = (
        torch.empty(
            (*img.shape[:-1], matrix.shape[-1]),
            dtype=img.dtype,
            device=img.device,
        )
        if out is None
        else out
    )
    if img.ndim == 4:
        for b, b_out in zip(img, output):
            torch.matmul(b, matrix, out=b_out)
    else:
        torch.matmul(img, matrix, out=output)
    output = output.movedim(-1, -3)  # To (C, H, W) or (N, C, H, W)
    return output


def matrix_transform_(img: torch.Tensor, matrix: torch.Tensor):
    img = img.movedim(-3, -1)  # To (H, W, C) or (N, H, W, C)
    if img.ndim == 4:
        for b in img:
            torch.matmul(b, matrix, out=b)
    else:
        torch.matmul(img, matrix, out=img)
    img = img.movedim(-1, -3)  # To (C, H, W) or (N, C, H, W)
    return img
