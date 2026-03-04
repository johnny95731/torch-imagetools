__all__ = [
    'pca',
]


import torch
from ..utils.helpers import check_valid_image_ndim


def pca(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Image PCA.

    Parameters
    ----------
    img : torch.Tensor
        Image with shape `(*, C, H, W)`

    Returns
    -------
    L : torch.Tensor
        Eigenvalues in ascending order.
    Vt : torch.Tensor
        Corresponding eigenvectors.
    """
    check_valid_image_ndim(img)
    is_float16 = img.dtype == torch.float16
    if is_float16 or not torch.is_floating_point(img):
        img = img.float()
    flatted = img.flatten(-2)
    # Covariance
    mean = flatted.mean(dim=-1, keepdim=True)
    cov = (flatted @ flatted.movedim(-1, -2)) / (flatted.size(-1) - 1)
    cov -= mean * mean.movedim(-1, -2)

    L, Vt = torch.linalg.eigh(cov)  # noqa: N806
    if is_float16:
        L = L.type(torch.float16)  # noqa: N806
        Vt = Vt.type(torch.float16)  # noqa: N806
    return L, Vt
