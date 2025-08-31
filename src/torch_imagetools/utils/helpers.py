import numpy as np
import torch


def arraylize(item: float | list[float] | np.ndarray | torch.Tensor):
    if isinstance(item, np.ndarray) or torch.is_tensor(item):
        return item
    return np.array(item)


def tensorlize(item: float | list[float] | np.ndarray | torch.Tensor):
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item)
    elif torch.is_tensor(item):
        return item
    return torch.tensor(item)
