import os
import sys
import unittest
from collections.abc import Callable
from functools import reduce
from operator import mul
from pathlib import Path
from timeit import timeit
from typing import Any, Generator, Iterable, Literal

import torch


def path_to_module_path(file_path: str) -> str:
    """Convert string from './path/to/file' to 'path.to.file'"""
    rel_path = Path(file_path).relative_to('./')
    if rel_path.suffix == '.py':
        rel_path = rel_path.with_suffix('')
    parts = rel_path.parts
    module_path = '.'.join(parts)
    return module_path


def run_module(module: str):
    print(f'Run `{module}`.')
    os.system(f'{os.getcwd()}/.venv/Scripts/python.exe -m {module}')


DEFAULT_CONST = {
    'batch': 2,
    'channel': 3,
}


def enum_combinations(*args: Iterable[Any]):
    arg_counts = tuple(map(lambda x: len(x), args))
    gen_length = reduce(mul, arg_counts)  # type: int
    indices = [0 for _ in args]

    for _ in range(gen_length):
        res = [t[j] for j, t in zip(indices, args)]
        yield res

        indices[0] += 1
        for j, n in enumerate(arg_counts[:-1]):
            if indices[j] >= n:
                indices[j] -= n
                indices[j + 1] += 1


def get_img(
    shape: tuple[int, ...] = None,
    ndim: Literal[3, 4] = 3,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
):
    invalid_shape = not hasattr(shape, '__len__') or len(shape) != ndim
    if ndim == 3:
        if invalid_shape:
            shape = (DEFAULT_CONST['channel'], 50, 70)
    elif ndim == 4:
        if invalid_shape:
            shape = (DEFAULT_CONST['batch'], DEFAULT_CONST['channel'], 50, 70)

    img = torch.randint(0, 256, shape, dtype=dtype, device=device)
    img.mul_(1 / 255)
    return img


def get_n_img_args(
    num_arg: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
):
    """Yield specific number of images with ndim=3 or ndim=4.
    Generate `2**num_arg` times in totally.
    """
    if not str(num_arg).isdigit():
        raise ValueError(f'`num_arg` must be a positive integer.')
    gen_length = 2**num_arg

    for idx in range(gen_length):
        args: list[torch.Tensor] = []
        for j in range(num_arg):
            ndim = 3 if (idx >> j) & 1 == 0 else 4
            args.append(get_img(None, ndim, dtype, device))
        yield args


def iter_dtype_device(
    tensors: list[torch.Tensor],
    dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
    devices: tuple[torch.device, ...] = ('cpu', 'cuda'),
) -> Generator[list[torch.Tensor], Any, None]:
    """Returns a generator iterator that generate tensors over given
    dtypes and devices

    Parameters
    ----------
    tensors : list[torch.Tensor]
        Tensors.
    dtypes : tuple[torch.dtype, ...], optional
        The data types that will be generated.
        By default (torch.float32, torch.float64)
    devices : tuple[torch.device, ...], optional
        The device that will be generated. By default ('cpu', 'cuda').

    Yields
    ------
    list[torch.Tensor]
        Input tensors in different dtype and devices.
    """
    n_dtypes = len(dtypes)
    n_devices = len(devices)
    n = n_dtypes + n_devices

    gen_length = (n_dtypes ** len(tensors)) * (n_devices ** len(tensors))
    indices = [0 for _ in tensors]

    for _ in range(gen_length):
        args: list[torch.Tensor] = []

        for j, t in zip(indices, tensors):
            idx_device, idx_dtype = divmod(j, n_dtypes)
            args.append(t.to(devices[idx_device], dtypes[idx_dtype]))
        yield args

        indices[0] += 1
        for j, t in enumerate(tensors[:-1]):
            if indices[j] >= n:
                indices[j] -= n
                indices[j + 1] += 1


def run_over_all_dtype_device(
    fn,
    *args,
    num_imgs: int = 1,
    dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
    devices: tuple[torch.device, ...] = ('cpu', 'cuda'),
    **kwargs,
):
    gen_img_length = 2**num_imgs

    for idx in range(gen_img_length):
        inps: list[torch.Tensor] = []
        for j in range(num_imgs):
            ndim = 3 if (idx >> j) & 1 == 0 else 4
            inps.append(get_img(None, ndim))
        for imgs in iter_dtype_device(inps, dtypes, devices):
            res = fn(*imgs, *args, **kwargs)
            yield imgs, res


def get_batch(
    t: int | float | torch.Tensor,
    ttype: Literal['img', 'coeff'] = 'img',
):
    if isinstance(t, torch.Tensor) and (
        (ttype == 'img' and t.ndim == 4) or (ttype == 'coeff')
    ):
        batch = t.size(0)
    else:
        batch = 0
    return batch


def get_max_batch(
    tensors: list[int | float | torch.Tensor],
    ttype: Literal['img', 'coeff'] = 'img',
):
    batch = max(map(lambda t: get_batch(t, ttype), tensors))
    return batch


class BasicTest(unittest.TestCase):
    def print_name(self):
        function_name = sys._getframe(1).f_code.co_name
        print(function_name)

    def benchmark(self, num: int):
        img = self.img
        trans, trans_inv = self.fns[:2]

        temp = trans(img)
        trans_inv(temp)
        kwargs = {'number': num, 'globals': locals()}
        print(f'Timeit {trans.__name__}:', timeit('trans(img)', **kwargs) / num)
        print(
            f'Timeit {trans_inv.__name__}:',
            timeit('trans_inv(temp)', **kwargs) / num,
        )

    def _basic_assertion(
        self,
        inps: list[torch.Tensor],
        res: torch.Tensor,
        check_shape: bool = True,
    ):
        inp = inps[0]
        if check_shape:
            self.assertIn(inp.ndim, (3, 4))
            if inp.ndim == res.ndim:
                self.assertEqual(res.shape, inp.shape)
            elif inp.ndim == res.ndim - 1:
                batch = get_max_batch(inps[1:])
                self.assertEqual(res.shape[1:], inp.shape)
                self.assertEqual(res.shape[0], batch)
        self.assertEqual(res.dtype, inp.dtype)
        self.assertEqual(res.device, inp.device)


class ColorTest(BasicTest):
    img: torch.Tensor
    fns: tuple[Callable[[torch.Tensor], torch.Tensor], ...]

    def get_img(
        self,
        shape=(8, 3, 512, 512),
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = 'cpu',
    ):
        img = get_img(shape, len(shape), dtype, device)
        return img

    def max_error(self):
        img = self.img
        trans, trans_inv = self.fns[:2]

        res1 = trans(img)
        res2 = trans_inv(res1)
        res3 = trans(res2)

        reduced = tuple(range(img.ndim))
        reduced = reduced[:-3] + reduced[-2:]
        diff1 = (res2 - img).abs_()
        print('Max error  BAx -  x:', diff1.amax(dim=reduced))
        diff2 = (res3 - res1).abs_()
        print('Max error ABAx - Ax:', diff2.amax(dim=reduced))
