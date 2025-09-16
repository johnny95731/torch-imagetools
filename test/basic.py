import sys
from typing import Callable
import unittest
from timeit import timeit

import torch


class ColorTest(unittest.TestCase):
    img: torch.Tensor
    fns: tuple[Callable[[torch.Tensor], torch.Tensor], ...]

    def get_img(self, shape=(8, 3, 512, 512)):
        img = torch.randint(0, 256, shape, dtype=torch.float32).mul_(1 / 255)
        return img

    def print_name(self):
        function_name = sys._getframe(1).f_code.co_name
        print(function_name)

    def max_error(self):
        img = self.img
        trans, trans_inv = self.fns[:2]

        res1 = trans(img)
        res2 = trans_inv(res1)
        res3 = trans(res2)

        reduced = tuple(range(img.ndim))
        reduced = reduced[:-3] + reduced[-2:]
        diff1 = (res2 - img).abs()
        print('Max error  BAx -  x:', diff1.amax(dim=reduced))
        diff2 = (res3 - res1).abs()
        print('Max error ABAx - Ax:', diff2.amax(dim=reduced))

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
