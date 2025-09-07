import sys
from typing import Callable
import unittest
from timeit import timeit

import torch


class ColorTest(unittest.TestCase):
    img: torch.Tensor
    fns: tuple[Callable, Callable]

    def get_img(self, shape=(16, 3, 512, 512)):
        img = torch.randint(0, 256, shape)
        img = img.type(torch.float32) / 255
        return img

    def print_name(self):
        function_name = sys._getframe(1).f_code.co_name
        print(function_name)

    def max_error(self, place: int = 5):
        img = self.img
        trans, trans_inv = self.fns[:2]

        res1 = trans(img)
        res2 = trans_inv(res1)
        res3 = trans(res2)

        diff1 = torch.abs(res2 - img)
        self.assertAlmostEqual(torch.max(diff1).item(), 0.0, 5, '123')
        print('Max error after 2 map:', torch.max(diff1).item())
        diff2 = torch.abs(res3 - res1)
        print('Max error after 3 map:', torch.max(diff2).item())

    def benchmark(self, num: int):
        img = self.img
        trans, trans_inv = self.fns[:2]

        temp = trans(img)
        kwargs = {
            'number': num,
            'globals': locals(),
        }
        print(f'Timeit {trans.__name__}:', timeit('trans(img)', **kwargs))
        print(
            f'Timeit {trans_inv.__name__}:', timeit('trans_inv(temp)', **kwargs)
        )
