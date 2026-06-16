import unittest

import torch
from tests.basic import (
    DEFAULT_CONST,
    BasicTest,
    enum_combinations,
    get_img,
    iter_dtype_device,
    run_over_all_dtype_device,
)

from imgtools.enhance import contrast

BATCH = DEFAULT_CONST['batch']


class GammaCorrection(BasicTest):
    def get_args(self):
        args = (
            1,
            1.0,
            *sum(iter_dtype_device([torch.rand(1)]), []),
            *sum(iter_dtype_device([torch.rand(BATCH, 1)]), []),
        )
        return args

    def test_auto_gamma_correction(self):
        self.print_name()

        args = self.get_args()
        for target in args:
            cases = run_over_all_dtype_device(
                contrast.auto_gamma_correction,
                target=target,
            )
            for inps, res in cases:
                self._basic_assertion(inps, res)

    def test_local_gamma_correction(self):
        self.print_name()

        args = self.get_args()
        args = enum_combinations(args, args)
        for gain, basic_gamma in args:
            cases = run_over_all_dtype_device(
                contrast.local_gamma_correction,
                gain=gain,
                basic_gamma=basic_gamma,
            )
            for inps, res in cases:
                self._basic_assertion(inps, res)

    def test_lide(self):
        self.print_name()

        args = (None, *self.get_args())
        inps = enum_combinations(
            ('gauss', 'laplace'),
            args,
            args,
            sum(iter_dtype_device([get_img(None, 3)]), [])
            + sum(iter_dtype_device([get_img(None, 4)]), []),
        )
        for model, std_min, std_max, img in inps:
            if (
                isinstance(std_min, torch.Tensor)
                and std_min.ndim + 2 > img.ndim
            ) or (
                isinstance(std_max, torch.Tensor)
                and std_max.ndim + 2 > img.ndim
            ):
                continue
            res = contrast.lide(img, std_min, std_max, model=model)
            self._basic_assertion([img], res)


if __name__ == '__main__':
    unittest.main()
