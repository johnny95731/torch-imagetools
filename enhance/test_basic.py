import unittest

import torch
from imgtools.enhance import basic
from tests.basic import (
    DEFAULT_CONST,
    BasicTest,
    iter_dtype_device,
    run_over_all_dtype_device,
)

BATCH = DEFAULT_CONST['batch']
CHANNEL = DEFAULT_CONST['channel']


class ScalingBase(BasicTest):
    def get_args(self):
        args = (
            1,
            1.0,
            *sum(iter_dtype_device([torch.randn(1)]), []),
            *sum(iter_dtype_device([torch.randn(CHANNEL)]), []),
            *sum(iter_dtype_device([torch.randn(BATCH, 1)]), []),
            *sum(iter_dtype_device([torch.randn(BATCH, CHANNEL)]), []),
        )
        return args

    def test_adjust_linear(self):
        self.print_name()

        args = self.get_args()
        for slope in args:
            for center in args:
                cases = run_over_all_dtype_device(
                    basic.adjust_linear,
                    slope=slope,
                    center=center,
                )
                for inps, res in cases:
                    self._basic_assertion(inps, res)

    def test_adjust_gamma(self):
        self.print_name()

        args = self.get_args()
        for gamma in args:
            for scale in args:
                cases = run_over_all_dtype_device(
                    basic.adjust_gamma,
                    gamma=gamma,
                    scale=scale,
                )
                for inps, res in cases:
                    self._basic_assertion(inps, res)

    def test_adjust_log(self):
        self.print_name()

        args = self.get_args()
        for scale in args:
            cases = run_over_all_dtype_device(
                basic.adjust_log,
                scale=scale,
            )
            for inps, res in cases:
                self._basic_assertion(inps, res)

    def test_adjust_sigmoid(self):
        self.print_name()

        args = self.get_args()
        for shift in args:
            for gain in args:
                cases = run_over_all_dtype_device(
                    basic.adjust_sigmoid,
                    shift=shift,
                    gain=gain,
                )
                for inps, res in cases:
                    self._basic_assertion(inps, res)

    def test_adjust_inverse(self):
        self.print_name()

        args = self.get_args()
        for maxi in args:
            cases = run_over_all_dtype_device(
                basic.adjust_inverse,
                maxi=maxi,
            )
            for inps, res in cases:
                self._basic_assertion(inps, res)


if __name__ == '__main__':
    unittest.main()
