import unittest

import torch
from src.imgtools.enhance import equlization
from tests.basic import (
    DEFAULT_CONST,
    BasicTest,
    run_over_all_dtype_device,
)

BATCH = DEFAULT_CONST['batch']
CHANNEL = DEFAULT_CONST['channel']


class Equalization(BasicTest):
    def _assert_result(
        self,
        inps: list[torch.Tensor],
        res: torch.Tensor,
    ):
        inp = inps[0]
        if inp.ndim == res.ndim:
            self.assertEqual(res.shape, inp.shape)
        elif inp.ndim == res.ndim - 1:
            self.assertEqual(res.shape[1:], inp.shape)
            self.assertEqual(res.shape[0], BATCH)
        self._basic_assertion(inps, res, check_shape=False)

    def test_hist_equalize(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            equlization.hist_equalize,
        )
        for inps, res in cases:
            self._assert_result(inps, res)

    def test_match_mean_std(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            equlization.match_mean_std, num_imgs=2
        )
        for inps, res in cases:
            self._assert_result(inps, res)


if __name__ == '__main__':
    unittest.main()
