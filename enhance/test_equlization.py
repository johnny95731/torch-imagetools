import unittest

import torch
from imgtools.enhance import equlization
from tests.basic import (
    DEFAULT_CONST,
    BasicTest,
    run_over_all_dtype_device,
)

BATCH = DEFAULT_CONST['batch']
CHANNEL = DEFAULT_CONST['channel']


class Equalization(BasicTest):
    def test_hist_equalize(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            equlization.hist_equalize,
        )
        for inps, res in cases:
            self._basic_assertion(inps, res)

    def test_hist_equalize(self):
        self.print_name()

        _bins = (255, 300)
        for bins in _bins:
            hist = torch.randn(bins)
            hist = hist / hist.sum()
            cases = run_over_all_dtype_device(
                equlization.match_historgram,
                hist,
                bins,
            )
            for inps, res in cases:
                self._basic_assertion(inps, res)

    def test_match_mean_std(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            equlization.match_mean_std, num_imgs=2
        )
        for inps, res in cases:
            self._basic_assertion(inps, res)


if __name__ == '__main__':
    unittest.main()
