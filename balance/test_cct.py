import unittest

import torch
from imgtools.balance import _cct
from tests.basic import BasicTest, iter_dtype_device


class IllumEstimation(BasicTest):
    def test_mccamy_approximation(self):
        self.print_name()

        _xy_cases = [
            torch.rand((2)),
            torch.rand((2, 10)),
            torch.rand((2, 4, 15)),
        ]
        for xy in _xy_cases:
            cases = iter_dtype_device([xy])
            for inp in cases:
                res = _cct.mccamy_approximation(*inp)
                self.assertEqual(res.shape, inp[0].shape[1:])
                self._basic_assertion(inp, res, check_shape=False)

    def test_hernandez_andre_approximation(self):
        self.print_name()

        _xy_cases = [
            torch.rand((2)),
            torch.rand((2, 10)),
            torch.rand((2, 4, 15)),
        ]
        for xy in _xy_cases:
            cases = iter_dtype_device([xy])
            for inp in cases:
                res = _cct.hernandez_andre_approximation(*inp)
                self.assertEqual(res.shape, inp[0].shape[1:])
                self._basic_assertion(inp, res, check_shape=False)


if __name__ == '__main__':
    unittest.main()
