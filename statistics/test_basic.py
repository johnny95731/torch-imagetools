import unittest

import torch
from tests.basic import (
    BasicTest,
    enum_combinations,
    get_max_batch,
    run_over_all_dtype_device,
)

from imgtools.statistics import basic


class Blurring(BasicTest):
    def test_moving_stats(self):
        self.print_name()

        _ksize = (3, (3, 3))
        _fft_approx = (False, True)
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_ksize, _fft_approx, _modes)
        fns = [basic.moving_mean, basic.moving_var, basic.moving_mean_std]

        for ksize, fft_approx, mode in arg_cases:
            kwargs = {
                'sigma' if fft_approx else 'ksize': ksize,
                'fft_approx': fft_approx,
                'mode': mode,
            }
            for fn in fns:
                cases = run_over_all_dtype_device(fn, **kwargs)
                for inp, res in cases:
                    if isinstance(res, tuple):
                        self._basic_assertion(inp, res[0])
                        self._basic_assertion(inp, res[1])
                    else:
                        self._basic_assertion(inp, res)

    def _assert_stats(
        self, inps: list[torch.Tensor], res: torch.Tensor, channelwise: bool
    ):
        inp = inps[0]
        self.assertIn(res.ndim, (3, 4))
        if channelwise:
            self.assertEqual(res.size(-3), inp.size(-3))
            self.assertEqual(res.size()[-2:], (1, 1))
        else:
            self.assertEqual(res.size()[-3:], (1, 1, 1))
        if inp.ndim == res.ndim - 1:
            batch = get_max_batch(inps[1:])
            self.assertEqual(res.shape[0], batch)
        self.assertEqual(res.dtype, inp.dtype)
        self.assertEqual(res.device, inp.device)

    def test_stats(self):
        self.print_name()

        _channelwise = (False, True)
        fns = [basic.mean, basic.std, basic.mean_std]

        for channelwise in _channelwise:
            for fn in fns:
                cases = run_over_all_dtype_device(fn, channelwise=channelwise)
                for inp, res in cases:
                    if isinstance(res, tuple):
                        self._assert_stats(inp, res[0], channelwise)
                        self._assert_stats(inp, res[1], channelwise)
                    else:
                        self._assert_stats(inp, res, channelwise)
            # Covariance
            cases = run_over_all_dtype_device(
                basic.covar, channelwise=channelwise, num_imgs=2
            )
            for inp, res in cases:
                self._assert_stats(inp, res, channelwise)
        # covariance matrix
        cases = run_over_all_dtype_device(basic.covar_matrix)
        for inp, res in cases:
            inp = inp[0]
            self.assertEqual(res.size()[-2:], (inp.size(-3),) * 2)
            self.assertEqual(res.dtype, inp.dtype)
            self.assertEqual(res.device, inp.device)


if __name__ == '__main__':
    unittest.main()
