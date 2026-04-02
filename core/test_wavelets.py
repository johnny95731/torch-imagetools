import unittest

import torch
from tests.basic import (
    BasicTest,
    get_img,
    get_n_img_args,
    iter_dtype_device,
    run_over_all_dtype_device,
)

from imgtools.core import _pywt_wrapping, wavelets


class DWT(BasicTest):
    def test_dwt2(self):
        self.print_name()

        wavelet = _pywt_wrapping.Wavelet('db2')
        arg_case = iter_dtype_device((wavelet.dec_low, wavelet.dec_high))
        for scaling, wavelet in arg_case:
            cases = run_over_all_dtype_device(
                wavelets.dwt2,
                scaling=scaling,
                wavelet=wavelet,
            )
            for inp, res in cases:
                for r in res:
                    self._basic_assertion(inp, r, check_shape=False)

    def test_idwt2(self):
        self.print_name()

        wavelet = _pywt_wrapping.Wavelet('db2')
        arg_case = iter_dtype_device((wavelet.dec_low, wavelet.dec_high))
        for scaling, wavelet in arg_case:
            for dtype in (torch.float32, torch.float64):
                for device in ('cpu', 'cuda'):
                    for inp in get_n_img_args(dtype=dtype, device=device):
                        dec = wavelets.dwt2(
                            inp[0], scaling=scaling, wavelet=wavelet
                        )
                        res = wavelets.idwt2(
                            dec, scaling=scaling, wavelet=wavelet
                        )
                        self._basic_assertion(inp, res, check_shape=False)

    def test_dwt2_partial(self):
        self.print_name()

        wavelet = _pywt_wrapping.Wavelet('db2')
        arg_case = iter_dtype_device((wavelet.dec_low, wavelet.dec_high))

        for ndim in (3, 4):
            img = get_img(ndim=ndim)
            img_cases = iter_dtype_device([img])
            for inp in img_cases:
                inp = inp[0]
                for scaling, wavelet in arg_case:
                    std = wavelets.dwt2(inp, scaling, wavelet)
                    print(inp.shape, std.shape, '\n')
                    for i, c in enumerate(('LL', 'LH', 'HL', 'HH')):
                        res = wavelets.dwt2_partial(
                            inp, scaling, wavelet, target=c
                        )
                        self._basic_assertion((std[..., i, :, :],), res)


if __name__ == '__main__':
    unittest.main()
