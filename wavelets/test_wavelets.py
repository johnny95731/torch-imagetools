import unittest

from imgtools.wavelets import wavelets, _pywt_wrapping
from tests.basic import (
    BasicTest,
    get_img,
    iter_dtype_device,
    run_over_all_dtype_device,
)


class DWT(BasicTest):
    def test_dwt(self):
        self.print_name()

        wavelet = _pywt_wrapping.Wavelet('db2')
        arg_case = iter_dtype_device((wavelet.dec_low, wavelet.dec_high))
        for scaling, wavelet in arg_case:
            cases = run_over_all_dtype_device(
                wavelets.dwt,
                scaling=scaling,
                wavelet=wavelet,
            )
            for inp, res in cases:
                for r in res:
                    self._basic_assertion(inp, r, check_shape=False)

    def test_dwt_partial(self):
        self.print_name()

        wavelet = _pywt_wrapping.Wavelet('db2')
        arg_case = iter_dtype_device((wavelet.dec_low, wavelet.dec_high))

        for ndim in (3, 4):
            img = get_img(ndim=ndim)
            img_cases = iter_dtype_device([img])
            for inp in img_cases:
                inp = inp[0]
                for scaling, wavelet in arg_case:
                    std = wavelets.dwt(inp, scaling, wavelet)
                    for i, c in enumerate(('LL', 'LH', 'HL', 'HH')):
                        res = wavelets.dwt_partial(
                            inp, scaling, wavelet, target=c
                        )
                        self._basic_assertion((std[i],), res)


if __name__ == '__main__':
    unittest.main()
