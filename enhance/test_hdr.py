import unittest

from imgtools.enhance import hdr
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
        sigma_c = 0.5
        _sigma_s = (2, None)
        _downsample = (1, 0.5)
        _kernel = ('huber', 'lorentz', 'turkey', 'gaussian')

        for sigma_s in _sigma_s:
            for downsample in _downsample:
                for kernel in _kernel:
                    cases = run_over_all_dtype_device(
                        hdr.bilateral_hdr,
                        sigma_c=sigma_c,
                        sigma_s=sigma_s,
                        downsample=downsample,
                        edge_stopping=kernel,
                    )
                    for inps, res in cases:
                        self._basic_assertion(inps, res)


if __name__ == '__main__':
    unittest.main()
