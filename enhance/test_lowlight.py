import unittest

from imgtools.enhance import lowlight
from tests.basic import BasicTest, run_over_all_dtype_device


class LowLightEnhance(BasicTest):
    def test_retinex(self):
        self.print_name()

        cases = run_over_all_dtype_device(lowlight.retinex)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(lowlight.retinex, [5, 10])
        for inps, res in cases:
            self._basic_assertion(inps, res)

    def test_msrcr(self):
        self.print_name()

        cases = run_over_all_dtype_device(lowlight.msrcr)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(lowlight.msrcr, [5, 10])
        for inps, res in cases:
            self._basic_assertion(inps, res)

    def test_msrcp(self):
        self.print_name()

        cases = run_over_all_dtype_device(lowlight.msrcp)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(lowlight.msrcp, [5, 10])
        for inps, res in cases:
            self._basic_assertion(inps, res)

    def test_faster_lime(self):
        self.print_name()

        cases = run_over_all_dtype_device(lowlight.faster_lime)
        for inps, res in cases:
            self._basic_assertion(inps, res)


if __name__ == '__main__':
    unittest.main()
