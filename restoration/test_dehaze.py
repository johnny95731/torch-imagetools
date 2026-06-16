import unittest

from imgtools.restoration import dehaze
from tests.basic import (
    DEFAULT_CONST,
    BasicTest,
    run_over_all_dtype_device,
)

BATCH = DEFAULT_CONST['batch']
CHANNEL = DEFAULT_CONST['channel']


class Dehaze(BasicTest):
    def test_adjust_linear(self):
        self.print_name()

        cases = run_over_all_dtype_device(dehaze.color_attenuation_dehaze)
        for inps, res in cases:
            self._basic_assertion(inps, res)

    def test_adjust_gamma(self):
        self.print_name()

        cases = run_over_all_dtype_device(dehaze.dark_channel_dehaze)
        for inps, res in cases:
            self._basic_assertion(inps, res)


if __name__ == '__main__':
    unittest.main()
