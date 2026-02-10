import unittest

from imgtools.balance import light
from tests.basic import BasicTest, run_over_all_dtype_device


class Compensation(BasicTest):
    def test_light_htchen(self):
        self.print_name()

        spaces = ('LAB', 'LUV', 'YUV', 'HSV', 'HSL', 'HSL', 'HSI')
        _overflow = ('clip', 'norm', 'both')

        for space in spaces:
            for overflow in _overflow:
                cases = run_over_all_dtype_device(
                    light.light_compensation_htchen,
                    space=space,
                    overflow=overflow,
                )
                for inp, res in cases:
                    self._basic_assertion(inp, res)


if __name__ == '__main__':
    unittest.main()
