import unittest

from imgtools import color

from tests.basic import (
    ColorTest,
    get_img,
    iter_dtype_device,
    run_over_all_dtype_device,
)


class Grayscale(ColorTest):
    def test_grayscale(self):
        self.print_name()

        cases = run_over_all_dtype_device(color.rgb_to_gray)
        for inps, res in cases:
            self._basic_assertion(inps, res, check_shape=False)

        gray = get_img((2, 1, 5, 7))
        cases = iter_dtype_device([gray])
        for inps in cases:
            res = color.gray_to_rgb(*inps)
            self._basic_assertion(inps, res, check_shape=False)


if __name__ == '__main__':
    unittest.main()
