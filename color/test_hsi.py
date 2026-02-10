import unittest

from imgtools import color

from tests.basic import ColorTest, run_over_all_dtype_device


class HSI(ColorTest):
    def test_hsi(self):
        self.print_name()

        cases = run_over_all_dtype_device(color.rgb_to_hsi)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(color.hsi_to_rgb)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        self.img = self.get_img((3, 1000, 1000))
        self.fns = [color.rgb_to_hsi, color.hsi_to_rgb]
        self.max_error()


if __name__ == '__main__':
    unittest.main()
