import unittest

from src.imgtools import color

from tests.basic import ColorTest, run_over_all_dtype_device


class LAB(ColorTest):
    def test_xyz_lab(self):
        self.print_name()

        cases = run_over_all_dtype_device(color.xyz_to_lab)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(color.lab_to_xyz)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        self.img = self.get_img((3, 1000, 1000))
        self.fns = [color.xyz_to_lab, color.lab_to_xyz]
        self.max_error()

    def test_rgb_lab(self):
        self.print_name()

        cases = run_over_all_dtype_device(color.rgb_to_lab)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(color.lab_to_rgb)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        self.img = self.get_img((3, 1000, 1000))
        self.fns = [color.rgb_to_lab, color.lab_to_rgb]
        self.max_error()


if __name__ == '__main__':
    unittest.main()
