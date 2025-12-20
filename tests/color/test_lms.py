import unittest

from src.imgtools import color

from tests.basic import ColorTest, run_over_all_dtype_device


class LMS(ColorTest):
    def test_hsi(self):
        self.print_name()

        cases = run_over_all_dtype_device(color.xyz_to_lms)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(color.lms_to_xyz)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        self.img = self.get_img((3, 1000, 1000))
        self.fns = [color.xyz_to_lms, color.lms_to_xyz]
        self.max_error()


if __name__ == '__main__':
    unittest.main()
