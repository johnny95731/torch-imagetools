import unittest

from test.basic import ColorTest
from torch_imagetools.color.cielab import (
    lab_to_rgb,
    lab_to_xyz,
    rgb_to_lab,
    rgb_to_xyz,
    xyz_to_lab,
)


class LAB(ColorTest):
    def test_xyz_and_lab(self):
        self.print_name()

        num = 10
        img = self.get_img((16, 3, 512, 512))
        self.img = rgb_to_xyz(img)

        self.fns = [xyz_to_lab, lab_to_xyz]

        self.max_error()
        self.benchmark(num)

    def test_rgb_and_lab(self):
        self.print_name()

        num = 10
        self.img = self.get_img((16, 3, 512, 512))

        self.fns = [rgb_to_lab, lab_to_rgb]

        self.max_error()
        self.benchmark(num)


if __name__ == '__main__':
    unittest.main()
