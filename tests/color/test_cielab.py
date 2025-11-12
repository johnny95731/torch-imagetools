import unittest

from src.imgtools.color import (
    lab_to_rgb,
    lab_to_xyz,
    rgb_to_lab,
    rgb_to_xyz,
    xyz_to_lab,
)
from tests.basic import ColorTest


class LAB(ColorTest):
    def test_xyz_lab_3dim(self):
        self.print_name()

        num = 100
        img = self.get_img((3, 512, 512))
        self.img = rgb_to_xyz(img)
        self.fns = [xyz_to_lab, lab_to_xyz]

        self.benchmark(num)

    def test_xyz_lab_4dim(self):
        self.print_name()

        num = 30
        img = self.get_img((8, 3, 512, 512))
        self.img = rgb_to_xyz(img)
        self.fns = [xyz_to_lab, lab_to_xyz]

        self.max_error()
        self.benchmark(num)

    def test_rgb_lab_3dim(self):
        self.print_name()

        num = 100
        self.img = self.get_img((3, 512, 512))
        self.fns = [rgb_to_lab, lab_to_rgb]

        self.benchmark(num)

    def test_rgb_lab_4dim(self):
        self.print_name()

        num = 30
        self.img = self.get_img((8, 3, 512, 512))
        self.fns = [rgb_to_lab, lab_to_rgb]

        self.max_error()
        self.benchmark(num)


if __name__ == '__main__':
    unittest.main()
