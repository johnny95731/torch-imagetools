import unittest

from test.basic import ColorTest
from torch_imagetools.color.ciexyz import rgb_to_xyz
from torch_imagetools.color.cieluv import (
    luv_to_rgb,
    luv_to_xyz,
    rgb_to_luv,
    xyz_to_luv,
)


class LAB(ColorTest):
    def test_xyz_luv_3dim(self):
        self.print_name()

        num = 100
        img = self.get_img((3, 512, 512))
        self.img = rgb_to_xyz(img)
        self.fns = [xyz_to_luv, luv_to_xyz]

        self.benchmark(num)

    def test_xyz_luv_4dim(self):
        self.print_name()

        num = 30
        img = self.get_img((8, 3, 512, 512))
        self.img = rgb_to_xyz(img)
        self.fns = [xyz_to_luv, luv_to_xyz]

        self.max_error()
        self.benchmark(num)

    def test_rgb_luv_3dim(self):
        self.print_name()

        num = 100
        self.img = self.get_img((3, 512, 512))
        self.fns = [rgb_to_luv, luv_to_rgb]

        self.benchmark(num)

    def test_rgb_luv_4dim(self):
        self.print_name()

        num = 30
        self.img = self.get_img((8, 3, 512, 512))
        self.fns = [rgb_to_luv, luv_to_rgb]

        self.max_error()
        self.benchmark(num)


if __name__ == '__main__':
    unittest.main()
