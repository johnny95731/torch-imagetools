import unittest

from tests.basic import ColorTest
from torch_imagetools.color import rgb_to_yuv, yuv_to_rgb


class YUV(ColorTest):
    def test_yuv_3dim(self):
        self.print_name()

        num = 100
        self.img = self.get_img((3, 512, 512))
        self.fns = [rgb_to_yuv, yuv_to_rgb]

        self.benchmark(num)

    def test_yuv_4dim(self):
        self.print_name()

        num = 30
        self.img = self.get_img((8, 3, 512, 512))
        self.fns = [rgb_to_yuv, yuv_to_rgb]

        self.max_error()
        self.benchmark(num)


if __name__ == '__main__':
    unittest.main()
