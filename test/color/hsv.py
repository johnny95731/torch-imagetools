import unittest

from test.basic import ColorTest
from torch_imagetools.color.hsv import rgb_to_hsv, hsv_to_rgb


class HSV(ColorTest):
    def test_rgb_and_yuv(self):
        self.print_name()

        num = 20
        self.img = self.get_img((16, 3, 512, 512))

        self.fns = [rgb_to_hsv, hsv_to_rgb]

        self.max_error()
        self.benchmark(num)


if __name__ == '__main__':
    unittest.main()
