import unittest

from test.basic import ColorTest
from torch_imagetools.color.yuv import rgb_to_yuv, yuv_to_rgb


class YUV(ColorTest):
    def test_rgb_and_yuv(self):
        self.print_name()

        num = 20
        self.img = self.get_img((16, 3, 512, 512))

        self.fns = [rgb_to_yuv, yuv_to_rgb]

        self.max_error()
        self.benchmark(num)


if __name__ == '__main__':
    unittest.main()
