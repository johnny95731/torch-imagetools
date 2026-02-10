import unittest

from tests.basic import ColorTest
from imgtools.color import gammaize_rgb, linearize_rgb, get_rgb_names


class RGB(ColorTest):
    def max_error(self, rgb_spec: str):
        img = self.img

        linear = linearize_rgb(img, rgb_spec)
        ret = gammaize_rgb(linear, rgb_spec)
        ret2 = linearize_rgb(ret, rgb_spec)

        diff = (ret - img).abs()
        print(f'Max error {rgb_spec:<11} gamma :', diff.amax())
        diff = (ret2 - linear).abs()
        print(f'Max error {rgb_spec:<11} linear:', diff.amax())

    def test_linearize_error(self):
        self.print_name()

        self.img = self.get_img((8, 3, 512, 512))
        for rgb_spec in get_rgb_names():
            self.max_error(rgb_spec)


if __name__ == '__main__':
    unittest.main()
