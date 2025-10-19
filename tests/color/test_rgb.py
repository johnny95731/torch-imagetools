from timeit import timeit
import unittest

from tests.basic import ColorTest
from torch_imagetools.color import gammaize_rgb, linearize_rgb


class RGB(ColorTest):
    rgbs = (
        'srgb',
        'adobergb',
        'prophotorgb',
        'rec2020',
        'displayp3',
        'widegamut',
        'ciergb',
    )

    def max_error(self, rgb_spec: str):
        img = self.img

        linear = linearize_rgb(img, rgb_spec)
        ret = gammaize_rgb(linear, rgb_spec)
        ret2 = linearize_rgb(ret, rgb_spec)

        diff = (ret - img).abs()
        print(f'Max error {rgb_spec:<11} gamma :', diff.amax())
        diff = (ret2 - linear).abs()
        print(f'Max error {rgb_spec:<11} linear:', diff.amax())

    def benchmark(self, rgb_spec: str, num: int):
        img = self.img

        linear = linearize_rgb(img, rgb_spec)
        gammaize_rgb(linear, rgb_spec)

        kwargs = {
            'number': num,
            'setup': 'from __main__ import linearize_rgb, gammaize_rgb',
            'globals': locals(),
        }
        print(
            f'Timeit {rgb_spec:<11} gamma :',
            timeit('linearize_rgb(img, rgb_spec)', **kwargs) / num,
        )
        print(
            f'Timeit {rgb_spec:<11} linear:',
            timeit('gammaize_rgb(linear, rgb_spec)', **kwargs) / num,
        )

    def test_linearize_error(self):
        self.print_name()

        self.img = self.get_img((8, 3, 512, 512))
        for rgb_spec in self.rgbs:
            self.max_error(rgb_spec)

    def test_linearize_benchmark(self):
        self.print_name()

        num = 100
        self.img = self.get_img((3, 512, 512))
        print('dim = 3')
        for rgb_spec in self.rgbs:
            self.benchmark(rgb_spec, num)

        num = 30
        self.img = self.get_img((8, 3, 512, 512))
        print('dim = 4')
        for rgb_spec in self.rgbs:
            self.benchmark(rgb_spec, num)


if __name__ == '__main__':
    unittest.main()
