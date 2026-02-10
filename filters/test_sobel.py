import unittest

from imgtools.filters import _sobel
from tests.basic import BasicTest, run_over_all_dtype_device


class Sobel(BasicTest):
    def test_sobel(self):
        self.print_name()

        magnitudes = (1, 1.0, 'inf', '-inf', 'stack')
        for mag in magnitudes:
            cases = run_over_all_dtype_device(_sobel.sobel, magnitude=mag)
            for inp, res in cases:
                if mag != 'stack':
                    self._basic_assertion(inp, res)
                else:
                    self.assertEqual(res.shape[1:], inp[0].shape)
                    self.assertEqual(res.shape[0], 2)
                    self._basic_assertion(inp, res, check_shape=False)


if __name__ == '__main__':
    unittest.main()
