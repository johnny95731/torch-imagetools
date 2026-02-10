import unittest

from imgtools.filters import blur
from tests.basic import BasicTest, run_over_all_dtype_device


class Blurring(BasicTest):
    def test_box_blur(self):
        self.print_name()

        _ksize = (3, (3, 3))

        for ksize in _ksize:
            cases = run_over_all_dtype_device(
                blur.box_blur,
                ksize=ksize,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)

    def test_gaussian_blurring(self):
        self.print_name()

        _ksize = (3, (3, 3), 0)
        _sigma = (1, (1, 1))

        for ksize in _ksize:
            for sigma in _sigma:
                cases = run_over_all_dtype_device(
                    blur.gaussian_blur,
                    ksize=ksize,
                    sigma=sigma,
                )
                for inp, res in cases:
                    self._basic_assertion(inp, res)
        for ksize in _ksize[:-1]:
            cases = run_over_all_dtype_device(
                blur.gaussian_blur,
                ksize=ksize,
                sigma=0,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)

    def test_guided_filter(self):
        self.print_name()

        _ksize = (3, (3, 3))
        _eps = (0.001, 0.05)

        for ksize in _ksize:
            for eps in _eps:
                cases = run_over_all_dtype_device(
                    blur.guided_filter, ksize=ksize, eps=eps, num_imgs=2
                )
                for inp, res in cases:
                    self._basic_assertion(inp, res)

    def test_max_min_filter(self):
        self.print_name()

        _ksize = (3, (3, 3))

        for ksize in _ksize:
            cases = run_over_all_dtype_device(blur.max_filter, ksize=ksize)
            for inp, res in cases:
                self._basic_assertion(inp, res)
        for ksize in _ksize:
            cases = run_over_all_dtype_device(blur.min_filter, ksize=ksize)
            for inp, res in cases:
                self._basic_assertion(inp, res)


if __name__ == '__main__':
    unittest.main()
