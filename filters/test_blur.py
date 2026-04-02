import unittest

from tests.basic import BasicTest, enum_combinations, run_over_all_dtype_device

from imgtools.filters import blur


class Blurring(BasicTest):
    def test_box_blur(self):
        self.print_name()

        _ksize = (3, (3, 3))
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_ksize, _modes)

        for ksize, mode in arg_cases:
            cases = run_over_all_dtype_device(
                blur.box_blur,
                ksize=ksize,
                mode=mode,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)

    def test_gaussian_blurring(self):
        self.print_name()

        _ksize = (3, (3, 3), 0)
        _sigma = (1, (1, 1))
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_ksize, _sigma, _modes)

        for ksize, sigma, mode in arg_cases:
            cases = run_over_all_dtype_device(
                blur.gaussian_blur,
                ksize=ksize,
                sigma=sigma,
                mode=mode,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)
        arg_cases = enum_combinations(_ksize[:-1], _modes)
        for ksize, mode in arg_cases:
            cases = run_over_all_dtype_device(
                blur.gaussian_blur,
                ksize=ksize,
                sigma=0,
                mode=mode,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)

    def test_guided_filter(self):
        self.print_name()

        _ksize = (3, (3, 3))
        _eps = (0.001, 0.05)
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_ksize, _eps, _modes)

        for ksize, eps, mode in arg_cases:
            # img only
            cases = run_over_all_dtype_device(
                blur.guided_filter,
                ksize=ksize,
                eps=eps,
                mode=mode,
                num_imgs=1,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)
            # img + guidance
            cases = run_over_all_dtype_device(
                blur.guided_filter,
                ksize=ksize,
                eps=eps,
                mode=mode,
                num_imgs=2,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)

    def test_max_min_filter(self):
        self.print_name()

        _ksize = (3, (3, 3))
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_ksize, _modes)

        for ksize, mode in arg_cases:
            cases = run_over_all_dtype_device(
                blur.max_filter,
                ksize=ksize,
                mode=mode,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)
            cases = run_over_all_dtype_device(
                blur.min_filter,
                ksize=ksize,
                mode=mode,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)


if __name__ == '__main__':
    unittest.main()
