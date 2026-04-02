import unittest

from tests.basic import (
    BasicTest,
    enum_combinations,
    get_img,
    iter_dtype_device,
    run_over_all_dtype_device,
)

from imgtools.filters import _edges


class Edge(BasicTest):
    def test_gradient_magnitude(self):
        self.print_name()

        magnitudes = (1, 1.0, 'inf', '-inf')

        for ndim in (3, 4):
            x = get_img(ndim=ndim)
            y = get_img(ndim=ndim)
            cases = iter_dtype_device([y, x])
            for inp in cases:
                for mag in magnitudes:
                    res = _edges.gradient_magnitude(*inp, magnitude=mag)
                    self._basic_assertion(inp, res)

                res = _edges.gradient_magnitude(*inp, magnitude='stack')
                self.assertEqual(res.shape[1:], inp[0].shape)
                self.assertEqual(res.shape[0], 2)
                self._basic_assertion(inp, res, check_shape=False)

    def test_laplacian(self):
        self.print_name()

        _bool = (True, False)
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_bool, _bool, _modes)

        for diagonal, inflection_only, mode in arg_cases:
            cases = run_over_all_dtype_device(
                _edges.laplacian,
                diagonal=diagonal,
                inflection_only=inflection_only,
                mode=mode,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)

    def test_robinson(self):
        self.print_name()

        _angle_unit = ('deg', 'rad')
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_angle_unit, _modes)

        for mode in _modes:
            cases = run_over_all_dtype_device(
                _edges.robinson,
                mode=mode,
            )
            for inp, mag in cases:
                self._basic_assertion(inp, mag)
        #
        for angle_unit, mode in arg_cases:
            cases = run_over_all_dtype_device(
                _edges.robinson,
                ret_angle=True,
                angle_unit=angle_unit,
                mode=mode,
            )
            for inp, (mag, angle) in cases:
                self._basic_assertion(inp, mag)
                self._basic_assertion(inp, angle)

    def test_kirsch(self):
        self.print_name()

        _angle_unit = ('deg', 'rad')
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(_angle_unit, _modes)

        for mode in _modes:
            cases = run_over_all_dtype_device(
                _edges.kirsch,
                mode=mode,
            )
            for inp, mag in cases:
                self._basic_assertion(inp, mag)
        #
        for angle_unit, mode in arg_cases:
            cases = run_over_all_dtype_device(
                _edges.kirsch,
                ret_angle=True,
                angle_unit=angle_unit,
                mode=mode,
            )
            for inp, (mag, angle) in cases:
                self._basic_assertion(inp, mag)
                self._basic_assertion(inp, angle)

    def test_prewitt(self):
        self.print_name()

        magnitudes = (1, 1.0, 'inf', '-inf', 'stack')
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(magnitudes, _modes)

        for mag, mode in arg_cases:
            cases = run_over_all_dtype_device(
                _edges.prewitt,
                magnitude=mag,
                mode=mode,
            )
            for inp, res in cases:
                if mag != 'stack':
                    self._basic_assertion(inp, res)
                else:
                    self.assertEqual(res.shape[1:], inp[0].shape)
                    self.assertEqual(res.shape[0], 2)
                    self._basic_assertion(inp, res, check_shape=False)
            #
            cases = run_over_all_dtype_device(
                _edges.prewitt,
                magnitude=mag,
                ret_angle=True,
                mode=mode,
            )
            for inp, (res, angle) in cases:
                if mag != 'stack':
                    self._basic_assertion(inp, res)
                else:
                    self.assertEqual(res.shape[1:], inp[0].shape)
                    self.assertEqual(res.shape[0], 2)
                    self._basic_assertion(inp, res, check_shape=False)
                self._basic_assertion(inp, angle)

    def test_sobel(self):
        self.print_name()

        magnitudes = (1, 1.0, 'inf', '-inf', 'stack')
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(magnitudes, _modes)

        for mag, mode in arg_cases:
            cases = run_over_all_dtype_device(
                _edges.sobel,
                magnitude=mag,
                mode=mode,
            )
            for inp, res in cases:
                if mag != 'stack':
                    self._basic_assertion(inp, res)
                else:
                    self.assertEqual(res.shape[1:], inp[0].shape)
                    self.assertEqual(res.shape[0], 2)
                    self._basic_assertion(inp, res, check_shape=False)
            #
            cases = run_over_all_dtype_device(
                _edges.sobel,
                magnitude=mag,
                ret_angle=True,
                mode=mode,
            )
            for inp, (res, angle) in cases:
                if mag != 'stack':
                    self._basic_assertion(inp, res)
                else:
                    self.assertEqual(res.shape[1:], inp[0].shape)
                    self.assertEqual(res.shape[0], 2)
                    self._basic_assertion(inp, res, check_shape=False)
                self._basic_assertion(inp, angle)

    def test_scharr(self):
        self.print_name()

        magnitudes = (1, 1.0, 'inf', '-inf', 'stack')
        _modes = ('constant', 'reflect', 'replicate', 'circular')
        arg_cases = enum_combinations(magnitudes, _modes)

        for mag, mode in arg_cases:
            cases = run_over_all_dtype_device(
                _edges.scharr,
                magnitude=mag,
                mode=mode,
            )
            for inp, res in cases:
                if mag != 'stack':
                    self._basic_assertion(inp, res)
                else:
                    self.assertEqual(res.shape[1:], inp[0].shape)
                    self.assertEqual(res.shape[0], 2)
                    self._basic_assertion(inp, res, check_shape=False)
            #
            cases = run_over_all_dtype_device(
                _edges.scharr,
                magnitude=mag,
                ret_angle=True,
                mode=mode,
            )
            for inp, (res, angle) in cases:
                if mag != 'stack':
                    self._basic_assertion(inp, res)
                else:
                    self.assertEqual(res.shape[1:], inp[0].shape)
                    self.assertEqual(res.shape[0], 2)
                    self._basic_assertion(inp, res, check_shape=False)
                self._basic_assertion(inp, angle)


if __name__ == '__main__':
    unittest.main()
