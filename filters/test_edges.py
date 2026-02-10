import unittest

from imgtools.filters import _edges
from tests.basic import (
    BasicTest,
    get_img,
    iter_dtype_device,
    run_over_all_dtype_device,
)


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

        for diagonal in _bool:
            for inflection_only in _bool:
                cases = run_over_all_dtype_device(
                    _edges.laplacian,
                    diagonal=diagonal,
                    inflection_only=inflection_only,
                )
                for inp, res in cases:
                    self._basic_assertion(inp, res)

    def test_robinson(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            _edges.robinson,
        )
        for inp, res in cases:
            self._basic_assertion(inp, res)

        _angle_unit = ('deg', 'rad')
        for angle_unit in _angle_unit:
            cases = run_over_all_dtype_device(
                _edges.robinson,
                ret_angle=True,
                angle_unit=angle_unit,
            )
            for inp, (mag, angle) in cases:
                self._basic_assertion(inp, mag)
                self._basic_assertion(inp, angle)

    def test_kirsch(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            _edges.kirsch,
        )
        for inp, res in cases:
            self._basic_assertion(inp, res)

        _angle_unit = ('deg', 'rad')
        for angle_unit in _angle_unit:
            cases = run_over_all_dtype_device(
                _edges.kirsch,
                ret_angle=True,
                angle_unit=angle_unit,
            )
            for inp, (mag, angle) in cases:
                self._basic_assertion(inp, mag)
                self._basic_assertion(inp, angle)


if __name__ == '__main__':
    unittest.main()
