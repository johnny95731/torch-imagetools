import unittest

import torch
from src.imgtools import color

from tests.basic import ColorTest, enum_combinations, run_over_all_dtype_device


class XYZ(ColorTest):
    def test_white_point(self):
        self.print_name()

        illum_names = color.get_white_point_names()
        for name in illum_names:
            for obs in (2, 10):
                w1 = color.get_white_point(name, obs)
                w2 = color.get_white_point(name, str(obs))
                self.assertDictEqual(w1, w2)
                self.assertEqual(w1['name'], name)
                self.assertEqual(len(w1['xy']), 2)
                self.assertIsInstance(w1['xy'][0], float)
                self.assertIsInstance(w1['xy'][1], float)
                self.assertEqual(w1['obs'], obs)

    def test_rgb_model(self):
        self.print_name()

        illum_names = color.get_white_point_names()
        rgb_names = color.get_rgb_names()
        for name in rgb_names:
            model = color.get_rgb_model(name)
            self.assertEqual(model['name'], name)
            self.assertIn(model['w'], illum_names)
            for ch in 'rgb':
                self.assertEqual(len(model[ch]), 2)
                self.assertIsInstance(model[ch][0], float)
                self.assertIsInstance(model[ch][1], float)

    def test_matrix(self):
        self.print_name()

        illum_names = color.get_white_point_names()
        rgb_names = color.get_rgb_names()

        cases = enum_combinations(rgb_names, illum_names, (2, 10))
        for rgb_name, iname, obs in cases:
            mats = (
                color.get_rgb_to_xyz_matrix(rgb_name, iname, obs),
                color.get_xyz_to_rgb_matrix(rgb_name, iname, obs),
            )
            s = mats[0].sum(1)
            self.assertAlmostEqual(s[1].item(), 1.0, places=6)

            prod = mats[0].matmul(mats[1])
            id = torch.eye(3)
            d = (prod - id).abs_().max()
            self.assertAlmostEqual(d.item(), 0.0, places=6)

    def test_xyz(self):
        self.print_name()

        cases = run_over_all_dtype_device(color.rgb_to_xyz)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(color.xyz_to_rgb)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        self.img = self.get_img((3, 1000, 1000))
        self.fns = [color.rgb_to_xyz, color.xyz_to_rgb]
        self.max_error()

    def test_normalize_xyz(self):
        self.print_name()

        cases = run_over_all_dtype_device(color.normalize_xyz)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        cases = run_over_all_dtype_device(color.unnormalize_xyz)
        for inps, res in cases:
            self._basic_assertion(inps, res)

        self.img = self.get_img((3, 1000, 1000))
        self.fns = [color.normalize_xyz, color.unnormalize_xyz]
        self.max_error()


if __name__ == '__main__':
    unittest.main()
