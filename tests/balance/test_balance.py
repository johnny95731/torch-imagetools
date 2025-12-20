import unittest

import torch
from src.imgtools.balance import _balance
from src.imgtools.color import get_chromatic_adaptation, get_rgb_to_xyz_matrix
from tests.basic import BasicTest, iter_dtype_device, run_over_all_dtype_device


class VonKries(BasicTest):
    def test_get_matrix(self):
        self.print_name()

        mat = get_rgb_to_xyz_matrix('srgb', 'D65')
        white_d65 = mat.sum(1)
        white_d50 = get_rgb_to_xyz_matrix('srgb', 'D50').sum(1)

        for src, tar in iter_dtype_device([white_d65, white_d50]):
            for adap in (
                'bradford',
                'cat02',
                'cat97s',
                'cam16',
                'hpe',
                'vonkries',
                'xyz',
            ):
                mat = _balance.get_von_kries_transform_matrix(
                    src, tar, method=adap
                )
                self.assertEqual(
                    mat.dtype, src.dtype, f'{mat.dtype} {src.dtype}'
                )
                self.assertEqual(
                    mat.device,
                    src.device,
                    f'{mat.device} {src.device}',
                )
                self.assertEqual(mat.shape, (3, 3), mat.shape)

    def test_transformation(self):
        self.print_name()

        mat = get_rgb_to_xyz_matrix('srgb', 'D65')
        white_d65 = mat.sum(1)
        white_d50 = get_rgb_to_xyz_matrix('srgb', 'D50').sum(1)

        for d65, d50 in iter_dtype_device([white_d65, white_d50]):
            for adap in get_chromatic_adaptation():
                cases = run_over_all_dtype_device(
                    _balance.von_kries_transform,
                    xyz_white=d65,
                    xyz_target_white=d50,
                    method=adap,
                )
                for inp, res in cases:
                    self._basic_assertion(inp, res)


class ScalingBase(BasicTest):
    def test_scaling(self):
        self.print_name()

        scaled_max = [
            1,
            1.0,
            *sum(iter_dtype_device([torch.tensor((1.0))]), []),
            *sum(iter_dtype_device([torch.tensor((1.0, 1.0, 1.0))]), []),
        ]
        for maxi in scaled_max:
            cases = run_over_all_dtype_device(
                _balance.balance_by_scaling,
                scaled_max=maxi,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)

    def test_gray_world(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            _balance.gray_world_balance,
        )
        for inp, res in cases:
            self._basic_assertion(inp, res)

    def test_gray_edge(self):
        self.print_name()

        cases = run_over_all_dtype_device(
            _balance.gray_edge_balance, num_imgs=2
        )
        for inp, res in cases:
            self._basic_assertion(inp, res)

    def test_white_patch(self):
        self.print_name()

        q_coeffs = [
            1,
            1.0,
            *sum(iter_dtype_device([torch.rand(1)]), []),
            *sum(iter_dtype_device([torch.rand(3)]), []),
        ]
        for q in q_coeffs:
            cases = run_over_all_dtype_device(
                _balance.white_patch_balance,
                q=q,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)


class IllumEstBase(BasicTest):
    def test_cheng_pca(self):
        self.print_name()

        for adap in ('rgb', 'von kries'):
            cases = run_over_all_dtype_device(
                _balance.cheng_pca_balance,
                adaptation=adap,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)


if __name__ == '__main__':
    unittest.main()
