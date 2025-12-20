import unittest

from src.imgtools.balance import est_illuminant
from tests.basic import BasicTest, run_over_all_dtype_device


class IllumEstimation(BasicTest):
    def test_illuminant_cheng(self):
        self.print_name()

        n_selected = [1, 1.0]
        for c in n_selected:
            cases = run_over_all_dtype_device(
                est_illuminant.estimate_illuminant_cheng,
                n_selected=c,
            )
            for inp, res in cases:
                self._basic_assertion(inp, res)


if __name__ == '__main__':
    unittest.main()
