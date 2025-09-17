import unittest

import torch

from torch_imagetools.statistics import combine_mean_std


class StatsCombinaiton(unittest.TestCase):
    def test_combination_torch(self, place: int = 2):
        data1 = torch.randn(2, 200)
        sol = [data1.mean(), data1.std(), data1.numel()]
        stats = [(d.mean(), d.std(), d.numel()) for d in data1]
        ret = combine_mean_std(*stats)
        for s, r in zip(sol, ret):
            if torch.is_tensor(s):
                s = s.item()
            if torch.is_tensor(r):
                r = r.item()
            self.assertAlmostEqual(s, r, place)

        data2 = torch.randn(5, 200)
        sol = [data2.mean().item(), data2.std().item(), data2.numel()]
        stats = [(d.mean(), d.std(), d.numel()) for d in data2]
        ret = combine_mean_std(*stats)
        for s, r in zip(sol, ret):
            if torch.is_tensor(s):
                s = s.item()
            if torch.is_tensor(r):
                r = r.item()
            self.assertAlmostEqual(s, r, place)


if __name__ == '__main__':
    unittest.main()
