import unittest

from src.imgtools.wavelets import _pywt_wrapping
from tests.basic import BasicTest


class Wavelet(BasicTest):
    def test_get_families(self):
        self.print_name()

        full = _pywt_wrapping.get_families(False)
        short = _pywt_wrapping.get_families(True)
        self.assertEqual(len(short), len(full))

    def test_kernels(self):
        self.print_name()

        for wavelet_name in _pywt_wrapping.get_wavelets():
            obj = _pywt_wrapping.Wavelet(wavelet_name)
            self.assertEqual(len(obj.dec_low), len(obj.dec_high))
            self.assertEqual(len(obj.rec_low), len(obj.rec_high))


if __name__ == '__main__':
    unittest.main()
