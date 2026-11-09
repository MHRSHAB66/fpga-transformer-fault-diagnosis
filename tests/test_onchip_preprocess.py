"""Integer on-FPGA preprocessing reference (src/quantize/onchip_preprocess.py)."""
import shutil
import tempfile
import unittest

import numpy as np

from src.data.prepare_dataset import normalize_windows_array
from src.quantize.onchip_preprocess import (CODE_MAX, CODE_MIN, channel_factors,
                                            preprocess_window, to_adc_codes)

S_IN = 0.028626115354773118


class OnchipPreprocessTest(unittest.TestCase):
    def test_constant_and_zero_channels_give_zero_codes(self):
        w = np.zeros((1024, 8), dtype=np.int64)
        w[:, 2] = 777
        self.assertFalse(preprocess_window(w, S_IN).any())
        self.assertEqual(channel_factors(0, S_IN), (0, 0))

    def test_matches_float_zscore_path(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=(16, 1024, 8)) * rng.uniform(0.02, 40, size=(16, 1, 8))
        x = x.astype(np.float32)
        fs = [256.0] * 8                      # no clipping: |x| stays below 200
        ref = np.clip(np.round(normalize_windows_array(x).astype(np.float64) / S_IN), -511, 511)
        got = np.stack([preprocess_window(c, S_IN) for c in to_adc_codes(x, fs)])
        diff = np.abs(got - ref)
        self.assertLessEqual(int(diff.max()), 1)
        self.assertLess(float((diff > 0).mean()), 0.005)

    def test_out_of_range_words_are_clamped(self):
        rng = np.random.default_rng(1)
        w = rng.integers(-1000, 1000, size=(1024, 8))
        wide = w.copy()
        wide[0, :] = 1 << 30
        clamped = w.copy()
        clamped[0, :] = CODE_MAX
        np.testing.assert_array_equal(preprocess_window(wide, S_IN), preprocess_window(clamped, S_IN))
        self.assertEqual(to_adc_codes(np.array([[1e9, -1e9]]), [1.0, 1.0]).tolist(), [[CODE_MAX, CODE_MIN]])

    @unittest.skipUnless(shutil.which("g++"), "g++ not on PATH")
    def test_cpp_ingress_is_bit_exact_on_edge_cases(self):
        from tools.verify_onchip_preprocess import build, edge_windows, reference, run_cpp
        w = edge_windows(np.random.default_rng(7))
        with tempfile.TemporaryDirectory() as tmp:
            np.testing.assert_array_equal(run_cpp(build(tmp), w, tmp), reference(w))


if __name__ == "__main__":
    unittest.main()
