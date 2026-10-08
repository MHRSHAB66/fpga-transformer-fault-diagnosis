"""Tests for the SEQ=256 INT10 PYNQ driver contract (pynq.fault_classifier).

The lightweight tests (quantization + AXI-stream serialization order) run
anywhere. The bit-exact NumpyBackend golden check is skipped automatically if
torch or the checkpoint is unavailable, since it reproduces the hardware using
the trained INT10 model.
"""
import unittest
from pathlib import Path

import numpy as np

from pynq.fault_classifier import (
    quantize_window, load_input_scale, SEQ_IN, CH, ACT_QMAX, INPUT_WORDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "hls" / "f1_int10_seq256" / "golden_ingress1024"

try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False
_HAVE_CKPT = (REPO_ROOT / "artifacts" / "experiments" / "transformer_f1_seq256_pool4"
              / "kaggle_v3" / "seq256_pool4" / "scratch" / "best.pt").exists()


class QuantizeContractTests(unittest.TestCase):
    def test_input_scale_matches_manifest(self):
        self.assertAlmostEqual(load_input_scale(), 0.028626115354773118, places=15)

    def test_shape_and_clamp(self):
        scale = load_input_scale()
        w = np.zeros((SEQ_IN, CH), dtype=np.float64)
        w[0, 0] = 1e9        # forces positive clamp
        w[0, 1] = -1e9       # forces negative clamp
        codes = quantize_window(w, scale)
        self.assertEqual(codes.shape, (SEQ_IN, CH))
        self.assertEqual(codes.dtype, np.int64)
        self.assertEqual(codes[0, 0], ACT_QMAX)
        self.assertEqual(codes[0, 1], -ACT_QMAX)
        self.assertTrue((np.abs(codes) <= ACT_QMAX).all())

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            quantize_window(np.zeros((256, 8)), load_input_scale())

    def test_axi_stream_serialization_is_row_major_sample_channel(self):
        # word index must be sample*CH + channel; check the ravel contract the
        # HardwareBackend relies on (self._in[:] = codes.reshape(SEQ_IN,CH).ravel()).
        codes = np.arange(SEQ_IN * CH, dtype=np.int64).reshape(SEQ_IN, CH)
        flat = codes.reshape(SEQ_IN, CH).ravel()
        self.assertEqual(flat.size, INPUT_WORDS)
        self.assertEqual(flat[0], 0)
        self.assertEqual(flat[CH], codes[1, 0])       # start of sample 1
        self.assertEqual(flat[8 * CH + 3], codes[8, 3])


@unittest.skipUnless(_HAVE_TORCH and _HAVE_CKPT,
                     "NumpyBackend needs torch + the trained checkpoint")
class NumpyBackendGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pynq.fault_classifier import FaultClassifier, NumpyBackend
        cls.clf = FaultClassifier(backend=NumpyBackend())
        lines = (GOLDEN / "golden_in.txt").read_text().strip().splitlines()
        n, seq, ch = map(int, lines[0].split())
        cls.windows = [np.array(lines[1 + i].split(), dtype=np.int64).reshape(seq, ch)
                       for i in range(n)]
        cls.expected = []
        for ln in (GOLDEN / "golden_expected.txt").read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()
            cls.expected.append((int(p[2]), list(map(int, p[3:7]))))

    def test_reproduces_ingress_golden_bit_exact(self):
        matched = 0
        for i, w in enumerate(self.windows):
            cls_id, logits = self.clf.infer_from_codes(w)
            if logits.tolist() == self.expected[i][1] and cls_id == self.expected[i][0]:
                matched += 1
        self.assertEqual(matched, len(self.windows),
                         f"NumpyBackend reproduced {matched}/{len(self.windows)} golden vectors")


if __name__ == "__main__":
    unittest.main()
