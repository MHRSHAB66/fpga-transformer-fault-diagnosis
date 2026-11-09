"""Repository driver: optional FPGA class register (argmax-variant IP) -- no board.

Exercises pynq/fault_classifier.py with a fake vendor `pynq`:
  * legacy overlay (no `pred_class` register): unchanged behaviour, class from
    np.argmax of the FPGA logits, class_source == "ps_argmax_legacy";
  * argmax-variant overlay: class read from the HWH-located register, checked
    against the lowest-index argmax of the same logits (ties included);
  * a register value disagreeing with the logits, or a missing ap_vld bit, is
    an error -- never silently replaced.
These validate driver logic only, not synthesis or physical hardware.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "pynq" / "fault_classifier.py"
CLASS_OFF, VLD_OFF = 0x10, 0x14


def _fake_pynq(registers, out_logits, reg_values):
    class Chan:
        def __init__(self, name):
            self.name, self.buf = name, None

        def transfer(self, buf):
            self.buf = buf

        def wait(self):
            if self.name == "recv":
                self.buf[:] = np.asarray(out_logits, dtype=self.buf.dtype)

    class DMA:
        sendchannel = Chan("send")
        recvchannel = Chan("recv")

    class IP:
        def write(self, off, val):
            pass

        def read(self, off):
            return reg_values[off]

    ip_dict = {"axi_dma_0": {}, "transformer_top_0": {"registers": registers}}

    class Overlay:
        def __init__(self, bit):
            self.ip_dict = ip_dict
            self.axi_dma_0 = DMA()
            self.transformer_top_0 = IP()

    mod = types.ModuleType("pynq")
    mod.Overlay = Overlay
    mod.allocate = lambda shape, dtype: np.zeros(shape, dtype=dtype)
    return mod


def _load(fake):
    sys.modules["pynq"] = fake
    spec = importlib.util.spec_from_file_location("repo_fault_classifier_hwclass", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ARGMAX_REGS = {"pred_class": {"address_offset": CLASS_OFF},
               "pred_class_ctrl": {"address_offset": VLD_OFF}}


class HwClassRegisterTests(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("pynq")
        self.bit = REPO_ROOT / "tmp" / "test_hwclass" / "x.bit"
        self.bit.parent.mkdir(parents=True, exist_ok=True)
        self.bit.write_bytes(b"0")
        self.bit.with_suffix(".hwh").write_bytes(b"0")
        self.codes = np.zeros((1024, 8), dtype=np.int64)

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("pynq", None)
        else:
            sys.modules["pynq"] = self._saved

    def _clf(self, regs, logits, values):
        m = _load(_fake_pynq(regs, logits, values))
        return m, m.FaultClassifier(backend=m.HardwareBackend(str(self.bit)))

    def test_legacy_overlay_unchanged(self):
        m, clf = self._clf({}, [5, 9, 9, -3], {})
        self.assertEqual(clf.class_source, "ps_argmax_legacy")
        cls, lg = clf.infer_from_codes(self.codes)
        self.assertEqual(cls, 1)                      # tie 1/2 -> lowest index
        self.assertEqual(lg.tolist(), [5, 9, 9, -3])

    def test_register_class_used_for_every_class(self):
        for c in range(4):
            lg = [-100] * 4
            lg[c] = 50
            m, clf = self._clf(ARGMAX_REGS, lg, {CLASS_OFF: c, VLD_OFF: 1})
            self.assertEqual(clf.class_source, "fpga_register")
            self.assertEqual(clf.infer_from_codes(self.codes)[0], c)

    def test_register_tie_policy(self):
        m, clf = self._clf(ARGMAX_REGS, [7, 7, 7, 7], {CLASS_OFF: 0, VLD_OFF: 1})
        self.assertEqual(clf.infer_from_codes(self.codes)[0], 0)
        m, clf = self._clf(ARGMAX_REGS, [1, 3, 2, 3], {CLASS_OFF: 1, VLD_OFF: 1})
        self.assertEqual(clf.infer_from_codes(self.codes)[0], 1)

    def test_register_disagreement_is_error(self):
        m, clf = self._clf(ARGMAX_REGS, [1, 3, 2, 3], {CLASS_OFF: 3, VLD_OFF: 1})
        with self.assertRaises(RuntimeError):
            clf.infer_from_codes(self.codes)

    def test_missing_valid_bit_is_error(self):
        m, clf = self._clf(ARGMAX_REGS, [0, 0, 9, 0], {CLASS_OFF: 2, VLD_OFF: 0})
        with self.assertRaises(RuntimeError):
            clf.infer_from_codes(self.codes)


if __name__ == "__main__":
    unittest.main()
