"""Repository driver: optional temporal-vote registers (vote-variant IP) -- no board."""
import unittest

import numpy as np

from tests.test_hw_class_register import REPO_ROOT, _fake_pynq, _load

CLASS_OFF, VLD_OFF = 0x10, 0x14
K_OFF, VC_OFF, VC_VLD, VN_OFF, VN_VLD = 0x18, 0x20, 0x24, 0x28, 0x2C
VOTE_REGS = {"pred_class": {"address_offset": CLASS_OFF}, "pred_class_ctrl": {"address_offset": VLD_OFF},
             "vote_k": {"address_offset": K_OFF},
             "vote_class": {"address_offset": VC_OFF}, "vote_class_ctrl": {"address_offset": VC_VLD},
             "vote_count": {"address_offset": VN_OFF}, "vote_count_ctrl": {"address_offset": VN_VLD}}


class HwVoteRegisterTests(unittest.TestCase):
    def setUp(self):
        import sys
        self._saved = sys.modules.get("pynq")
        self.bit = REPO_ROOT / "tmp" / "test_hwvote" / "x.bit"
        self.bit.parent.mkdir(parents=True, exist_ok=True)
        self.bit.write_bytes(b"0")
        self.bit.with_suffix(".hwh").write_bytes(b"0")

    def tearDown(self):
        import sys
        if self._saved is None:
            sys.modules.pop("pynq", None)
        else:
            sys.modules["pynq"] = self._saved

    def _backend(self, regs, values):
        m = _load(_fake_pynq(regs, [0, 9, 0, 0], values))
        return m.HardwareBackend(str(self.bit))

    def test_vote_registers_read(self):
        be = self._backend(VOTE_REGS, {CLASS_OFF: 1, VLD_OFF: 1, VC_OFF: 2, VC_VLD: 1, VN_OFF: 4, VN_VLD: 1})
        self.assertTrue(be.has_vote)
        self.assertEqual(be.read_vote(), (2, 4))

    def test_missing_vote_valid_bit_is_error(self):
        be = self._backend(VOTE_REGS, {VC_OFF: 2, VC_VLD: 0, VN_OFF: 4, VN_VLD: 1})
        with self.assertRaises(RuntimeError):
            be.read_vote()

    def test_overlay_without_vote(self):
        be = self._backend({"pred_class": {"address_offset": CLASS_OFF}}, {})
        self.assertFalse(be.has_vote)
        with self.assertRaises(RuntimeError):
            be.set_vote_k(5)


if __name__ == "__main__":
    unittest.main()
