"""Tests for host.live_preflight.evaluate -- the PURE validation logic.

Uses mock facts dicts (no SSH, no board). Confirms each check passes on a
healthy probe and fails on the relevant broken fact, including the DMA contract
(65536 B) and input-frame (32768 B) numbers and the no-prior-runner guard.
"""
import unittest

from host.live_preflight import evaluate, all_ok


GOOD = {
    "reachable": "1",
    "PY": "Python 3.6.5",
    "DEP_EXISTS": "1",
    "BIT_EXISTS": "1",
    "HWH_EXISTS": "1",
    "IN_EXISTS": "1",
    "DRIVER_IMPORT": "ok",
    "DMA_MAX": "65536",
    "FRAME_BYTES": "32768",
    "XADC_RAW": "2620",
    "RUNNER": "",
}


class PreflightEvaluateTests(unittest.TestCase):
    def test_all_pass_on_healthy_probe(self):
        checks = evaluate(GOOD, n=150)
        self.assertTrue(all_ok(checks), [c for c in checks if not c["ok"]])
        names = {c["check"] for c in checks}
        self.assertIn("dma_contract_65536B", names)
        self.assertIn("input_frame_32768B", names)
        self.assertIn("no_prior_runner", names)

    def test_unreachable_short_circuits(self):
        checks = evaluate({"reachable": "0", "error": "timeout"}, n=150)
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0]["ok"])
        self.assertEqual(checks[0]["check"], "ssh_connectivity")

    def _fail(self, **override):
        facts = dict(GOOD)
        facts.update(override)
        checks = evaluate(facts, n=override.pop("_n", 150))
        return {c["check"]: c["ok"] for c in checks}

    def test_wrong_dma_contract_fails(self):
        self.assertFalse(self._fail(DMA_MAX="16384")["dma_contract_65536B"])

    def test_wrong_frame_bytes_fails(self):
        self.assertFalse(self._fail(FRAME_BYTES="16384")["input_frame_32768B"])

    def test_missing_bitstream_fails(self):
        self.assertFalse(self._fail(BIT_EXISTS="0")["bitstream_present"])

    def test_driver_import_failure(self):
        self.assertFalse(self._fail(DRIVER_IMPORT="fail")["driver_import"])

    def test_prior_runner_blocks(self):
        res = self._fail(RUNNER="3456")
        self.assertFalse(res["no_prior_runner"])

    def test_bad_python_version(self):
        self.assertFalse(self._fail(PY="Python 3.12.2")["python3.6"])

    def test_missing_xadc(self):
        self.assertFalse(self._fail(XADC_RAW="")["xadc_temperature"])

    def test_requested_n_bounds(self):
        checks = evaluate(GOOD, n=0)
        self.assertFalse({c["check"]: c["ok"] for c in checks}["requested_n_valid"])
        checks = evaluate(GOOD, n=95405)
        self.assertFalse({c["check"]: c["ok"] for c in checks}["requested_n_valid"])


if __name__ == "__main__":
    unittest.main()
