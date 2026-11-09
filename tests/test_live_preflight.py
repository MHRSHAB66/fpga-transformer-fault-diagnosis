"""Tests for host.live_preflight.evaluate -- the PURE validation logic.

Uses mock facts dicts (no SSH, no board). Confirms each check passes on a
healthy probe and fails on the relevant broken fact, including the DMA contract
(65536 B) and input-frame (32768 B) numbers and the no-prior-runner guard.
"""
import unittest

from host.live_preflight import evaluate, all_ok, _probe_script


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
    "IN_BYTES": "1128562912",
    "BIT_SHA": "b" * 64,
    "HWH_SHA": "h" * 64,
    "TEMP_C": "48.109",
    "OTHER": "",
}
EXP = {"bit": "b" * 64, "hwh": "h" * 64}


class PreflightEvaluateTests(unittest.TestCase):
    def test_all_pass_on_healthy_probe(self):
        checks = evaluate(GOOD, n=150, expected_hashes=EXP)
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
        checks = evaluate(facts, n=override.pop("_n", 150), expected_hashes=EXP)
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
        checks = evaluate(GOOD, n=0, expected_hashes=EXP)
        self.assertFalse({c["check"]: c["ok"] for c in checks}["requested_n_valid"])
        checks = evaluate(GOOD, n=95405, expected_hashes=EXP)
        self.assertFalse({c["check"]: c["ok"] for c in checks}["requested_n_valid"])


    def test_wrong_bitstream_identity_fails(self):
        self.assertFalse(self._fail(BIT_SHA="0" * 64)["bit_identity"])
        self.assertFalse(self._fail(HWH_SHA="")["hwh_identity"])

    def test_missing_expected_hash_fails(self):
        checks = evaluate(GOOD, n=150, expected_hashes={"bit": None, "hwh": None})
        res = {c["check"]: c["ok"] for c in checks}
        self.assertFalse(res["bit_identity"])
        self.assertFalse(res["hwh_identity"])

    def test_truncated_input_fails(self):
        self.assertFalse(self._fail(IN_BYTES="1000")["input_complete"])

    def test_hot_or_invalid_temperature_blocks_start(self):
        self.assertFalse(self._fail(TEMP_C="75.000")["temperature_below_stop"])
        self.assertFalse(self._fail(TEMP_C="")["temperature_below_stop"])
        self.assertTrue(self._fail(TEMP_C="74.999")["temperature_below_stop"])

    def test_leftover_guard_or_bench_blocks_start(self):
        self.assertFalse(self._fail(OTHER="1234,")["no_other_workload_or_guard"])

    def test_bundle_manifest_provides_hashes(self):
        from host.live_preflight import expected_overlay_hashes
        exp = expected_overlay_hashes()
        self.assertEqual(exp["bit"][:12], "ce3c254b8679")
        self.assertEqual(exp["hwh"][:12], "eabc0163be6a")


class ProbeScriptTests(unittest.TestCase):
    def test_workload_pattern_cannot_match_own_probe(self):
        import re
        script = _probe_script("/dep", "/dep/overlay/x.bit",
                               "/home/xilinx/fpga_fulltest_20261008/fpga_testset_in.i32.gz")
        pat = re.search(r"grep -E '([^']+)'", script).group(1)
        self.assertIsNone(re.search(pat, "bash -c " + script))
        self.assertIsNotNone(re.search(pat, "python3.6 -u /x/fpga_live_runner.py a b"))
        self.assertIsNotNone(re.search(pat, "bash /home/xilinx/live_runs/x/temp_guard.sh 12 log 75 1"))
        self.assertIsNotNone(re.search(pat, "python3.6 -u /x/fpga_ram_rt_demo.py a b"))
        self.assertIsNotNone(re.search(pat, "python3.6 -u /x/fpga_rt_acceptance.py a b"))

    def test_runner_pgrep_uses_bracket_trick(self):
        # The no_prior_runner pgrep must NOT match its own shell command line
        # (which literally contains the pattern) -> it must use the [f]pga... form.
        script = _probe_script("/dep", "/dep/overlay/x.bit", "/in.gz")
        self.assertIn('pgrep -f "[f]pga_live_runner.py"', script)
        self.assertNotIn("pgrep -f fpga_live_runner.py", script)


if __name__ == "__main__":
    unittest.main()
