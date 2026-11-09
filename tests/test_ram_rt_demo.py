"""Software tests for the ram_realtime_demo mode (no board)."""
import re
import unittest

from host import ram_rt_demo as rd


class ClockMapTests(unittest.TestCase):
    def _sync(self, phase, pc_mid, board, rtts):
        out = []
        for i, r in enumerate(rtts):
            out.append({"phase": phase, "i": i, "ts": pc_mid - r / 2, "tr": pc_mid + r / 2, "tb": board})
        return out

    def test_two_anchors_offset_and_drift(self):
        # board = 1000 + 1.00005 * pc ; best rounds have rtt 2 ms
        a, b = 1000.0, 1.00005
        s = self._sync("start", 10.0, a + b * 10.0, [0.010, 0.002, 0.005])
        s += self._sync("end", 610.0, a + b * 610.0, [0.004, 0.002])
        m = rd.clock_map(s)
        self.assertEqual(m["anchors"], 2)
        self.assertAlmostEqual(m["b"], b, places=9)
        self.assertAlmostEqual(m["a"], a, places=5)
        self.assertAlmostEqual(m["err_bound_ms"], 1.0, places=6)
        self.assertAlmostEqual(m["drift_ppm"], 50.0, places=3)

    def test_start_only(self):
        m = rd.clock_map(self._sync("start", 5.0, 105.0, [0.003]))
        self.assertEqual(m["anchors"], 1)
        self.assertEqual(m["b"], 1.0)
        self.assertAlmostEqual(m["a"], 100.0)

    def test_no_sync(self):
        self.assertIsNone(rd.clock_map([]))

    def test_e2e_latency(self):
        m = {"a": 100.0, "b": 1.0}
        res = [{"rel_b": 105.0, "rx_pc": 5.030, "disp_pc": 5.080}]
        self.assertAlmostEqual(rd.e2e_latencies(res, m, "rx_pc")[0], 30.0, places=6)
        self.assertAlmostEqual(rd.e2e_latencies(res, m, "disp_pc")[0], 80.0, places=6)

    def test_latency_stats_counts_over_period(self):
        st = rd.latency_stats([10.0, 20.0, 25.0, 30.0])
        self.assertEqual(st["over_20.48ms"], 2)
        self.assertEqual(st["max"], 30.0)


class PreflightSetTests(unittest.TestCase):
    PROV = {"set_sha256": "a" * 64}

    def _ok(self, **facts):
        base = {"set.i32": "a" * 64, "indices.txt": "b" * 64}
        base.update(facts)
        return {c["check"]: c["ok"] for c in rd.evaluate_set(base, self.PROV, "b" * 64, 1200)}

    def test_all_ok(self):
        self.assertTrue(all(self._ok().values()))

    def test_wrong_set_hash(self):
        self.assertFalse(self._ok(**{"set.i32": "c" * 64})["ram_set_identity"])

    def test_missing_indices(self):
        self.assertFalse(self._ok(**{"indices.txt": ""})["ram_set_indices_identity"])

    def test_running_demo_refused(self):
        self.assertFalse(self._ok(running="1234")["no_ram_demo_running"])

    def test_n_must_exceed_warmup(self):
        checks = {c["check"]: c["ok"] for c in rd.evaluate_set(
            {"set.i32": "a" * 64, "indices.txt": "b" * 64}, self.PROV, "b" * 64, 200)}
        self.assertFalse(checks["ram_n_total_valid"])


class RemoteCommandTests(unittest.TestCase):
    def setUp(self):
        self.c = rd.RamRealtimeDemoClient(n_total=30200,
                                          runner_remote="/home/xilinx/live_runs/ram_x/fpga_ram_rt_demo.py",
                                          monitor_remote="/home/xilinx/live_runs/ram_x/temp_guard.sh")

    def test_policy_and_guard(self):
        cmd = self.c._remote_cmd()
        self.assertIn(" 30200 200 1 50 ", cmd)                 # n_total warmup cpu prio
        self.assertIn("temp_guard.sh $P", cmd.replace("'", ""))
        self.assertIn("75 1", cmd)
        self.assertIn("<&3 &", cmd)                            # stdin reaches the runner (clock sync)
        self.assertIn("GUARD_RC", cmd)

    def test_kill_pattern_only_this_run(self):
        pat = self.c._kill_pattern()
        self.assertIsNotNone(re.search(pat, "python3.6 -u /home/xilinx/live_runs/ram_x/fpga_ram_rt_demo.py a"))
        self.assertIsNone(re.search(pat, "python3.6 -u /home/xilinx/live_runs/ram_y/fpga_ram_rt_demo.py a"))
        self.assertIsNone(re.search(pat, "bash -c cd x; python3.6 -u /home/xilinx/live_runs/ram_x/fpga_ram_rt_demo.py"))

    def test_stop_before_launch_starts_nothing(self):
        self.c.stop()
        evs = list(self.c.stream())
        self.assertEqual(evs[0]["type"], "status")
        self.assertIn("nothing started", evs[0]["msg"])
        self.assertIsNone(self.c.proc)


class PreflightOnlyClientTests(unittest.TestCase):
    def test_client_without_runner_can_be_built(self):
        # GUI Preflight builds the client before anything is pushed
        c = rd.RamRealtimeDemoClient(n_total=1200)
        self.assertEqual(c.n_total, 1200)


class ViewerModeTests(unittest.TestCase):
    def test_legacy_is_default_and_banners_scoped(self):
        from host import fpga_result_viewer as v
        self.assertIn("AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED",
                      v.LIVE_MODE_BANNERS[v.LIVE_MODE_LEGACY][0])
        self.assertIn("NOT hard real-time", v.LIVE_MODE_BANNERS[v.LIVE_MODE_RAM][0])
        self.assertEqual(v.LIVE_MODE_RAM, rd.MODE)


if __name__ == "__main__":
    unittest.main()
