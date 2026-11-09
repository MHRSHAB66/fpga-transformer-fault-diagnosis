"""Software tests for the ram_tcp_e2e_demo mode and the e2e clock mapping (no board)."""
import re
import struct
import unittest

from host import e2e_clock as ec
from host import ram_e2e_demo as ed


def _frame(seq, **kw):
    f = {"seq": seq, "idx": 1000 + seq, "cls": seq % 4, "logits": [seq, -seq, 3, -4],
         "t_rel": 10.0 + seq * 0.02048, "t_ready": 10.016 + seq * 0.02048,
         "board_overflow_cum": 0, "t_pubrx": 10.0161 + seq * 0.02048, "t_pub": 10.0162 + seq * 0.02048}
    f.update(kw)
    return f


class WireFormatTests(unittest.TestCase):
    def test_record_size_matches_board(self):
        self.assertEqual(ed.REC_SIZE, 40)
        self.assertEqual(ed.FRAME_SIZE, 57)

    def test_roundtrip_and_partial_chunks(self):
        frames = [_frame(i) for i in range(5)]
        blob = b"".join(ed.encode_frame(f) for f in frames)
        blob += b"Y" + struct.pack("<Qd", 7, 123.5) + b"E" + struct.pack("<III", 5, 0, 0)
        p, got = ed.Parser(), []
        for i in range(0, len(blob), 13):            # arbitrary TCP segmentation
            got += p.feed(blob[i:i + 13])
        self.assertEqual([k for k, _ in got], ["R"] * 5 + ["Y", "E"])
        for (k, v), f in zip(got[:5], frames):
            self.assertEqual(v["seq"], f["seq"])
            self.assertEqual(v["logits"], f["logits"])
            self.assertAlmostEqual(v["t_rel"], f["t_rel"])
            self.assertAlmostEqual(v["t_pub"], f["t_pub"])
        self.assertEqual(got[5][1], (7, 123.5))
        self.assertEqual(got[6][1], (5, 0, 0))

    def test_bad_type_raises(self):
        with self.assertRaises(ValueError):
            ed.Parser().feed(b"Z" + b"\x00" * 60)


class SeqTrackerTests(unittest.TestCase):
    def test_in_order(self):
        t = ed.SeqTracker()
        for i in range(10):
            t.add(i)
        self.assertEqual((t.missing, t.duplicate, t.out_of_order), ([], [], []))

    def test_gap_duplicate_out_of_order(self):
        t = ed.SeqTracker()
        for i in (0, 1, 4, 4, 2, 5):
            t.add(i)
        self.assertEqual(t.missing, [3])          # 2 arrived late (out of order), 3 never
        self.assertEqual(t.duplicate, [4])
        self.assertEqual(t.out_of_order, [2])


class SinkTests(unittest.TestCase):
    def test_overflow_is_recorded_by_seq(self):
        s = ed.Sink(maxlen=3)
        ok = [s.put(_frame(i)) for i in range(5)]
        self.assertEqual(ok, [True, True, True, False, False])
        self.assertEqual(s.overflow, [3, 4])
        self.assertEqual([f["seq"] for f in s.pop_all()], [0, 1, 2])
        self.assertTrue(s.put(_frame(5)))


class ClockMapTests(unittest.TestCase):
    @staticmethod
    def rounds(a, b, pcs, rtt=0.001):
        # board = a + b * pc ; reply exactly at the midpoint
        return [{"phase": "x", "i": i, "ts": pc - rtt / 2, "tr": pc + rtt / 2, "tb": a + b * pc}
                for i, pc in enumerate(pcs)]

    def test_affine_recovered_with_bound(self):
        a, b = -1000.0, 1.00003
        cm = ec.ClockMap(self.rounds(a, b, [i * 0.5 for i in range(1, 41)]))
        for pc in (3.1, 7.77, 15.0):
            got, err = cm.board_to_pc(a + b * pc)
            self.assertAlmostEqual(got, pc, places=7)
            self.assertLessEqual(err, 0.0005 + 1e-9 + 30e-6 * 2)
        s = cm.summary()
        self.assertAlmostEqual(s["drift_ppm_max"], (1 / b - 1) * 1e6, places=2)

    def test_min_rtt_per_bucket_and_asymmetry_inside_bound(self):
        # reply at an arbitrary point inside [ts, tr]: true pc must lie within the bound
        rr = []
        for i in range(20):
            ts = 1.0 + i * 0.5
            rtt = 0.004 if i % 3 else 0.0008
            true_pc = ts + rtt * 0.9                     # asymmetric
            rr.append({"phase": "x", "i": i, "ts": ts, "tr": ts + rtt, "tb": 50.0 + true_pc})
        cm = ec.ClockMap(rr)
        self.assertTrue(all(an[2] <= 0.004 for an in cm.anc))
        for r in rr:
            pc, err = cm.board_to_pc(r["tb"])
            self.assertLessEqual(abs(pc - (r["tb"] - 50.0)), err + 1e-9)

    def test_no_rounds_rejected(self):
        with self.assertRaises(ValueError):
            ec.ClockMap([])

    def test_non_monotonic_anchor_dropped(self):
        rr = self.rounds(0.0, 1.0, [1.0, 3.0, 5.0])
        rr.append({"phase": "x", "i": 9, "ts": 6.9995, "tr": 7.0005, "tb": 2.0})   # bogus, goes backwards
        cm = ec.ClockMap(rr)
        self.assertEqual(len(cm.anc), 3)

    def test_max_gap(self):
        self.assertAlmostEqual(ec.max_gap_s(self.rounds(0, 1, [1.0, 1.5, 3.5])), 2.0, places=6)


class ClientContractTests(unittest.TestCase):
    def setUp(self):
        self.c = ed.E2EClient(n_total=30200,
                              runner_remote="/home/xilinx/live_runs/e2e_x/fpga_ram_e2e_demo.py",
                              monitor_remote="/home/xilinx/live_runs/e2e_x/temp_guard.sh")

    def test_remote_command_policy(self):
        cmd = self.c._remote_cmd()
        self.assertIn(" 30200 200 1 50 ", cmd)                     # n, warm-up, cpu, prio
        self.assertIn(" %d %d %d" % (ed.PORT, ed.PUB_CPU, ed.PUB_PRIO), cmd)
        self.assertIn("75 1", cmd)
        self.assertIn("GUARD_RC", cmd)

    def test_kill_pattern_only_this_run(self):
        pat = self.c._kill_pattern()
        self.assertIsNotNone(re.search(pat, "python3.6 -u /home/xilinx/live_runs/e2e_x/fpga_ram_e2e_demo.py a"))
        self.assertIsNone(re.search(pat, "python3.6 -u /home/xilinx/live_runs/e2e_y/fpga_ram_e2e_demo.py a"))

    def test_stop_before_launch(self):
        self.c.stop()
        evs = list(self.c.stream())
        self.assertIn("nothing started", evs[0]["msg"])
        self.assertIsNone(self.c.proc)

    def test_preflight_port_checks(self):
        ok = {c["check"]: c["ok"] for c in ed.evaluate_e2e({"port_listeners": "0", "running": ""})}
        self.assertTrue(all(ok.values()))
        bad = {c["check"]: c["ok"] for c in ed.evaluate_e2e({"port_listeners": "1", "running": "12"})}
        self.assertFalse(bad["e2e_port_free"])
        self.assertFalse(bad["no_e2e_runner"])


class ViewerModeTests(unittest.TestCase):
    def test_modes(self):
        from host import fpga_result_viewer as v
        self.assertEqual(v.LIVE_MODE_E2E, ed.MODE)
        self.assertIn(v.LIVE_MODE_E2E, v.LIVE_MODE_BANNERS)
        self.assertIn("HARD REAL-TIME NOT VERIFIED", v.LIVE_MODE_BANNERS[v.LIVE_MODE_LEGACY][0])


if __name__ == "__main__":
    unittest.main()
