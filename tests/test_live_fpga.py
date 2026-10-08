"""Tests for host.live_fpga line parsing (no board / no SSH required).

Only the pure parser is exercised here; the actual live run is hardware-only and
is validated separately by `python -m host.live_fpga` against the real board.
"""
import unittest

from host.live_fpga import parse_line


class ParseLineTests(unittest.TestCase):
    def test_result_line(self):
        ev = parse_line("R idx=7 cls=3 lat_ms=21.456 temp_c=50.125")
        self.assertEqual(ev["type"], "result")
        self.assertEqual((ev["idx"], ev["cls"], ev["label"]), (7, 3, "bearing_fault"))
        self.assertAlmostEqual(ev["lat_ms"], 21.456)
        self.assertAlmostEqual(ev["temp_c"], 50.125)
        self.assertEqual(ev["frame"], "FRAME=7,CLASS=3,LABEL=bearing_fault,CONF=NA")

    def test_result_line_temp_na(self):
        ev = parse_line("R idx=0 cls=0 lat_ms=19.0 temp_c=NA")
        self.assertIsNone(ev["temp_c"])
        self.assertEqual(ev["label"], "normal")

    def test_temp_start(self):
        self.assertEqual(parse_line("TEMP_START=49.708"),
                         {"type": "temp_start", "temp_c": 49.708})
        self.assertIsNone(parse_line("TEMP_START=None")["temp_c"])

    def test_meta(self):
        self.assertEqual(parse_line("META n=150 seq=1024 ch=8"),
                         {"type": "meta", "n": 150, "seq": 1024, "ch": 8})

    def test_summary(self):
        ev = parse_line("SUMMARY executed=150 aborted=None wall_s=3.2 throughput=46.9 "
                        "temp_start=49.7 temp_max=51.0 temp_end=50.8")
        self.assertEqual(ev["type"], "summary")
        self.assertEqual(ev["executed"], 150)
        self.assertIsNone(ev["aborted"])
        self.assertAlmostEqual(ev["throughput"], 46.9)

    def test_abort_and_blank(self):
        self.assertEqual(parse_line("ABORT temperature 75.1 C")["type"], "abort")
        self.assertIsNone(parse_line("   "))


if __name__ == "__main__":
    unittest.main()
