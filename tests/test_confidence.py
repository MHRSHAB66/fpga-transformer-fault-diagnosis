"""host.confidence + live parse_line logits handling (no board)."""
import math
import unittest

from host.confidence import confidence_from_logits, logits_scale
from host.frame_protocol import parse_frame
from host.live_fpga import parse_line


class ConfidenceTests(unittest.TestCase):
    def test_scale_is_deployed_logits_scale(self):
        self.assertAlmostEqual(logits_scale(), 0.10779884107882078, places=15)

    def test_scale_matches_deployed_bundles_and_export(self):
        import json, os
        from host import confidence
        repo = os.path.dirname(os.path.dirname(os.path.abspath(confidence.__file__)))
        paths = [confidence.SCALES_PATH,
                 os.path.join(repo, "artifacts", "deploy", "seq256_z7020_pynq_dma16_py36_20261008",
                              "config", "integer_config_scales.json"),
                 os.path.join(repo, "artifacts", "quantized", "f_seq256_int10", "integer_config.json")]
        vals = [json.load(open(p, encoding="utf-8"))["scales"]["logits"] for p in paths]
        self.assertEqual(len(set(vals)), 1, vals)
        self.assertIn("seq256_argmax_z7020_20261022", confidence.SCALES_PATH)

    def test_softmax_value(self):
        s = 0.1
        lg = [10, -5, 0, 3]
        ex = [math.exp((v - 10) * s) for v in lg]
        self.assertAlmostEqual(confidence_from_logits(lg, 0, scale=s), ex[0] / sum(ex), places=12)

    def test_tie_gives_at_most_half_and_lowest_index(self):
        c = confidence_from_logits([7, 7, -100, -100], 0, scale=0.1)
        self.assertLessEqual(c, 0.5)
        with self.assertRaises(ValueError):          # class 1 is not the tie winner
            confidence_from_logits([7, 7, -100, -100], 1, scale=0.1)

    def test_mismatched_class_rejected(self):
        with self.assertRaises(ValueError):
            confidence_from_logits([1, 2, 3, 4], 0)


class ParseLineLogitsTests(unittest.TestCase):
    def test_legacy_line_keeps_na(self):
        ev = parse_line("R idx=5 cls=3 lat_ms=16.1 temp_c=49.0")
        self.assertIsNone(ev["conf"])
        self.assertTrue(ev["frame"].endswith("CONF=NA"))

    def test_line_with_logits_has_conf(self):
        ev = parse_line("R idx=0 cls=3 lat_ms=16.197 temp_c=47.986 logits=-116,-129,-36,112")
        self.assertEqual(ev["logits"], [-116, -129, -36, 112])
        self.assertGreater(ev["conf"], 0.99)
        fr = parse_frame(ev["frame"])
        self.assertEqual(fr.class_id, 3)
        self.assertAlmostEqual(fr.confidence, round(ev["conf"], 3), places=6)

    def test_inconsistent_logits_raise(self):
        with self.assertRaises(ValueError):
            parse_line("R idx=0 cls=0 lat_ms=1 temp_c=NA logits=-116,-129,-36,112")


if __name__ == "__main__":
    unittest.main()


class CalibratedConfidenceTests(unittest.TestCase):
    def setUp(self):
        import host.confidence as hc
        self.hc = hc
        hc._CAL.clear()

    def test_frozen_artifact_is_used_and_consistent(self):
        import json
        cal = self.hc.load_calibration()
        self.assertIsNotNone(cal)
        d = json.load(open(self.hc.CALIBRATION_PATH, encoding="utf-8"))
        self.assertFalse(d["test_used_for_fitting"])
        self.assertEqual(d["s_logits"], self.hc.logits_scale())
        lg = [-116, -129, -36, 112]
        v, kind = self.hc.display_confidence(lg, 3)
        self.assertTrue(kind.startswith("calibrated_T="))
        self.assertAlmostEqual(v, self.hc.confidence_from_logits(lg, 3, temperature=cal["T"]), places=12)

    def test_temperature_formula_and_argmax_invariance(self):
        s = 0.1
        for T in (0.5, 1.0, 1.4248, 3.0):
            lg = [10, -5, 0, 3]
            ex = [math.exp((v - 10) * s / T) for v in lg]
            self.assertAlmostEqual(self.hc.confidence_from_logits(lg, 0, scale=s, temperature=T),
                                   ex[0] / sum(ex), places=12)
        with self.assertRaises(ValueError):          # class must still be the logits' argmax
            self.hc.confidence_from_logits([1, 9, 2, 3], 0, temperature=1.4248)

    def test_missing_or_mismatched_artifact_falls_back_uncalibrated(self):
        import json, os, tempfile
        v, kind = self.hc.display_confidence([0, 0, 9, 0], 2, calibration_path="does/not/exist.json")
        self.assertEqual(kind, "uncalibrated")
        self.assertAlmostEqual(v, self.hc.confidence_from_logits([0, 0, 9, 0], 2), places=12)
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"T": 1.4, "s_logits": 0.5}, f)    # wrong scale -> not this model
        try:
            self.assertEqual(self.hc.display_confidence([0, 0, 9, 0], 2, calibration_path=path)[1],
                             "uncalibrated")
        finally:
            os.remove(path)

    def test_parse_line_reports_kind_and_never_conf_without_logits(self):
        ev = parse_line("R idx=0 cls=3 lat_ms=16.0 temp_c=47.0 logits=-116,-129,-36,112")
        self.assertTrue(ev["conf_kind"].startswith("calibrated_T="))
        self.assertGreater(ev["conf_uncalibrated"], ev["conf"])      # T>1 lowers an extreme score
        self.assertIn("CONF=", ev["frame"])
        ev2 = parse_line("R idx=5 cls=3 lat_ms=16.1 temp_c=49.0")
        self.assertIsNone(ev2["conf"])
        self.assertIsNone(ev2["conf_kind"])
        self.assertTrue(ev2["frame"].endswith("CONF=NA"))
