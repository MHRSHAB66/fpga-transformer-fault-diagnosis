"""Tests for host.replay_data: int-file parsing, record pairing, and the metrics
accumulator -- including an exact check against the real board-produced
fpga_fulltest_metrics.json so the viewer can never silently diverge from it.

Pure standard library (unittest); no torch / no board / no Tk.
"""
import json
import os
import tempfile
import unittest

from host.replay_data import (
    load_int_file, ReplayDataset, MetricsAccumulator, DataError,
    compare_to_reference,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FULLTEST_DIR = os.path.join(REPO, "artifacts", "results", "fpga_fulltest_20261008_141335")


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


class LoadIntFileTests(unittest.TestCase):
    def test_reads_ints_ignoring_blanks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "a.txt", "0\n1\n\n2\n3\n")
            self.assertEqual(load_int_file(p), [0, 1, 2, 3])

    def test_non_integer_raises_with_lineno(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "a.txt", "0\nx\n")
            with self.assertRaises(DataError) as cm:
                load_int_file(p)
            self.assertIn(":2", str(cm.exception))

    def test_out_of_range_class_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "a.txt", "0\n5\n")
            with self.assertRaises(DataError):
                load_int_file(p)

    def test_missing_file_raises(self):
        with self.assertRaises(DataError):
            load_int_file(os.path.join(tempfile.gettempdir(), "nope_missing_123.txt"))


class ReplayDatasetTests(unittest.TestCase):
    def test_count_mismatch_raises(self):
        with self.assertRaises(DataError):
            ReplayDataset([0, 1], [0])

    def test_empty_raises(self):
        with self.assertRaises(DataError):
            ReplayDataset([], [])

    def test_record_fields(self):
        ds = ReplayDataset([3, 0], [3, 1])
        r0 = ds.record(0)
        self.assertEqual((r0["frame"], r0["class_id"], r0["label"],
                          r0["true_class"], r0["true_label"], r0["correct"], r0["conf"]),
                         (0, 3, "bearing_fault", 3, "bearing_fault", True, None))
        self.assertFalse(ds.record(1)["correct"])

    def test_record_index_bounds(self):
        ds = ReplayDataset([0], [0])
        with self.assertRaises(IndexError):
            ds.record(1)


class MetricsAccumulatorTests(unittest.TestCase):
    def test_small_hand_computed(self):
        # true, pred pairs: one perfect class0, one class1 misread as 0
        acc = MetricsAccumulator()
        acc.add_all([0, 0, 1, 1], [0, 0, 0, 1])
        res = acc.finalize()
        self.assertEqual(res["confusion_matrix"], [[2, 0, 0, 0],
                                                   [1, 1, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 0]])
        self.assertEqual(res["support"], [2, 2, 0, 0])
        self.assertAlmostEqual(res["accuracy"], 3 / 4)
        # class0 precision = 2/3, recall = 2/2
        self.assertAlmostEqual(res["per_class_precision"][0], 2 / 3)
        self.assertAlmostEqual(res["per_class_recall"][0], 1.0)

    def test_running_accuracy_prefix(self):
        acc = MetricsAccumulator()
        acc.add(0, 0)
        self.assertAlmostEqual(acc.running_accuracy, 1.0)
        acc.add(1, 0)
        self.assertAlmostEqual(acc.running_accuracy, 0.5)
        self.assertEqual(acc.incorrect, 1)
        self.assertEqual(acc.predicted_counts()[0], 2)


@unittest.skipUnless(os.path.isdir(FULLTEST_DIR),
                     "fpga_fulltest result dir not present")
class RealFulltestMetricsTests(unittest.TestCase):
    """The deterministic acceptance test: recomputed metrics must equal the
    board's fpga_fulltest_metrics.json to 1e-6, with an identical confusion
    matrix and 95404 windows."""

    @classmethod
    def setUpClass(cls):
        cls.ds = ReplayDataset.from_dir(FULLTEST_DIR)
        with open(os.path.join(FULLTEST_DIR, "fpga_fulltest_metrics.json"),
                  encoding="utf-8") as f:
            cls.ref = json.load(f)

    def test_window_count(self):
        self.assertEqual(self.ds.n, 95404)
        self.assertEqual(self.ds.n, self.ref["test_windows"])

    def test_metrics_match_reference(self):
        acc = MetricsAccumulator()
        acc.add_all(self.ds.labels, self.ds.preds)
        res = acc.finalize()
        diffs = compare_to_reference(res, self.ref, tol=1e-6)
        self.assertEqual(diffs, [], "unexpected metric discrepancies: {}".format(diffs))

    def test_confusion_matrix_exact(self):
        acc = MetricsAccumulator()
        acc.add_all(self.ds.labels, self.ds.preds)
        self.assertEqual(acc.finalize()["confusion_matrix"], self.ref["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
