"""Tests for host.live_sampling -- deterministic Live-demo window selection.

Pure stdlib. No board, no inference. Verifies back-compat (sequential),
spread (strided), multiclass coverage (class_balanced), explicit validation,
and that every plan yields in-range real dataset indices.
"""
import unittest

from host.live_sampling import (
    build_plan, strided_plan, class_balanced_plan, explicit_plan, SamplingError,
)


class SamplingTests(unittest.TestCase):
    def test_sequential_is_backward_compatible(self):
        self.assertEqual(build_plan("sequential", total=1000, n=5), [0, 1, 2, 3, 4])

    def test_strided_spreads_and_is_sorted_unique(self):
        plan = strided_plan(total=1000, n=10)
        self.assertEqual(len(plan), 10)
        self.assertEqual(plan, sorted(plan))
        self.assertEqual(len(set(plan)), 10)
        self.assertTrue(all(0 <= i < 1000 for i in plan))
        self.assertGreater(plan[-1], 800)  # actually reaches the far end

    def test_strided_full_set(self):
        self.assertEqual(strided_plan(5, 5), [0, 1, 2, 3, 4])

    def test_class_balanced_covers_all_present_classes(self):
        # 400 windows: classes 0..3 in contiguous blocks of 100
        labels = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100
        plan = class_balanced_plan(labels, n=40)
        self.assertEqual(len(plan), 40)
        self.assertTrue(all(0 <= i < 400 for i in plan))
        covered = {labels[i] for i in plan}
        self.assertEqual(covered, {0, 1, 2, 3})
        # round-robin: the first four picks touch four different classes
        first4 = [labels[i] for i in plan[:4]]
        self.assertEqual(len(set(first4)), 4)

    def test_class_balanced_deterministic(self):
        labels = [0, 1, 2, 3] * 50
        self.assertEqual(class_balanced_plan(labels, 12), class_balanced_plan(labels, 12))

    def test_class_balanced_skips_absent_classes(self):
        labels = [3] * 200  # only one class present
        plan = class_balanced_plan(labels, n=10)
        self.assertEqual(len(plan), 10)
        self.assertTrue(all(labels[i] == 3 for i in plan))

    def test_explicit_validates_range(self):
        self.assertEqual(explicit_plan([2, 5, 9], total=10), [2, 5, 9])
        with self.assertRaises(SamplingError):
            explicit_plan([2, 10], total=10)

    def test_requested_more_than_available(self):
        with self.assertRaises(SamplingError):
            build_plan("strided", total=5, n=6)

    def test_bad_mode(self):
        with self.assertRaises(SamplingError):
            build_plan("nonsense", total=10, n=2)

    def test_class_balanced_needs_labels(self):
        with self.assertRaises(SamplingError):
            build_plan("class_balanced", total=10, n=2, labels=None)


if __name__ == "__main__":
    unittest.main()
