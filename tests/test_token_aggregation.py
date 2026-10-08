"""Focused tests for the isolated SEQ=256 token aggregation (Phase 1)."""
import unittest

import numpy as np

from src.data.token_aggregation import (
    aggregate_mean_float,
    aggregate_round_div,
    aggregated_seq_len,
)


class TokenAggregationTests(unittest.TestCase):
    def test_output_shape_1024x8_to_256x8(self):
        x = np.zeros((1024, 8), dtype=np.float64)
        self.assertEqual(aggregate_mean_float(x, 4).shape, (256, 8))
        self.assertEqual(aggregate_round_div(x.astype(np.int64), 4).shape, (256, 8))
        self.assertEqual(aggregated_seq_len(1024, 4), 256)

    def test_exact_float_mean_known_values(self):
        # one group of 4 per channel, two channels
        x = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0], [6.0, 40.0]])
        out = aggregate_mean_float(x, 4)
        np.testing.assert_array_equal(out, np.array([[3.0, 25.0]]))

    def test_int_round_half_away_from_zero(self):
        # one 4-sample group per case; check ties and both signs.
        def q(vals):
            g = np.array(vals, dtype=np.int64).reshape(4, 1)
            return int(aggregate_round_div(g, 4)[0, 0])

        self.assertEqual(q([1, 1, 0, 0]), 1)     # sum=2 -> 0.5 -> +1 (away)
        self.assertEqual(q([-1, -1, 0, 0]), -1)  # sum=-2 -> -0.5 -> -1 (away)
        self.assertEqual(q([3, 3, 0, 0]), 2)     # sum=6 -> 1.5 -> 2
        self.assertEqual(q([-3, -3, 0, 0]), -2)  # sum=-6 -> -1.5 -> -2
        self.assertEqual(q([1, 1, 1, 1]), 1)     # sum=4 -> 1 exact

    def test_all_eight_channels_preserved(self):
        x = np.arange(1024 * 8, dtype=np.float64).reshape(1024, 8)
        out = aggregate_mean_float(x, 4)
        self.assertEqual(out.shape[1], 8)
        # channel c of first token = mean of rows 0..3 of channel c
        for c in range(8):
            self.assertAlmostEqual(out[0, c], x[0:4, c].mean())

    def test_deterministic(self):
        rng = np.random.default_rng(0)
        x = rng.integers(-511, 512, size=(1024, 8)).astype(np.int64)
        a = aggregate_round_div(x, 4)
        b = aggregate_round_div(x, 4)
        np.testing.assert_array_equal(a, b)

    def test_rejects_incompatible_length(self):
        with self.assertRaises(ValueError):
            aggregate_mean_float(np.zeros((1023, 8)), 4)  # not divisible
        with self.assertRaises(ValueError):
            aggregate_round_div(np.zeros((1024, 8), dtype=np.int64), 3)  # not power of two

    def test_factor_one_is_noop_seq1024_path_unchanged(self):
        # values AND dtype preserved for both float64 and float32 inputs
        x = np.arange(1024 * 8, dtype=np.float64).reshape(1024, 8)
        out = aggregate_mean_float(x, 1)
        np.testing.assert_array_equal(out, x)
        self.assertEqual(out.dtype, np.float64)
        x32 = x.astype(np.float32)
        out32 = aggregate_mean_float(x32, 1)
        np.testing.assert_array_equal(out32, x32)
        self.assertEqual(out32.dtype, np.float32)
        xi = x.astype(np.int64)
        np.testing.assert_array_equal(aggregate_round_div(xi, 1), xi)

    def test_float_output_is_float32_by_default(self):
        x = np.arange(1024 * 8, dtype=np.float32).reshape(1024, 8)
        out = aggregate_mean_float(x, 4)  # training default
        self.assertEqual(out.shape, (256, 8))
        self.assertEqual(out.dtype, np.float32)

    def test_float32_values_equal_float64_reference_after_cast(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1024, 8)).astype(np.float32)
        ref64 = aggregate_mean_float(x, 4, output_dtype=np.float64)
        out32 = aggregate_mean_float(x, 4)  # default float32
        np.testing.assert_array_equal(out32, ref64.astype(np.float32))

    def test_model_forward_accepts_aggregated_batch(self):
        import torch
        from src.models.transformer_baseline import TransformerBaseline

        rng = np.random.default_rng(2)
        window = rng.standard_normal((2, 1024, 8)).astype(np.float32)  # [B,1024,8]
        agg = aggregate_mean_float(window, 4)  # [B,256,8] float32
        self.assertEqual(agg.shape, (2, 256, 8))
        self.assertEqual(agg.dtype, np.float32)
        model = TransformerBaseline(
            input_dim=8, seq_len=256, num_classes=4, d_model=32, nhead=4,
            dim_feedforward=64, num_layers=1,
        ).eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(agg))  # no dtype mismatch
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(logits.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
