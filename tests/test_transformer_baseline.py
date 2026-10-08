import unittest

import torch

from src.models.transformer_baseline import PositionalEncoding, TransformerBaseline


class PositionalEncodingTests(unittest.TestCase):
    def test_supports_odd_model_dimension_and_reuses_one_table_for_batch(self):
        encoding = PositionalEncoding(d_model=5, max_len=7, dropout=0.0)
        output = encoding(torch.zeros(3, 7, 5))

        self.assertEqual(tuple(encoding.pe.shape), (1, 7, 5))
        self.assertEqual(tuple(output.shape), (3, 7, 5))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output[0], output[1])
        torch.testing.assert_close(output[1], output[2])

    def test_rejects_sequence_longer_than_precomputed_table(self):
        encoding = PositionalEncoding(d_model=4, max_len=3, dropout=0.0)

        with self.assertRaisesRegex(ValueError, "exceeds maximum"):
            encoding(torch.zeros(1, 4, 4))


class TransformerBaselineTests(unittest.TestCase):
    def setUp(self):
        self.model = TransformerBaseline(
            input_dim=3,
            seq_len=8,
            num_classes=4,
            d_model=6,
            nhead=2,
            num_layers=1,
            dim_feedforward=12,
            dropout=0.0,
        ).eval()

    def test_forward_returns_finite_logits_with_expected_shape(self):
        with torch.no_grad():
            output = self.model(torch.randn(2, 8, 3))

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_rejects_empty_sequence_before_mean_pooling(self):
        with self.assertRaisesRegex(ValueError, "Sequence length must be positive"):
            self.model(torch.empty(2, 0, 3))

    def test_rejects_wrong_rank_features_and_excessive_length(self):
        invalid_inputs = (
            (torch.zeros(8, 3), "Expected input shape"),
            (torch.zeros(2, 8, 2), "3 input features"),
            (torch.zeros(2, 9, 3), "configured maximum"),
        )

        for tensor, message in invalid_inputs:
            with self.subTest(shape=tuple(tensor.shape)):
                with self.assertRaisesRegex(ValueError, message):
                    self.model(tensor)

    def test_state_dict_round_trip_remains_compatible(self):
        restored = TransformerBaseline(
            input_dim=3,
            seq_len=8,
            num_classes=4,
            d_model=6,
            nhead=2,
            num_layers=1,
            dim_feedforward=12,
            dropout=0.0,
        )

        result = restored.load_state_dict(self.model.state_dict())

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
