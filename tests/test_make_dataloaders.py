import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.make_dataloaders import (
    build_dataloaders,
    build_X_y_from_file_index,
    deterministic_small_split,
    safe_train_test_split,
    split_file_index,
)


class DeterministicSmallSplitTests(unittest.TestCase):
    def test_rejects_invalid_train_size(self):
        index_df = pd.DataFrame({"filepath": ["a.csv", "b.csv"], "label": ["normal", "normal"]})

        invalid_values = (True, None, "0.5", np.nan, np.inf, -0.1, 0.0, 1.0, 1.1)
        for value in invalid_values:
            with self.subTest(train_size=value):
                with self.assertRaisesRegex(ValueError, "train_size"):
                    deterministic_small_split(index_df, value)

    def test_split_is_repeatable_nonempty_and_preserves_rows(self):
        index_df = pd.DataFrame(
            {
                "filepath": [f"{name}.csv" for name in "abcde"],
                "label": ["normal", "imbalance", "normal", "misalignment", "bearing_fault"],
            }
        )

        first_a, second_a = deterministic_small_split(index_df, 0.6)
        first_b, second_b = deterministic_small_split(index_df, 0.6)

        pd.testing.assert_frame_equal(first_a, first_b)
        pd.testing.assert_frame_equal(second_a, second_b)
        self.assertEqual(len(first_a), 3)
        self.assertEqual(len(second_a), 2)
        self.assertEqual(
            sorted(pd.concat([first_a, second_a])["filepath"].tolist()),
            sorted(index_df["filepath"].tolist()),
        )

    def test_empty_and_single_row_inputs_keep_dataframe_structure(self):
        empty_df = pd.DataFrame(columns=["filepath", "label"])
        empty_first, empty_second = deterministic_small_split(empty_df, 0.7)
        self.assertTrue(empty_first.empty)
        self.assertTrue(empty_second.empty)
        self.assertListEqual(list(empty_first.columns), ["filepath", "label"])
        self.assertListEqual(list(empty_second.columns), ["filepath", "label"])

        one_row_df = pd.DataFrame({"filepath": ["a.csv"], "label": ["normal"]})
        first, second = deterministic_small_split(one_row_df, 0.7)
        pd.testing.assert_frame_equal(first, one_row_df)
        self.assertTrue(second.empty)
        self.assertListEqual(list(second.columns), ["filepath", "label"])


class SafeTrainTestSplitTests(unittest.TestCase):
    def setUp(self):
        self.index_df = pd.DataFrame(
            {
                "filepath": ["a.csv", "b.csv", "c.csv", "d.csv"],
                "label": ["normal", "normal", "imbalance", "imbalance"],
            }
        )

    def test_rejects_invalid_train_size_before_splitting(self):
        for value in (True, None, "0.5", np.nan, np.inf, 0.0, 1.0):
            with self.subTest(train_size=value):
                with self.assertRaisesRegex(ValueError, "train_size"):
                    safe_train_test_split(self.index_df, value, self.index_df["label"])

    def test_rejects_mismatched_stratify_labels(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            safe_train_test_split(self.index_df, 0.5, self.index_df["label"].iloc[:2])

    @patch("src.data.make_dataloaders.train_test_split")
    def test_infeasible_stratification_uses_one_unstratified_split(self, split_mock):
        index_df = self.index_df.copy()
        index_df.loc[3, "label"] = "bearing_fault"
        safe_train_test_split(index_df, 0.5, index_df["label"])

        split_mock.assert_called_once()
        self.assertIsNone(split_mock.call_args.kwargs["stratify"])

    @patch("src.data.make_dataloaders.train_test_split")
    def test_unstratified_failure_goes_directly_to_debug_split(self, split_mock):
        first, second = safe_train_test_split(self.index_df.iloc[:1], 0.5, None)

        split_mock.assert_not_called()
        self.assertEqual(len(first), 1)
        self.assertTrue(second.empty)

    @patch("src.data.make_dataloaders.train_test_split")
    def test_unexpected_sklearn_error_is_not_hidden(self, split_mock):
        split_mock.side_effect = ValueError("unexpected sklearn error")
        with self.assertRaisesRegex(ValueError, "unexpected sklearn error"):
            safe_train_test_split(self.index_df, 0.5, self.index_df["label"])


class FileIndexSchemaTests(unittest.TestCase):
    def test_missing_columns_are_reported_for_both_public_functions(self):
        for function in (split_file_index, build_X_y_from_file_index):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "filepath"):
                    function(pd.DataFrame({"label": ["normal"]}))
                with self.assertRaisesRegex(ValueError, "label"):
                    function(pd.DataFrame({"filepath": ["a.csv"]}))

    def test_empty_index_with_expected_columns_is_accepted(self):
        index_df = pd.DataFrame(columns=["filepath", "label"])
        self.assertTrue(all(part.empty for part in split_file_index(index_df)))
        X, y, kept, dropped, _ = build_X_y_from_file_index(index_df)
        self.assertEqual((len(X), len(y), kept, dropped), (0, 0, 0, 0))


class BuildArraysTests(unittest.TestCase):
    def test_invalid_window_parameters_are_rejected_on_empty_index(self):
        index_df = pd.DataFrame(columns=["filepath", "label"])
        for name, value, error_type in (
            ("window_size", True, TypeError),
            ("window_size", 0, ValueError),
            ("stride", "4", TypeError),
            ("stride", 0, ValueError),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(error_type, name):
                    build_X_y_from_file_index(index_df, **{name: value})
    @patch("src.data.make_dataloaders.load_one_csv")
    def test_invalid_label_is_counted_without_loading_file(self, load_mock):
        index_df = pd.DataFrame({"filepath": ["unknown.csv"], "label": ["unknown"]})

        X, y, kept, dropped, counts = build_X_y_from_file_index(index_df)

        load_mock.assert_not_called()
        self.assertEqual((kept, dropped), (0, 1))
        self.assertEqual(X.shape[0], 0)
        self.assertEqual(y.shape[0], 0)
        self.assertEqual(counts, {"normalized": 0})

    @patch("src.data.make_dataloaders.load_one_csv")
    def test_normalized_count_uses_only_clear_key(self, load_mock):
        load_mock.return_value = pd.DataFrame(np.arange(64, dtype=np.float32).reshape(8, 8))
        index_df = pd.DataFrame({"filepath": ["valid.csv"], "label": ["normal"]})

        X, y, kept, dropped, counts = build_X_y_from_file_index(
            index_df,
            window_size=4,
            stride=4,
        )

        self.assertEqual((kept, dropped), (1, 0))
        self.assertEqual((len(X), len(y)), (2, 2))
        self.assertEqual(counts, {"normalized": 2})

    def test_mismatched_window_and_label_counts_raise(self):
        index_df = pd.DataFrame(columns=["filepath", "label"])
        fake_X = np.zeros((1, 4, 8), dtype=np.float32)
        with patch("src.data.make_dataloaders.np.empty", return_value=fake_X):
            with self.assertRaisesRegex(RuntimeError, "Window and label counts"):
                build_X_y_from_file_index(index_df, window_size=4)


class BuildDataLoadersValidationTests(unittest.TestCase):
    @patch("src.data.make_dataloaders.build_file_index")
    def test_rejects_non_integer_parameters_before_scanning_data(self, index_mock):
        invalid_cases = (
            ("max_files", True),
            ("max_files", 2.5),
            ("batch_size", False),
            ("batch_size", 4.5),
            ("window_size", "1024"),
            ("stride", 1.5),
            ("num_workers", True),
            ("num_workers", 2.5),
        )
        for parameter, value in invalid_cases:
            with self.subTest(parameter=parameter, value=value):
                with self.assertRaisesRegex(TypeError, parameter):
                    build_dataloaders(**{parameter: value})
        index_mock.assert_not_called()

    @patch("src.data.make_dataloaders.build_file_index")
    def test_rejects_out_of_range_parameters_before_scanning_data(self, index_mock):
        invalid_cases = (
            ("max_files", 0),
            ("max_files", -1),
            ("batch_size", 0),
            ("window_size", -1),
            ("stride", 0),
            ("num_workers", -1),
        )
        for parameter, value in invalid_cases:
            with self.subTest(parameter=parameter, value=value):
                with self.assertRaisesRegex(ValueError, parameter):
                    build_dataloaders(**{parameter: value})
        index_mock.assert_not_called()

    @patch("src.data.make_dataloaders.build_file_index")
    def test_valid_parameter_types_reach_data_discovery(self, index_mock):
        index_mock.return_value = pd.DataFrame(columns=["filepath", "label"])
        with self.assertRaisesRegex(ValueError, "No CSV files"):
            build_dataloaders(
                max_files=np.int64(2),
                batch_size=np.int64(4),
                window_size=np.int64(8),
                stride=np.int64(8),
                num_workers=np.int64(0),
            )
        index_mock.assert_called_once()

    @patch("src.data.make_dataloaders.split_file_index")
    @patch("src.data.make_dataloaders.build_file_index")
    def test_rejects_fewer_than_four_valid_files_before_splitting(
        self,
        index_mock,
        split_mock,
    ):
        index_mock.return_value = pd.DataFrame(
            {
                "filepath": ["a.csv", "b.csv", "c.csv"],
                "label": ["normal", "imbalance", "misalignment"],
            }
        )

        with self.assertRaisesRegex(ValueError, "At least 4 valid files"):
            build_dataloaders()

        split_mock.assert_not_called()

    @patch("src.data.make_dataloaders.build_X_y_from_file_index")
    @patch("src.data.make_dataloaders.split_file_index")
    @patch("src.data.make_dataloaders.build_file_index")
    def test_file_count_mismatch_is_reported(self, index_mock, split_mock, arrays_mock):
        index_df = pd.DataFrame(
            {
                "filepath": [f"{i}.csv" for i in range(4)],
                "label": ["normal"] * 4,
            }
        )
        index_mock.return_value = index_df
        split_mock.return_value = (
            index_df.iloc[:2],
            index_df.iloc[2:3],
            index_df.iloc[3:4],
        )
        arrays_mock.return_value = (
            np.zeros((1, 4, 8), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            1,
            0,
            {"normalized": 1},
        )

        with self.assertRaisesRegex(RuntimeError, "train file counts"):
            build_dataloaders(window_size=4)


if __name__ == "__main__":
    unittest.main()
