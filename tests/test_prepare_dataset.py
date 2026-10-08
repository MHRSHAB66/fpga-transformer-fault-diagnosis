"""Synthetic regression tests; no training data or artifacts are modified."""
import unittest

import numpy as np
import pandas as pd

from src.data.prepare_dataset import (
    create_windows_from_dataframe, normalize_windows_array,
    preprocess_dataframe, should_drop_file,
)


class PreprocessingTests(unittest.TestCase):
    def test_threshold_validation(self):
        for value in (-1, 2, np.nan, np.inf, True, '0.2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                should_drop_file(pd.DataFrame(), value)
        frame = pd.DataFrame({'x': [1, 2, 3, 4, np.nan]})
        self.assertFalse(should_drop_file(frame, 0.2))
        self.assertTrue(should_drop_file(frame, 0.19))

    def test_preprocessing(self):
        for value in (np.inf, -np.inf, 'inf'):
            self.assertIsNone(preprocess_dataframe(pd.DataFrame({'x': [1, value]})))
        self.assertIsNone(preprocess_dataframe(pd.DataFrame()))
        frame = pd.DataFrame(np.ones((10, 8)))
        frame[0] = np.nan
        self.assertIsNone(preprocess_dataframe(frame))
        frame = pd.DataFrame({'x': [1., 2., np.nan, 4., 5.]})
        result = preprocess_dataframe(frame)
        self.assertEqual(result.iloc[2, 0], 3.)
        self.assertTrue(pd.isna(frame.iloc[2, 0]))
        with self.assertRaises(NotImplementedError):
            preprocess_dataframe(pd.DataFrame(), normalize=True)

    def test_normalization_validation(self):
        valid = np.ones((2, 4, 3))
        for value in (0, -1, np.nan, np.inf, True, None, '1e-8'):
            with self.subTest(epsilon=value), self.assertRaises(ValueError):
                normalize_windows_array(valid, epsilon=value)
        for shape in ((), (3,), (2, 3), (2, 0, 3), (2, 3, 0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                normalize_windows_array(np.empty(shape))
        for value in (np.nan, np.inf, -np.inf, 1e300, 3e38):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_windows_array(np.full((1, 4, 1), value))
        with self.assertRaises(ValueError):
            normalize_windows_array(valid, method='minmax')
        self.assertEqual(normalize_windows_array(np.empty((0, 4, 3))).shape, (0, 4, 3))

    def test_normalization_unchanged(self):
        x = np.random.default_rng(42).normal(size=(3, 32, 8)).astype(np.float32)
        mean, std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
        expected = ((x - mean) / np.where(np.abs(std) < 1e-8, 1., std)).astype(np.float32)
        np.testing.assert_array_equal(normalize_windows_array(x), expected)
        np.testing.assert_array_equal(normalize_windows_array(np.ones((2, 4, 3))), 0.)

    def test_windows(self):
        x = np.arange(80).reshape(10, 8)
        for data in (x, pd.DataFrame(x)):
            result = create_windows_from_dataframe(data, 4, 3)
            self.assertEqual(len(result), 3)
            np.testing.assert_array_equal(result[1], x[3:7])
            self.assertEqual(create_windows_from_dataframe(data[:2], 4, 3), [])
        for value in (0, -1, True, 1.5):
            with self.assertRaises((TypeError, ValueError)):
                create_windows_from_dataframe(x, stride=value)
        with self.assertRaises(ValueError):
            create_windows_from_dataframe(np.zeros(5))


if __name__ == '__main__':
    unittest.main()
