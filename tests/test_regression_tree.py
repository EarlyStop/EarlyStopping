import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from EarlyStopping import RegressionTree


class TestRegressionTree(unittest.TestCase):
    """Tests for the regression-tree algorithm."""

    def setUp(self):
        self.design = np.array([[0.0], [1.0], [2.0], [3.0]])
        self.response = np.array([0.0, 0.0, 10.0, 10.0])

    def test_expected_root_split_and_depth_one_predictions(self):
        tree = RegressionTree(self.design, self.response, min_samples_split=1)
        tree.iterate(max_depth=2)

        self.assertEqual(tree.regression_tree.variable, 0)
        self.assertEqual(tree.regression_tree.split_threshold, 1.0)
        np.testing.assert_array_equal(tree.predict(self.design, depth=1), self.response)

    def test_residuals_are_nonincreasing(self):
        tree = RegressionTree(self.design, self.response, min_samples_split=1)
        tree.iterate(max_depth=2)

        self.assertIsInstance(tree.residuals, np.ndarray)
        np.testing.assert_array_equal(tree.residuals, np.array([25.0, 0.0]))
        self.assertTrue(np.all(np.diff(tree.residuals) <= 0))

    def test_depth_zero_returns_unconditional_mean(self):
        tree = RegressionTree(self.design, self.response, min_samples_split=1)

        np.testing.assert_array_equal(
            tree.predict(self.design, depth=0),
            np.array([5.0, 5.0, 5.0, 5.0]),
        )

    def test_dataframe_and_ndarray_predictions_match(self):
        tree = RegressionTree(self.design, self.response, min_samples_split=1)
        tree.iterate(max_depth=2)

        array_predictions = tree.predict(self.design, depth=1)
        dataframe_predictions = tree.predict(pd.DataFrame(self.design), depth=1)
        np.testing.assert_allclose(dataframe_predictions, array_predictions)

    def test_oracle_fixture_matches_frozen_results(self):
        true_signal = np.array([0.0, 0.0, 10.0, 10.0])
        noise = np.array([1.0, -1.0, 1.0, -1.0])
        response = true_signal + noise
        tree = RegressionTree(
            self.design,
            response,
            min_samples_split=1,
            true_signal=true_signal,
            true_noise_vector=noise,
        )
        tree.iterate(max_depth=2)

        self.assertEqual(tree.regression_tree.variable, 0)
        self.assertEqual(tree.regression_tree.split_threshold, 1.0)
        np.testing.assert_array_equal(tree.residuals, np.array([26.0, 1.0]))
        np.testing.assert_array_equal(tree.predict(self.design, depth=1), true_signal)
        np.testing.assert_array_equal(tree.bias2, np.array([0.0]))
        np.testing.assert_array_equal(tree.variance, np.array([0.0]))
        np.testing.assert_array_equal(tree.risk, np.array([0.0]))
        self.assertEqual(tree.get_balanced_oracle(), 0)

    def test_theoretical_matrices_are_skipped_without_oracle_inputs(self):
        tree = RegressionTree(self.design, self.response, min_samples_split=1)

        with patch.object(tree, "_block_matrix_processing", wraps=tree._block_matrix_processing) as process:
            tree.iterate(max_depth=2)

        process.assert_not_called()
        self.assertEqual(tree.block_matrix, {})
        self.assertEqual(tree.indices_processed, {})
        self.assertEqual(tree.block_matrices_full, {})

    def test_theoretical_storage_can_be_disabled_with_oracle_inputs(self):
        true_signal = np.array([0.0, 0.0, 10.0, 10.0])
        noise = np.array([1.0, -1.0, 1.0, -1.0])
        tree = RegressionTree(
            self.design,
            true_signal + noise,
            min_samples_split=1,
            true_signal=true_signal,
            true_noise_vector=noise,
            store_theoretical_quantities=False,
        )

        with patch.object(tree, "_block_matrix_processing", wraps=tree._block_matrix_processing) as process:
            tree.iterate(max_depth=2)

        process.assert_not_called()
        np.testing.assert_array_equal(tree.residuals, np.array([26.0, 1.0]))
        self.assertEqual(tree.bias2.size, 0)
        self.assertEqual(tree.variance.size, 0)
        self.assertEqual(tree.risk.size, 0)

    def test_duplicate_features_create_terminal_root(self):
        design = np.ones((4, 2))
        response = np.array([1.0, 2.0, 3.0, 4.0])
        tree = RegressionTree(design, response, min_samples_split=1)
        tree.iterate(max_depth=3)

        self.assertTrue(tree.regression_tree.is_terminal)
        np.testing.assert_array_equal(
            tree.predict(design, depth=1),
            np.repeat(np.mean(response), response.size),
        )

if __name__ == "__main__":
    unittest.main()
