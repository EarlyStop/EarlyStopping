import unittest

import numpy as np

from EarlyStopping import L2_boost


class TestL2Boost(unittest.TestCase):
    """Tests for the L2-boosting algorithm."""

    def setUp(self):
        self.sample_size = 5
        self.parameter_size = 5
        self.design = np.eye(self.sample_size)
        self.true_signal = np.array([15.0, 10.0, 5.0, 2.0, 1.0])
        self.noise = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        self.response = self.design @ self.true_signal + self.noise
        self.tolerance = 1e-10

    def test_termination_of_the_algorithm(self):
        algorithm = L2_boost(self.design, self.response)
        algorithm.iterate(algorithm.sample_size + 1)

        self.assertLess(algorithm.iteration, algorithm.sample_size + 1)

    def test_orthonormalization(self):
        algorithm = L2_boost(self.design, self.response)
        algorithm.iterate(algorithm.sample_size)

        directions = np.asarray(algorithm.orth_directions)
        empirical_gram_matrix = directions @ directions.T / algorithm.sample_size
        np.testing.assert_allclose(
            empirical_gram_matrix,
            np.eye(algorithm.iteration),
            atol=self.tolerance,
        )

    def test_monotonicity_of_bias_and_stochastic_error(self):
        algorithm = L2_boost(self.design, self.response, true_signal=self.true_signal)
        algorithm.iterate(self.sample_size)

        self.assertTrue(np.all(np.diff(algorithm.bias2) <= self.tolerance))
        self.assertTrue(np.all(np.diff(algorithm.stochastic_error) >= -self.tolerance))

    def test_consistency_of_risk_decomposition(self):
        algorithm = L2_boost(self.design, self.response, true_signal=self.true_signal)
        algorithm.iterate(algorithm.sample_size)

        np.testing.assert_allclose(
            algorithm.bias2 + algorithm.stochastic_error,
            algorithm.risk,
            atol=self.tolerance,
        )

    def test_limit_of_the_stochastic_error(self):
        algorithm = L2_boost(self.design, self.response, true_signal=self.true_signal)
        algorithm.iterate(algorithm.sample_size)

        self.assertAlmostEqual(
            algorithm.stochastic_error[algorithm.iteration],
            np.mean(self.noise**2),
        )

    def test_get_balanced_oracle(self):
        algorithm = L2_boost(self.design, self.response, true_signal=self.true_signal)
        balanced_oracle = algorithm.get_balanced_oracle(max_iteration=self.sample_size)

        self.assertEqual(balanced_oracle, 4)
        self.assertEqual(algorithm.iteration, balanced_oracle)
        self.assertLessEqual(
            algorithm.bias2[balanced_oracle],
            algorithm.stochastic_error[balanced_oracle],
        )
        self.assertGreater(
            algorithm.bias2[balanced_oracle - 1],
            algorithm.stochastic_error[balanced_oracle - 1],
        )


if __name__ == "__main__":
    unittest.main()
