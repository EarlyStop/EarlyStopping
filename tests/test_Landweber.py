import unittest
import numpy as np
import EarlyStopping as es


class Test_landweber(unittest.TestCase):
    """Tests for the landweber algorithm"""

    def setUp(self):
        self.design = np.diag([1, 2, 3, 4, 5])
        self.true_signal = np.random.normal(5, 1, 5)
        self.tol = 10 ** (-7)
        self.response = self.design @ self.true_signal

    def test_matrix_inversion(self):
        self.alg = es.Landweber(self.design, self.response, true_signal=self.true_signal, learning_rate=0.01)
        landweber_estimate = self.alg.get_estimate(500)
        self.assertAlmostEqual(np.sum((landweber_estimate - self.true_signal) ** 2), 0, places=2)

    def test_discrepancy_stop(self):
        self.alg = es.Landweber(self.design, self.response, true_signal=self.true_signal, learning_rate=0.01)
        critical_value = 0.001
        early_stopping_index = self.alg.get_discrepancy_stop(critical_value, max_iteration=1000)
        residual_condition_1 = self.alg.residuals[early_stopping_index - 1] > critical_value
        residual_condition_2 = self.alg.residuals[early_stopping_index] <= critical_value

        self.assertTrue(residual_condition_1 and residual_condition_2)

        landweber_estimate = self.alg.get_estimate(early_stopping_index)
        self.assertAlmostEqual(np.sum((landweber_estimate - self.true_signal) ** 2), 0, places=2)

    def test_weak_oracle(self):
        self.response = self.response + np.random.normal(0, 0.01, 5)
        self.alg = es.Landweber(
            self.design, self.response, true_signal=self.true_signal, learning_rate=0.01, true_noise_level=0.01
        )

        weak_balanced_oracle = self.alg.get_weak_balanced_oracle(1000)
        weak_oracle_condition_1 = (
            self.alg.weak_bias2[weak_balanced_oracle - 1] > self.alg.weak_variance[weak_balanced_oracle - 1]
        )
        weak_oracle_condition_2 = (
            self.alg.weak_bias2[weak_balanced_oracle] <= self.alg.weak_variance[weak_balanced_oracle]
        )

        self.assertTrue(weak_oracle_condition_1 and weak_oracle_condition_2)

    def test_strong_oracle(self):
        self.response = self.response + np.random.normal(0, 0.01, 5)
        self.alg = es.Landweber(
            self.design, self.response, true_signal=self.true_signal, learning_rate=0.01, true_noise_level=0.01
        )

        strong_balanced_oracle = self.alg.get_strong_balanced_oracle(1000)
        strong_oracle_condition_1 = (
            self.alg.strong_bias2[strong_balanced_oracle - 1] > self.alg.strong_variance[strong_balanced_oracle - 1]
        )
        strong_oracle_condition_2 = (
            self.alg.strong_bias2[strong_balanced_oracle] <= self.alg.strong_variance[strong_balanced_oracle]
        )

        self.assertTrue(strong_oracle_condition_1 and strong_oracle_condition_2)


# Running the tests
if __name__ == "__main__":
    unittest.main()
