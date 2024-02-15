import unittest
import numpy as np
from EarlyStopping import L2_boost

class Test_L2_boost(unittest.TestCase):
    """Tests for the L2_boost algorithm"""

    def setUp(self):
        # setUp is a class from unittest
        # Simulate data
        self.sample_size = 5
        self.para_size   = 5
        self.X           = np.random.normal(0, 1, size = (self.sample_size, self.para_size))
        self.f           = 15 * self.X[:, 0] + 10 * self.X[:, 1] + 5 * self.X[:, 2]
        self.eps         = np.random.normal(0, 5, self.sample_size)
        self.Y           = self.f + self.eps
        self.tol         = 10**(-5)

    def test_termination_of_the_algorithm(self):
        self.alg = L2_boost(self.X, self.Y)
        self.alg.iterate(self.alg.sample_size + 1)
        self.assertTrue(self.alg.iter < self.alg.sample_size + 1)

    def test_orthonormalization(self):
        self.alg = L2_boost(self.X, self.Y)
        self.alg.iterate(self.alg.sample_size)
        for m in range(self.alg.iter):
            direction_m = self.alg.orth_directions[m]
            deviation_vector = np.zeros(self.alg.sample_size)
            for j in range(self.alg.iter):
                direction_j = self.alg.orth_directions[j]
                if j == m:
                    deviation_vector[j] = 1 - np.mean(direction_j**2)
                else:
                    deviation_vector[j] = np.dot(direction_m, direction_j) / self.alg.sample_size
            condition_vector = ( np.absolute(deviation_vector) > self.tol )
            number_of_deviations_larger_tol = np.sum(condition_vector)
            self.assertTrue(number_of_deviations_larger_tol == 0)

    def test_monotonicity_of_bias_and_variance(self):
        self.alg = L2_boost(self.X, self.Y, true_signal = self.f)
        self.alg.iterate(self.sample_size)
        for m in range((self.alg.iter - 1)):
            if self.alg.bias2[m] >= self.tol:
                self.assertTrue(self.alg.bias2[m] >= self.alg.bias2[(m + 1)])
            self.assertTrue(self.alg.stoch_error[m] <= self.alg.stoch_error[(m + 1)])

    def test_consistency_of_bias_variance_computation(self):
        self.alg = L2_boost(self.X, self.Y, true_signal = self.f)
        self.alg.iterate(self.alg.sample_size)
        alternative_computation_mse = self.alg.bias2 + self.alg.stoch_error
        deviation_vector = np.abs(alternative_computation_mse - self.alg.mse)
        for m in range(self.alg.iter):
            self.assertTrue(deviation_vector[m] < self.tol)

    def test_limit_of_the_stochastic_error(self):
        self.alg = L2_boost(self.X, self.Y, true_signal = self.f)
        self.alg.iterate(self.alg.sample_size)
        avg_squared_error = np.mean(self.eps**2)
        last_index = self.alg.iter
        deviation = np.abs(avg_squared_error - self.alg.stoch_error[last_index])
        self.assertTrue(deviation < self.tol)

    def test_boost_to_balanced_oracle(self):
        self.alg = L2_boost(self.X, self.Y, true_signal = self.f)
        self.alg.boost_to_balanced_oracle()
        self.assertTrue(self.alg.bias2[self.alg.iter] <= self.alg.stoch_error[self.alg.iter])
        self.assertTrue(self.alg.bias2[(self.alg.iter - 1)] > self.alg.stoch_error[(self.alg.iter -1)])

# if __name__ == '__main__':
#     unittest.main()
