import unittest

import numpy as np
from scipy.sparse import dia_matrix

import EarlyStopping as es


class TestTruncatedSVD(unittest.TestCase):
    def setUp(self):
        self.sample_size = 5
        self.parameter_size = 5
        self.rng = np.random.default_rng(42)

    def test_inversion_without_noise(self):
        design = self.rng.normal(0, 1, size=(self.sample_size, self.sample_size))
        signal = self.rng.uniform(0, 1, size=self.sample_size)
        noiseless_response = design @ signal
        algorithm = es.TruncatedSVD(design, noiseless_response)

        algorithm.iterate(self.sample_size)

        np.testing.assert_allclose(
            algorithm.get_estimate(algorithm.iteration),
            signal,
            atol=1e-5,
        )

    def test_diagonal_inversion_without_noise(self):
        diagonal = self.rng.uniform(0.2, 1, size=self.sample_size)
        design = dia_matrix((diagonal[np.newaxis, :], [0]), shape=(self.sample_size, self.sample_size))
        signal = self.rng.uniform(0, 1, size=self.sample_size)
        noiseless_response = design @ signal
        algorithm = es.TruncatedSVD(design, noiseless_response, diagonal=True)

        algorithm.iterate(self.sample_size)

        np.testing.assert_allclose(
            algorithm.get_estimate(algorithm.iteration),
            signal,
            atol=1e-5,
        )

    def test_diagonal_mode_rejects_dense_design(self):
        design = np.diag(self.rng.uniform(0.2, 1, size=self.sample_size))
        signal = self.rng.uniform(0, 1, size=self.sample_size)

        with self.assertRaises(TypeError):
            es.TruncatedSVD(design, design @ signal, diagonal=True)

    def test_monotonicity_of_theoretical_quantities(self):
        design = self.rng.normal(0, 1, size=(self.sample_size, self.sample_size))
        signal = self.rng.uniform(0, 1, size=self.sample_size)
        response = design @ signal + self.rng.normal(0, 0.1, self.sample_size)
        algorithm = es.TruncatedSVD(design, response, true_signal=signal, true_noise_level=0.1)

        for _ in range(self.sample_size):
            algorithm.iterate(1)

            self.assertLessEqual(
                algorithm.weak_bias2[algorithm.iteration],
                algorithm.weak_bias2[algorithm.iteration - 1],
            )
            self.assertLessEqual(
                algorithm.strong_bias2[algorithm.iteration],
                algorithm.strong_bias2[algorithm.iteration - 1],
            )
            self.assertLessEqual(
                algorithm.weak_variance[algorithm.iteration - 1],
                algorithm.weak_variance[algorithm.iteration],
            )
            self.assertLessEqual(
                algorithm.strong_variance[algorithm.iteration - 1],
                algorithm.strong_variance[algorithm.iteration],
            )

    def test_diagonal_monotonicity_of_theoretical_quantities(self):
        diagonal = self.rng.uniform(0.2, 1, size=self.sample_size)
        design = dia_matrix((diagonal[np.newaxis, :], [0]), shape=(self.sample_size, self.sample_size))
        signal = self.rng.uniform(0, 1, size=self.sample_size)
        response = design @ signal + self.rng.normal(0, 0.1, self.sample_size)
        algorithm = es.TruncatedSVD(
            design,
            response,
            true_signal=signal,
            true_noise_level=0.1,
            diagonal=True,
        )

        for _ in range(self.sample_size):
            algorithm.iterate(1)

            self.assertLessEqual(
                algorithm.weak_bias2[algorithm.iteration],
                algorithm.weak_bias2[algorithm.iteration - 1],
            )
            self.assertLessEqual(
                algorithm.strong_bias2[algorithm.iteration],
                algorithm.strong_bias2[algorithm.iteration - 1],
            )
            self.assertLessEqual(
                algorithm.weak_variance[algorithm.iteration - 1],
                algorithm.weak_variance[algorithm.iteration],
            )
            self.assertLessEqual(
                algorithm.strong_variance[algorithm.iteration - 1],
                algorithm.strong_variance[algorithm.iteration],
            )


if __name__ == "__main__":
    unittest.main()
