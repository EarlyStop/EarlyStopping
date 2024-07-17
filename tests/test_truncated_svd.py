import unittest
import numpy as np
import EarlyStopping as es
from scipy.sparse.linalg import svds

class TestTruncatedSVD(unittest.TestCase):
    def setUp(self):
        # setUp is a class from unittest
        # Simulate data
        self.sample_size = 5
        self.para_size = 5

    def test_diagonal_inversion_without_noise(self):
        design = np.diag(np.random.normal(0, 1, size = self.sample_size))
        signal = np.random.uniform(0, 1, size = self.sample_size)
        noiseless_response = design @ signal
        alg = es.TruncatedSVD(design, noiseless_response)
        alg.iterate(self.sample_size)
        self.assertAlmostEqual(np.mean(alg.truncated_svd_estimate - signal), 0, places=5)

    def test_inversion_without_noise(self):
        design = np.random.normal(0, 1, 
                                  size = (self.sample_size, self.sample_size))
        signal = np.random.uniform(0, 1, size = self.sample_size)
        noiseless_response = design @ signal
        alg = es.TruncatedSVD(design, noiseless_response)
        alg.iterate(self.sample_size)
        self.assertAlmostEqual(np.mean(alg.truncated_svd_estimate - signal), 0, places=5)

    def test_residuals_go_to_zero(self):
        design = np.random.normal(0, 1, 
                                  size = (self.sample_size, self.sample_size))
        signal = np.random.uniform(0, 1, size = self.sample_size)
        noiseless_response = design @ signal
        alg = es.TruncatedSVD(design, noiseless_response)
        alg.iterate(self.sample_size)
        self.assertAlmostEqual(alg.residuals[alg.iteration], 0, places=5)


        
if __name__ == '__main__':
     unittest.main()
