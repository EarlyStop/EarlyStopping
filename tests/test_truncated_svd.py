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
        self.design = np.diag(np.random.normal(0, 1, size = self.sample_size))
        u, s, vh = svds(self.design, k=4)
        print(s)
        self.signal = np.random.uniform(0, 1, size = self.sample_size)
        self.noise = np.random.normal(0, 5, self.sample_size)
        self.noiseless_response = self.design @ self.signal
        self.response = self.noiseless_response + self.noise

    def test_inversion_without_noise(self):
        self.alg = es.TruncatedSVD(self.design, self.noiseless_response)
        self.alg.iterate(self.sample_size)
        print(f"signal {self.signal}")
        print(f"estimate {self.alg.truncated_svd_estimate}")
        self.assertAlmostEqual(np.mean(self.alg.truncated_svd_estimate - self.signal), 0, places=5)

        
if __name__ == '__main__':
     unittest.main()