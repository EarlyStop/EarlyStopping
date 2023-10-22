import unittest
import numpy as np
from EarlyStopping import FilterEstimator, EstimationMethod

class TestFilterEstimator(unittest.TestCase):

    def setUp(self):
        self.D = 10
        self.lambda_ = np.arange(1, self.D+1)  
        self.mu = np.random.normal(0, 1, self.D)  
        self.Y = self.lambda_ * self.mu  

    def test_cutoff_estimation(self):
        """Test the cutoff estimation method for a range of m values."""
        for m in range(1, self.D+1):
            estimator = FilterEstimator(self.Y, self.lambda_)
            result = estimator.fEst(m, filt=EstimationMethod.CUTOFF)
            expected = np.concatenate((self.mu[:m], np.zeros(self.D - m)))
            
            for r, e in zip(result, expected):
                self.assertEqual(r, e)

    def test_landweber_estimation(self):
        """Test the Landweber estimation method."""
        lambda_ = np.ones(self.D)  
        estimator = FilterEstimator(self.Y, lambda_)
        result = estimator.fEst(15, filt=EstimationMethod.LANDWEBER)
        
        for r, e in zip(result, self.mu):
            self.assertAlmostEqual(r, e, places=7)  

# Running the tests
#if __name__ == '__main__':
#    unittest.main()
