import unittest
import numpy as np
from EarlyStopping import FilterEstimator, EstimationMethod

class TestFilterEstimator(unittest.TestCase):

    def setUp(self):
        self.D = 10
        self.mu = np.random.normal(0, 1, self.D)  
         

    def test_cutoff_estimation(self):
        """Test the cutoff estimation method for a range of m values."""
        lambda_ = np.arange(1, self.D+1) 
        Y = lambda_ * self.mu 
        for m in range(1, self.D+1):
            estimator = FilterEstimator(Y, lambda_)
            result = estimator.fEst(m, filt=EstimationMethod.CUTOFF)
            expected = np.concatenate((self.mu[:m], np.zeros(self.D - m)))
            
            for r, e in zip(result, expected):
                self.assertAlmostEqual(r, e, places=7)

    def test_landweber_estimation(self):
        """Test the Landweber estimation method."""
        lambda_ = np.ones(self.D)  
        Y = lambda_ * self.mu  
        estimator = FilterEstimator(Y, lambda_)
        result = estimator.fEst(15, filt=EstimationMethod.LANDWEBER)
    
        for i in range(len(result)):
            self.assertAlmostEqual(result[i], self.mu[i], places=7)


# Running the tests
#if __name__ == '__main__':
#    unittest.main()

