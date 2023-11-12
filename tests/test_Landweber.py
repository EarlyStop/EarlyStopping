import unittest
import numpy as np
from EarlyStopping import Landweber

class TestFilterEstimator(unittest.TestCase):

    def setUp(self):
        self.D = 10
        self.mu = np.random.normal(0, 1, self.D)  
        
    def test_landweber_estimation(self):
        """Test the Landweber estimation method."""
        lambda_ = np.ones(self.D)  
        Y = lambda_ * self.mu  
        estimator = Landweber(Y, lambda_)
        result = estimator.estimate(15)
    
        for i in range(len(result)):
            self.assertAlmostEqual(result[i], self.mu[i], places=7)


# Running the tests
#if __name__ == '__main__':
#    unittest.main()

