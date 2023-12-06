import unittest
import numpy as np
import EarlyStopping as es

class Test_landweber(unittest.TestCase):
    """Tests for the landweber algorithm"""

    def setUp(self):
        self.para_size = 10
        self.f = np.random.normal(0, 1, self.para_size)  
        
    def test_elementary_estimation(self):
        X = np.eye(self.para_size)  
        Y = X @ self.f  
        iter = 30
        alg = es.landweber(X, Y)
        alg.landweber(iter)
        beta = alg.landweber_estimate
    
        for i in range(len(beta)):
            self.assertAlmostEqual(beta[i], self.f[i], places=7)

 #Running the tests
#if __name__ == '__main__':
#    unittest.main()