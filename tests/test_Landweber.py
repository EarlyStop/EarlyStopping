import unittest
import numpy as np
import EarlyStopping as es

class Test_landweber(unittest.TestCase):
    """Tests for the landweber algorithm"""

    def setUp(self):
        self.para_size = 10
        self.f = np.random.normal(0, 1, self.para_size)  
        self.iteration = 30
        self.tol = 10**(-7)

        
    def test_elementary_estimation(self):
        X = np.eye(self.para_size)  
        Y = X @ self.f  
        alg = es.landweber(X, Y)
        alg.landweber(self.iteration)
        beta = alg.landweber_estimate
    
        for i in range(len(beta)):
            deviation = np.abs(beta[i] - self.f[i])
            self.assertTrue(deviation < self.tol)


    def test_monotonicity_residuals(self):
        indices = np.arange(self.para_size)+1
        X = np.diag(1/(np.sqrt(indices)))
        f = 5*np.exp(-0.1*indices)
        Y = X @ f
        alg = es.landweber(X, Y)
        alg.landweber(self.iteration)

        for m in range((alg.iter - 1)):
            self.assertTrue(alg.residuals[m] >= alg.residuals[(m + 1)])




 #Running the tests
if __name__ == '__main__':
    unittest.main()

