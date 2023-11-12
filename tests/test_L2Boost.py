import unittest
import numpy as np

# import sys                               # Needed to find the module
# sys.path.append('../src')
from EarlyStopping import L2Boost
# TODO: Find a cleaner solution for this, see:
# https://stackoverflow.com/questions/4761041/python-import-src-modules-when-running-tests

class TestL2Boost(unittest.TestCase):

    def setUp(self):
        # Simulate data
        self.sampleSize = 5
        self.paraSize   = 5
        self.X          = np.random.normal(0, 1, size = (self.sampleSize, self.paraSize))
        self.f          = 15 * self.X[:, 0] + 10 * self.X[:, 1] + 5 * self.X[:, 2]
        self.eps        = np.random.normal(0, 5, self.sampleSize)
        self.Y          = self.f + self.eps
        self.tol        = 10**(-5)

    def testTerminationOfTheAlgorithm(self):
        self.alg = L2Boost(self.X, self.Y)
        self.alg.boost(self.alg.sampleSize + 1)
        self.assertTrue(self.alg.iter < self.alg.sampleSize + 1)

    def testOrthonormalisation(self):
        self.alg = L2Boost(self.X, self.Y)
        self.alg.boost(self.alg.sampleSize)
        for m in range(self.alg.iter): 
            direction_m = self.alg.orthDirections[m]
            deviationVector = np.zeros(self.alg.sampleSize)
            for j in range(self.alg.iter):
                direction_j = self.alg.orthDirections[j] 
                if j == m:
                    deviationVector[j] = 1 - np.mean(direction_j**2)
                else: 
                    deviationVector[j] = np.dot(direction_m, direction_j) / self.alg.sampleSize
            conditionVector = (np.absolute(deviationVector) > self.tol)
            numberOfDeviationsLargerTol = np.sum(conditionVector)
            self.assertTrue(numberOfDeviationsLargerTol == 0)

    def testMonotonicityOfBiasAndVariance(self):
        self.alg = L2Boost(self.X, self.Y, trueSignal = self.f)
        self.alg.boost(self.sampleSize)
        print(self.alg.bias2)
        for m in range((self.alg.iter - 1)): 
            self.assertTrue(self.alg.bias2[m] >= self.alg.bias2[(m + 1)])
            self.assertTrue(self.alg.stochError[m] <= self.alg.stochError[(m + 1)])

    def testConsistencyOfBiasVarianceComputation(self):
        self.alg = L2Boost(self.X, self.Y, trueSignal = self.f)
        self.alg.boost(self.alg.sampleSize)
        alternativeComputationMse = self.alg.bias2 + self.alg.stochError
        deviationVector = np.abs(alternativeComputationMse - self.alg.mse)
        for m in range(self.alg.iter): 
            self.assertTrue(deviationVector[m] < self.tol)

    def testLimitOfTheStochasticError(self):
        self.alg = L2Boost(self.X, self.Y, trueSignal = self.f)
        self.alg.boost(self.alg.sampleSize)
        avgSquaredError = np.mean(self.eps**2)
        lastIndex = self.alg.iter
        deviation = np.abs(avgSquaredError - self.alg.stochError[lastIndex])
        self.assertTrue(deviation < self.tol)

    def testBoostToBalancedOracle(self):
        self.alg = L2Boost(self.X, self.Y, trueSignal = self.f)
        self.alg.boostToBalancedOracle()
        self.assertTrue(self.alg.bias2[self.alg.iter] <=
                        self.alg.stochError[self.alg.iter])
        self.assertTrue(self.alg.bias2[(self.alg.iter - 1)] >
                        self.alg.stochError[(self.alg.iter -1)])

# if __name__ == '__main__':
#     unittest.main()
