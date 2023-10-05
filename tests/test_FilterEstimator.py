import unittest
import random
from EarlyStopping import FilterEstimator

class TestFilterEstimator(unittest.TestCase):

    def test_cutoff_estimation(self):
        D = 10  # Data length
        lambda_ = list(range(1, D+1))  # Creating a sequence from 1 to D
        mu = [random.gauss(0, 1) for _ in range(D)]  # Generating D random numbers from a normal distribution
        Y = [a*b for a, b in zip(lambda_, mu)]  # Multiplying lambda and mu element-wise

        for m in range(1, D+1):
            estimator = FilterEstimator(Y, lambda_)
            result = estimator.fEst(m, filt="cutoff")
            expected = mu[:m] + [0] * (D - m)  # mu[1:m] followed by D-m zeros

            self.assertEqual(result, expected)

    def test_landweber_estimation(self):
        D = 10  # Data length
        lambda_ = [1] * D  # A list of D ones
        mu = [random.gauss(0, 1) for _ in range(D)]  # Generating D random numbers from a normal distribution
        Y = [a*b for a, b in zip(lambda_, mu)]  # Multiplying lambda and mu element-wise

        estimator = FilterEstimator(Y, lambda_)
        result = estimator.fEst(15, filt="landw")

        self.assertEqual(result, mu)

# Running the tests
#if __name__ == '__main__':
#    unittest.main()

