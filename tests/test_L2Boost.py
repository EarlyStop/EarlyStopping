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
        self.X          = np.random.normal(0, 1, size =
                                           (self.sampleSize, self.paraSize))
        self.f          = 15 * self.X[:, 0] + 10 * self.X[:, 1] + \
                           5 * self.X[:, 2]
        self.eps        = np.random.normal(0, 5, self.sampleSize)
        self.Y          = self.f + self.eps

    def testSampleSize(self):
        self.alg = L2Boost(self.X, self.Y)
        self.assertTrue(self.alg.sampleSize == self.alg.paraSize)

# if __name__ == '__main__':
#     unittest.main()




