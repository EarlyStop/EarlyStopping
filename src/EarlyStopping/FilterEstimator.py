import numpy as np
from enum import Enum

class EstimationMethod(Enum):
    CUTOFF = 1
    LANDWEBER = 2

class FilterEstimator:
    """
    A class to perform filter-based estimation using either cutoff or Landweber methods.

    Parameters
    -----------
    Y : array
        Observed data.
    lambda_ : array
        Lambda values used in the estimation.

    Attributes
    -----------
    D : int
        Length of the observed data Y.
    muHat : array
        Estimated values, initialized with zeros.

    """

    def __init__(self, Y, lambda_):
        """
        Initialize with observed data, lambda values, and zero-filled estimation.
        """
        self.Y = np.array(Y)
        self.lambda_ = np.array(lambda_)
        self.D = len(Y)
        self.muHat = np.zeros(self.D)

    def fEst(self, m):
        """
        Choose processing method (cutoff or Landweber) based on user input.

        Parameters:
        -----------
        m : int
            Number of elements (for cutoff) or iterations (for Landweber).
        filt : EstimationMethod
            Method to use for estimation. Valid choices are [EstimationMethod.CUTOFF, EstimationMethod.LANDWEBER].

        Returns:
        --------
        array
            Estimated values.

        """

        self.landw(m)
   

        return self.muHat
