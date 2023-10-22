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

    def cutoff(self, m):
        """
        Estimation using the cutoff method for m > 0.

        Parameters:
        -----------
        m : int
            Number of elements to consider for the cutoff estimation.

        """
        if m > 0:
            self.muHat[:m] = self.Y[:m] / self.lambda_[:m]

    def landw(self, m):
        """
        Estimation using the Landweber method for m iterations.

        Parameters:
        -----------
        m : int
            Number of iterations for the Landweber estimation.

        """
        iter = 0
        while iter < m:
            self.muHat += self.lambda_ * (self.Y - self.lambda_ * self.muHat)
            iter += 1

    def fEst(self, m, filt=EstimationMethod.CUTOFF):
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
        if filt == EstimationMethod.CUTOFF:
            self.cutoff(m)
        elif filt == EstimationMethod.LANDWEBER:
            self.landw(m)
        else:
            raise ValueError(f"Invalid method choice. Choose from {[e.name for e in EstimationMethod]}")

        return self.muHat
