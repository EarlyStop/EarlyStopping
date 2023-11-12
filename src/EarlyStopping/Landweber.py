import numpy as np

class Landweber:
    """
    A class to perform estimation using the Landweber iterative method.

    Parameters
    ----------
    Y : array
        Observed data.
    lambda_ : array
        Lambda values used in the estimation.

    Attributes
    ----------
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


    def estimate(self, m):
        """
        Perform estimation using the Landweber method for m iterations.

        Parameters
        ----------
        m : int
            Number of iterations for the Landweber estimation.

        Returns
        -------
        array
            The estimated values after m iterations.
        """

        for _ in range(m):
            self.muHat += self.lambda_ * (self.Y - self.lambda_ * self.muHat)

        return self.muHat


