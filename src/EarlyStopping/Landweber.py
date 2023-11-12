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