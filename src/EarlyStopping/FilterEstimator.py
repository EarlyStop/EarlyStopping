


class FilterEstimator:

    def __init__(self, Y, lambda_):
        # Initialize with observed data, design matrix, and zero-filled estimation.
        self.Y = Y
        self.lambda_ = lambda_
        self.D = len(Y)
        self.muHat = [0] * self.D

    def cutoff(self, m):
        # Estimation using cutoff method for m > 0.
        if m > 0:
            for i in range(m):
                self.muHat[i] = self.Y[i] / self.lambda_[i]

    def landw(self, m):
        # Estimation using Landweber method for m iterations.
        iter = 0
        while iter < m:
            for i in range(self.D):
                self.muHat[i] += self.lambda_[i] * (self.Y[i] - self.lambda_[i] * self.muHat[i])
            iter += 1

    def fEst(self, m, filt="cutoff"):
        # Choose processing method (cutoff or Landweber) based on user input.
        valid_filters = ["cutoff", "landw"]
        if filt not in valid_filters:
            raise ValueError(f"Invalid method choice. Choose from {valid_filters}")

        if filt == "cutoff":
            self.cutoff(m)
        elif filt == "landw":
            self.landw(m)

        return self.muHat
