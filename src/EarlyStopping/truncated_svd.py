import numpy as np
import scipy
from scipy import sparse

class TruncatedSVD:
    def __init__(self,
        design,
        response,
        critical_value=None,
        true_signal=None,
        true_noise_level=None,
    ):
        self.design = design
        self.response = response
        self.critical_value = critical_value
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level

        # TODO: preprocessing so that SVD is applied for preprocessing
        self.diagonal_design = self.design
        self.sample_size = self.diagonal_design.size
        self.diagonal_response = self.response

        self.iteration = 0
        self.truncated_svd_estimate = np.zeros(self.sample_size)

    def iterate(self, number_of_iterations):
        for _ in range(number_of_iterations):
            self.__truncated_SVD_one_iteration()
    
    def __truncated_SVD_one_iteration(self):
        self.truncated_svd_estimate[self.iteration] = self.diagonal_response[self.iteration] / self.diagonal_design[self.iteration]
        self.iteration += 1
