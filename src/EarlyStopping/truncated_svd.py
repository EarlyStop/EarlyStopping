import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import svds

class TruncatedSVD:
    def __init__(self,
        design,
        response,
        critical_value=None,
        true_signal=None,
        true_noise_level=None,
    ):
        self.design = design
        self.reduced_design = design
        self.response = response
        self.critical_value = critical_value
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level

        # TODO: preprocessing so that SVD is applied for preprocessing
        self.diagonal_design = np.array([])
        self.sample_size = np.shape(self.design)[0]
        self.parameter_size = np.shape(self.design)[1]
        self.diagonal_response = np.array([])
        self.diagonal_estimate = np.array([])
        self.eigenvector_matrix = np.empty((self.parameter_size, 0))

        self.iteration = 0
        self.truncated_svd_estimate = np.zeros(self.parameter_size)
        

    def iterate(self, number_of_iterations):
        for _ in range(number_of_iterations):
            self.__truncated_SVD_one_iteration()
    
    def __truncated_SVD_one_iteration(self):
        u, s, vh = svds(self.reduced_design, k=1)
        print(f"s: {s}")

        self.diagonal_design = np.append(self.diagonal_design, s)
        self.diagonal_response = np.append(self.diagonal_response, vh @ self.response)
        self.eigenvector_matrix = np.append(self.eigenvector_matrix, vh.transpose(), axis=1)

        self.reduced_design = self.reduced_design - s * u @ vh

        self.diagonal_estimate = np.append(self.diagonal_estimate, self.diagonal_response[self.iteration] / s)
        self.truncated_svd_estimate += self.diagonal_estimate[self.iteration] * vh.flatten()
        # self.truncated_svd_estimate[self.iteration] = self.diagonal_response[self.iteration] / self.diagonal_design[self.iteration]
        self.iteration += 1
