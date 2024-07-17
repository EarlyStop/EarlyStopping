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
        self.response = response
        self.critical_value = critical_value
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level

        self.sample_size = np.shape(self.design)[0]
        self.parameter_size = np.shape(self.design)[1]

        self.diagonal_design = np.array([])
        self.diagonal_response = np.array([])
        self.diagonal_estimate = np.array([])

        self.reduced_design = design
        self.eigenvector_matrix = np.empty((self.parameter_size, 0))

        self.iteration = 0
        self.residuals = np.array([np.sum(self.response**2)])
        self.truncated_svd_estimate = np.zeros(self.parameter_size)
        

    def iterate(self, number_of_iterations):
        for _ in range(number_of_iterations):
            self.__truncated_SVD_one_iteration()
    
    def __truncated_SVD_one_iteration(self):
        # Get next singular triplet
        u, s, vh = svds(self.reduced_design, k=1)

        # Get diagonal sequence model quantities
        self.diagonal_design = np.append(self.diagonal_design, s)
        self.diagonal_response = np.append(self.diagonal_response, u.transpose() @ self.response)
        self.eigenvector_matrix = np.append(self.eigenvector_matrix, vh.transpose(), axis=1)
        self.diagonal_estimate = np.append(self.diagonal_estimate,
                                           self.diagonal_response[self.iteration] / s)

        # Update full model quantities
        self.truncated_svd_estimate += self.diagonal_estimate[self.iteration] * vh.flatten()
        new_residual = self.residuals[self.iteration] - (u.transpose() @ self.response)**2
        self.residuals = np.append(self.residuals, new_residual)

        # Reduce design by one eigen triplet 
        self.reduced_design = self.reduced_design - s * u @ vh
        
        self.iteration += 1
