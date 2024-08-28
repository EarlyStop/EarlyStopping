import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import svds

class TruncatedSVD:
    """

    **Parameters**

    *design*: ``array``. design matrix of the linear model. ( :math:`A \in \mathbb{R}^{D \\times p}` )

    *response*: ``array``. n-dim vector of the observed data in the linear model. ( :math:`Y \in \mathbb{R}^{D}` )

    *true_signal*: ``array, default = None``.  p-dim vector For simulation purposes only. For simulated data the true signal can be included to compute theoretical quantities such as the bias and the mse alongside the iterative procedure. ( :math:`f \in \mathbb{R}^{p}` )

    *true_noise_level*: ``float, default = None`` For simulation purposes only. Corresponds to the standard deviation of normally distributed noise contributing to the response variable. Allows the analytic computation of the strong and weak variance. ( :math:`\delta \geq 0` )

    **Attributes**

    *iteration*: ``int``. Current iteration of the algorithm ( :math:`m \in \mathbb{N}` )

    *sample_size*: ``int``. Sample size of the linear model ( :math:`D \in \mathbb{N}` )

    *parameter_size*: ``int``. Parameter size of the linear model ( :math:`p \in \mathbb{N}` )

    *residuals*: ``array``. Lists the sequence of the squared residuals between the observed data and the estimator.

    **Methods**

    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | iterate(``number_of_iterations=1``)                     | Performs a specified number of iterations of the Landweber algorithm.    |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_estimate(``iteration``)                             | Returns the truncated SVD estimator at iteration.                        |
    +---------------------------------------------------------+--------------------------------------------------------------------------+

    """
    def __init__(self,
        design,
        response,
        true_signal = None,
        true_noise_level = None,
    ):

        self.design = design
        self.response = response
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level

        # Parameters of the model
        self.sample_size = np.shape(self.design)[0]
        self.parameter_size = np.shape(self.design)[1]

        self.iteration = 0

        # Quantities in terms of the SVD
        self.diagonal_design = np.array([])
        self.diagonal_response = np.array([])
        self.diagonal_estimate = np.array([])

        self.reduced_design = design
        self.eigenvector_matrix = np.empty((self.parameter_size, 0))

        self.residuals = np.array([np.sum(self.response**2)])
        self.truncated_svd_estimate_list = [np.zeros(self.parameter_size)]

    def iterate(self, number_of_iterations):
        """Performs number_of_iterations iterations of the algorithm

        **Parameters**

        *number_of_iterations*: ``int``. The number of iterations to perform.
        """
        for _ in range(number_of_iterations):
            self.__truncated_SVD_one_iteration()

    def get_estimate(self, iteration):
        """Returns the truncated SVD estimate at iteration.

        **Parameters**

        *iteration*: ``int``. The iteration at which the estimate is requested.

        **Returns**

        *truncated_svd_estimate*: ``ndarray``. The truncated svd estimate at iteration.
        """
        if iteration > self.iteration:
            self.iterate(iteration - self.iteration)

        truncated_svd_estimate = self.truncated_svd_estimate_list[iteration]
        return truncated_svd_estimate
    
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
        new_truncated_svd_estimate = self.truncated_svd_estimate_list[self.iteration] + self.diagonal_estimate[self.iteration] * vh.flatten()
        self.truncated_svd_estimate_list.append(new_truncated_svd_estimate) 
        new_residual = self.residuals[self.iteration] - (u.transpose() @ self.response)**2
        self.residuals = np.append(self.residuals, new_residual)

        # Reduce design by one eigen triplet 
        self.reduced_design = self.reduced_design - s * u @ vh
        
        self.iteration += 1

    def discrepancy_stop(self, critical_value, max_iteration):
        """ Early stopping for the SVD procedure based on the discrepancy principle. Procedure is
            stopped when the residuals go below the critical value or max_iteration is reached.

            **Parameters**

            *critical_value*: ``float``. Critical value for the early stopping procedure.

            *max_iteration*: ``int``. Maximal number of iterations to be performed.
        """
        while self.residuals[self.iteration] > critical_value and self.iteration < max_iteration:
            self.__truncated_SVD_one_iteration()
