import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import svds
import warnings

class TruncatedSVD:
    """
    `[Source] <https://github.com/ESFIEP/EarlyStopping/edit/main/src/EarlyStopping/truncated_svd.py>`_ 
    A class to perform estimation using truncated SVD estimation.

    **Parameters**

    *design*: ``array``. Design matrix of the linear model. ( :math:`A \in \mathbb{R}^{D \\times p}` ).

    *response*: ``array``. n-dim vector of the observed data in the linear model. ( :math:`Y \in \mathbb{R}^{n}` ).

    *true_signal*: ``array, default = None``. p-dim vector of the true signal. For simulation purposes only.
    For simulated data, the true signal can be included to compute theoretical quantities such as
    the bias and the MSE alongside the iterative procedure. ( :math:`f \in \mathbb{R}^{p}` ).

    *true_noise_level*: ``float, default = None`` For simulation purposes only. Corresponds to the
    standard deviation of normally distributed noise contributing to the response variable. Allows
    the analytic computation of the strong and weak variance. ( :math:`\delta \geq 0` ).

    *diagonal*: ``bool, default = False`` The user may set this to true if the design matrix is
    diagonal with strictly positive singular values to avoid unnecessary computation in the diagonal
    sequence space model.
    # 2024-10-14, Bernhard: Checked docu of the parameters. 

    **Attributes**

    *iteration*: ``int``. Current iteration of the algorithm ( :math:`m \in \mathbb{N}` )

    *sample_size*: ``int``. Sample size of the linear model ( :math:`D \in \mathbb{N}` )

    *parameter_size*: ``int``. Parameter size of the linear model ( :math:`p \in \mathbb{N}` )

    *residuals*: ``array``. Lists the sequence of the squared residuals between the observed data
    and the estimator.

    *weak_bias2*: ``array``. Only exists if true_signal was given. Lists the values of the weak
    squared bias up to the current iteration.

    *weak_variance*: ``array``. Only exists if true_signal was given. Lists the values of the weak
    variance up to the current iteration.

    *weak_mse*: ``array``. Only exists if true_signal was given. Lists the values of the weak
    mean squared error up to the current iteration.

    *strong_bias2*: ``array``. Only exists if true_signal was given. Lists the values of the strong
    squared bias up to the current iteration.

    *strong_variance*: ``array``. Only exists if true_signal was given. Lists the values of the
    strong variance up to the current iteration.

    *strong_mse*: ``array``. Only exists if true_signal was given. Lists the values of the strong
    mean squared error up to the current iteration.

    **Methods**

    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | iterate(``number_of_iterations=1``)                     | Performs a specified number of iterations of the Landweber algorithm.    |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_estimate(``iteration``)                             | Returns the truncated SVD estimator at iteration.                        |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_discrepancy_stop(``critical_value, max_iteration``) | Returns the early stopping index according to the discrepancy principle. |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_aic(``max_iteration, K=2``)                         | Returns the iteration chosen by the Akaike information criterion.        |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_weak_balanced_oracle(``max_iteration``)             | Returns the weak balanced oracle if found up to max_iteration.           |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_strong_balanced_oracle(``max_iteration``)           | Returns the strong balanced oracle if found up to max_iteration.         |
    +---------------------------------------------------------+--------------------------------------------------------------------------+

    """
    def __init__(self,
        design,
        response,
        true_signal = None,
        true_noise_level = None,
        diagonal = False
    ):

        self.design = design
        self.response = response
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.diagonal = diagonal

        # Parameters of the model
        self.sample_size = np.shape(self.design)[0]
        self.parameter_size = np.shape(self.design)[1]

        self.iteration = 0

        # Quantities in terms of the SVD
        self.diagonal_design      = np.array([])
        self.diagonal_true_signal = np.array([])
        self.diagonal_response    = np.array([])
        self.diagonal_estimate    = np.array([])

        self.reduced_design = design
        self.eigenvector_matrix = np.empty((self.parameter_size, 0))

        self.residuals = np.array([np.sum(self.response**2)])
        self.truncated_svd_estimate_list = [np.zeros(self.parameter_size)]
        # Mo 14. Okt 14:48:06 CEST 2024, Bernhard: Checked __itit__ until here.

        # Initialize theoretical quantities
        if self.true_signal is not None:
            self.weak_bias2    = np.array([np.sum((self.design @ self.true_signal)**2)])
            self.weak_variance = np.array([0])
            self.weak_mse      = np.array([0])

            self.strong_bias2    = np.array([np.sum(self.true_signal**2)])
            self.strong_variance = np.array([0])
            self.strong_mse      = np.array([0])

    def iterate(self, number_of_iterations):
        """Performs number_of_iterations iterations of the algorithm

        **Parameters**

        *number_of_iterations*: ``int``. The number of iterations to perform.
        """
        if not self.diagonal: 
            for _ in range(number_of_iterations):
                self.__truncated_SVD_one_iteration()
        else:
            for _ in range(number_of_iterations):
                self.__truncated_SVD_one_iteration_diagonal()

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

    def get_discrepancy_stop(self, critical_value, max_iteration):
        """Returns early stopping index based on discrepancy principle up to max_iteration.

        **Parameters**

        *critical_value*: ``float``. The critical value for the discrepancy
        principle. The algorithm stops when :math: `\\Vert Y - A \hat{f}^{(m)}
        \\Vert^{2} \leq \\kappa^{2},` where :math: `\\kappa` is the critical
        value.

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        **Returns**

        *early_stopping_index*: ``int``. The first iteration at which the discrepancy principle is satisfied.  (None is returned if the stopping index is not found.)
        """
        if self.residuals[self.iteration] <= critical_value:

            # argmax takes the first instance of True in the true-false array
            early_stopping_index = np.argmax(self.residuals <= critical_value)
            return int(early_stopping_index)

        if self.residuals[self.iteration] > critical_value:
            while self.residuals[self.iteration] > critical_value and self.iteration <= max_iteration:
                self.iterate(1)

        if self.residuals[self.iteration] <= critical_value:
            early_stopping_index = self.iteration
            return early_stopping_index
        else:
            warnings.warn("Early stopping index not found up to max_iteration. Returning None.", category=UserWarning)
            return None

    def get_aic(self, max_iteration, K = 2):
        """Returns the iteration chosen by the Akaike information criterion computed up max_iteration.

        **Parameters**

        *K*: ``float``. Constant in the definition of the AIC.

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        **Returns**

        *aic_index*: ``int``. Minimiser of the AIC criterion.
        """
        if max_iteration > self.iteration:
            self.iterate(max_iteration - self.iteration)

        if self.diagonal:
            self.diagonal_response = self.response

        data_fit_term = - np.cumsum(self.diagonal_response[:max_iteration]**2 / self.diagonal_design[:max_iteration]**2)
        penalty_term  = K * self.true_noise_level**2 * np.cumsum(self.diagonal_design[:max_iteration]**(-2))
        aic           = data_fit_term + penalty_term 
        aic_index     = np.argmin(aic)

        return [aic_index, aic]

    def get_weak_balanced_oracle(self, max_iteration):
        """Returns weak balanced oracle if found up to max_iteration.

        **Parameters**

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        **Returns**

        *weak_balanced_oracle*: ``int``. The first iteration at which the weak bias is smaller than the weak variance.
        """
        if self.weak_bias2[self.iteration] <= self.weak_variance[self.iteration]:
            # argmax takes the first instance of True in the true-false array
            weak_balanced_oracle = np.argmax(self.weak_bias2 <= self.weak_variance)
            return int(weak_balanced_oracle)

        if self.weak_bias2[self.iteration] > self.weak_variance[self.iteration]:
            while (
                self.weak_bias2[self.iteration] > self.weak_variance[self.iteration] and self.iteration <= max_iteration
            ):
                self.iterate(1)

        if self.weak_bias2[self.iteration] <= self.weak_variance[self.iteration]:
            weak_balanced_oracle = self.iteration
            return weak_balanced_oracle
        else:
            warnings.warn("Weakly balanced oracle not found up to max_iteration. Returning None.", category=UserWarning)
            return None

    def get_strong_balanced_oracle(self, max_iteration):
        """Returns strong balanced oracle if found up to max_iteration.

        **Parameters**

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        **Returns**

        *strong_balanced_oracle*: ``int``. The first iteration at which the strong bias is smaller than the strong variance.
        """
        if self.strong_bias2[self.iteration] <= self.strong_variance[self.iteration]:
            # argmax takes the first instance of True in the true-false array
            strong_balanced_oracle = np.argmax(self.strong_bias2 <= self.strong_variance)
            return int(strong_balanced_oracle)

        if self.strong_bias2[self.iteration] > self.strong_variance[self.iteration]:
            while (
                self.strong_bias2[self.iteration] > self.strong_variance[self.iteration]
                and self.iteration <= max_iteration
            ):
                self.iterate(1)

        if self.strong_bias2[self.iteration] <= self.strong_variance[self.iteration]:
            strong_balanced_oracle = self.iteration
            return strong_balanced_oracle
        else:
            warnings.warn("Weakly balanced oracle not found up to max_iteration. Returning None.", category=UserWarning)
            return None

    def __truncated_SVD_one_iteration(self):
        # Get next singular triplet
        u, s, vh = svds(self.reduced_design, k=1)

        # Get diagonal sequence model quantities
        self.diagonal_design    = np.append(self.diagonal_design, s)
        self.diagonal_response  = np.append(self.diagonal_response, u.transpose() @ self.response)
        self.eigenvector_matrix = np.append(self.eigenvector_matrix, vh.transpose(), axis=1)
        self.diagonal_estimate  = np.append(self.diagonal_estimate,
                                            self.diagonal_response[self.iteration] / s)

        # Update full model quantities
        new_truncated_svd_estimate = self.truncated_svd_estimate_list[self.iteration] + \
                                     self.diagonal_estimate[self.iteration] * vh.flatten()
        self.truncated_svd_estimate_list.append(new_truncated_svd_estimate) 

        new_residual   = self.residuals[self.iteration] - (u.transpose() @ self.response)**2
        self.residuals = np.append(self.residuals, new_residual)

        # Reduce design by one eigen triplet 
        self.reduced_design = self.reduced_design - s * u @ vh

        # Updating theoretical quantities
        if self.true_signal is not None:
            self.diagonal_true_signal = np.append(self.diagonal_true_signal, u.transpose() @ self.true_signal)

            new_weak_bias2  = self.weak_bias2[self.iteration] - s**2 * self.diagonal_true_signal[self.iteration]**2
            self.weak_bias2 = np.append(self.weak_bias2, new_weak_bias2)

            new_weak_variance  = self.weak_variance[self.iteration] + self.true_noise_level**2
            self.weak_variance = np.append(self.weak_variance, new_weak_variance)

            new_weak_mse = new_weak_bias2 + new_weak_variance
            self.weak_mse = np.append(self.weak_mse, new_weak_mse)

            new_strong_bias2  = self.strong_bias2[self.iteration] - self.diagonal_true_signal[self.iteration]**2
            self.strong_bias2 = np.append(self.strong_bias2, new_strong_bias2)

            new_strong_variance = self.strong_variance[self.iteration] + self.true_noise_level**2 / s**2
            self.strong_variance = np.append(self.strong_variance, new_strong_variance)

            new_strong_mse = new_strong_bias2 + new_strong_variance
            self.strong_mse = np.append(self.strong_mse, new_strong_mse)
        
        self.iteration += 1
        # Mo 21. Okt, Checked one_iteration until here.

    def __truncated_SVD_one_iteration_diagonal(self):
        s = self.design[self.iteration, self.iteration]
        self.diagonal_design = np.append(self.diagonal_design, s)

        # Estimator update
        standard_basis_vector                 = np.zeros(self.parameter_size)
        standard_basis_vector[self.iteration] = 1.0
        new_truncated_svd_estimate            = self.truncated_svd_estimate_list[self.iteration] + \
                                                self.response[self.iteration] / s * standard_basis_vector
        self.truncated_svd_estimate_list.append(new_truncated_svd_estimate) 

        # Residual update
        new_residual   = self.residuals[self.iteration] - self.response[self.iteration]**2
        self.residuals = np.append(self.residuals, new_residual)

        # Updating theoretical quantities
        if self.true_signal is not None:
            new_weak_bias2  = self.weak_bias2[self.iteration] - s**2 * self.true_signal[self.iteration]**2
            self.weak_bias2 = np.append(self.weak_bias2, new_weak_bias2)

            new_weak_variance  = self.weak_variance[self.iteration] + self.true_noise_level**2
            self.weak_variance = np.append(self.weak_variance, new_weak_variance)

            new_weak_mse = new_weak_bias2 + new_weak_variance
            self.weak_mse = np.append(self.weak_mse, new_weak_mse)

            new_strong_bias2  = self.strong_bias2[self.iteration] - self.true_signal[self.iteration]**2
            self.strong_bias2 = np.append(self.strong_bias2, new_strong_bias2)

            new_strong_variance  = self.strong_variance[self.iteration] + self.true_noise_level**2 / s**2
            self.strong_variance = np.append(self.strong_variance, new_strong_variance)

            new_strong_mse = new_strong_bias2 + new_strong_variance
            self.strong_mse = np.append(self.strong_mse, new_strong_mse)

        self.iteration += 1
