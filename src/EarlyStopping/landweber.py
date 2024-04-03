import numpy as np
import scipy
from scipy import sparse


class Landweber:
    """
     `[Source] <https://github.com/ESFIEP/EarlyStopping/edit/main/src/EarlyStopping/landweber.py>`_ A class to perform estimation using the Landweber iterative method.

    **Description**

    Consider the *linear model*

    .. math::
        Y = Af + \delta Z,

    where :math:`Z` is a :math:`D`-dimensional normal distribution. The landweber iteration is defined through:

    .. math::
        \hat{f}^{(0)}=\hat{f}_0, \quad \hat{f}^{(m+1)}= \hat{f}^{(m)} + A^{\\top}(Y-A \hat{f}^{(m)}).

    **Parameters**

    *design*: ``array``. design matrix of the linear model. ( :math:`A \in \mathbb{R}^{D \\times p}` )

    *response*: ``array``. n-dim vector of the observed data in the linear model. ( :math:`Y \in \mathbb{R}^{D}` )

    *starting_value*: ``array, default = None``. Determines the zeroth step of the iterative procedure. Default is zero. ( :math:`\hat{f}_0` )

    *true_signal*: ``array, default = None``.  p-dim vector For simulation purposes only. For simulated data the true signal can be included to compute theoretical quantities such as the bias and the mse alongside the iterative procedure. ( :math:`f \in \mathbb{R}^{p}` )

    *true_noise_level*: ``float, default = None`` For simulation purposes only. Corresponds to the standard deviation of normally distributed noise contributing to the response variable. Allows the analytic computation of the strong and weak variance. ( :math:`\delta \geq 0` )

    **Attributes**

    *sample_size*: ``int``. Sample size of the linear model ( :math:`D \in \mathbb{N}` )

    *para_size*: ``int``. Parameter size of the linear model ( :math:`p \in \mathbb{N}` )

    *iter*: ``int``. Current Landweber iteration of the algorithm ( :math:`m \in \mathbb{N}` )

    *early_stopping_index*: ``int``. Early Stopping iteration index. Is set to None if no early stopping is triggered. ( :math:`\hat{m}` )

    *landweber_estimate*: ``array``. Landweber estimate at the current iteration for the data given in design ( :math:`\hat{f}_m` )

    *residuals*: ``array``. Lists the sequence of the squared residuals between the observed data and the Landweber estimator.

    *strong_bias2*: ``array``. Only exists if true_signal was given. Lists the values of the strong squared bias up to the current Landweber iteration. 

    .. math::
       B^{2}_{m} = \\Vert (I-A^{\\top}A)(f-\hat{f}_{m-1}) \\Vert^{2}

    *strong_variance*: ``array``. Only exists if true_signal was given. Lists the values of the strong variance up to the current Landweber iteration. 
    
    .. math::
        V_m = \\delta^2 \\mathrm{tr}((A^{\\top}A)^{-1}(I-(I-A^{\\top}A)^{m})^{2})

    *strong_error*: ``array``. Only exists if true_signal was given. Lists the values of the strong norm error between the Landweber estimator and the true signal up to the current Landweber iteration.

    .. math::
        E[\\Vert \hat{f}_{m} - f \\Vert^2] = B^{2}_{m} + V_m 

    *weak_bias2*: ``array``. Only exists if true_signal was given. Lists the values of the weak squared bias up to the current Landweber iteration. 
    
    .. math::
        B^{2}_{m,A} = \\Vert A(I-A^{\\top}A)(f-\hat{f}_{m-1}) \\Vert^{2}

    *weak_variance*: ``array``. Only exists if true_signal was given. Lists the values of the weak variance up to the current Landweber iteration. 

    .. math:: 
       V_{m,A} = \\delta^2 \\mathrm{tr}((I-(I-A^{\\top}A)^{m})^{2})

    *weak_error*: ``array``. Only exists if true_signal was given. Lists the values of the weak norm error between the Landweber estimator and the true signal up to the current Landweber iteration.

    .. math::
        E[\\Vert \hat{f}_{m} - f \\Vert_A^2] = B^{2}_{m,A} + V_{m,A} 

    *weak_balanced_oracle*: ``integer``. Only exists if true_signal was given. Gives the stopping iteration, when :math:`B^{2}_{m,A} \leq V_{m,A}`.

    *strong_balanced_oracle*: ``integer``. Only exists if true_signal was given. Gives the stopping iteration, when :math:`B^{2}_{m} \leq V_{m}`.
    
    **Methods**

    +--------------------------------------------+----------------------------------------------------------------------------------+
    | landweber(``iter_num=1``)                  |Performs a specified number of iterations of the Landweber algorithm.             |
    +--------------------------------------------+----------------------------------------------------------------------------------+
    | landweber_to_early_stop(``max_iter``)      |Applies early stopping to the Landweber iterative procedure.                      |
    +--------------------------------------------+----------------------------------------------------------------------------------+
    | landweber_gather_all(``max_iter``)         |Runs the algorithm till max_iter is reached. The early_stopping_index is recorded.|
    +--------------------------------------------+----------------------------------------------------------------------------------+
    """

    def __init__(
        self,
        design,
        response,
        learning_rate=1,
        critical_value=None,
        starting_value=None,
        true_signal=None,
        true_noise_level=None,
    ):

        # self.design = sparse.csr_matrix(design)
        # self.response = np.transpose(sparse.csr_matrix(response))
        self.design = design
        self.response = response
        self.learning_rate = learning_rate
        self.starting_value = starting_value
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level  # sigma
        self.critical_value = critical_value

        # Parameters of the model
        self.sample_size = np.shape(design)[0]
        self.parameter_size = np.shape(design)[1]

        # Determine starting value for the procedure
        if starting_value is None:
            # self.starting_value = np.transpose(sparse.csr_matrix(np.zeros(self.parameter_size)))
            self.starting_value = np.zeros(self.parameter_size)
        else:
            self.starting_value = starting_value

        # Estimation quantities
        self.iteration = 0
        self.landweber_estimate = self.starting_value
        # Collect coefficients:
        self.landweber_estimate_collect = []

        self.early_stopping_index = None

        self.gram_matrix = np.transpose(self.design) @ self.design

        if self.critical_value is None and self.true_noise_level is None:
            # maximum likelihood estimator
            # least_squares_estimator = np.linalg.solve(self.congruency_matrix, np.transpose(self.design) @ self.response)
            sparse_least_squares_estimator, istop = sparse.linalg.lsqr(
                self.gram_matrix, np.transpose(self.design) @ self.response
            )[:2]
            noise_level_estimate = (
                np.sum((self.response - self.design @ sparse_least_squares_estimator) ** 2) / self.sample_size
            )
            self.critical_value = noise_level_estimate * self.sample_size

        elif self.critical_value is None and self.true_noise_level is not None:
            # if the true noise level is given, it does not need to be estimated
            self.critical_value = self.true_noise_level**2 * self.sample_size

        # Residual quantities
        self.__residual_vector = self.response - self.design @ self.starting_value
        self.residuals = np.array([np.sum(self.__residual_vector**2)])

        if (self.true_signal is not None) and (self.true_noise_level is not None):
            # initialize matrices required for computing the strong/weak bias and variance
            self.identity = sparse.dia_matrix(np.eye(self.parameter_size))

            if scipy.sparse.issparse(self.gram_matrix):
                self.inverse_congruency_matrix = scipy.sparse.linalg.inv(self.gram_matrix)
            else:
                self.inverse_congruency_matrix = np.linalg.inv(self.gram_matrix)

            self.perturbation_congruency_matrix = (
                sparse.dia_matrix(np.eye(self.parameter_size)) - self.learning_rate * self.gram_matrix
            )
            self.weak_perturbation_congruency_matrix = self.design @ self.perturbation_congruency_matrix
            self.perturbation_congruency_matrix_power = self.perturbation_congruency_matrix

            # initialize strong/weak bias and variance
            self.expectation_estimator = self.starting_value
            self.strong_bias2 = np.array([np.sum(self.true_signal**2)])
            self.weak_bias2 = np.array([np.sum((self.design @ self.true_signal) ** 2)])

            self.strong_variance = np.array([0])
            self.weak_variance = np.array([0])

            self.strong_error = self.strong_bias2 + self.strong_variance
            self.weak_error = self.weak_bias2 + self.weak_variance

            self.weak_balanced_oracle = 0
            self.strong_balanced_oracle = 0

    def iterate(self, iter_num=1):
        """Performs iter_num iterations of the Landweber algorithm

        **Parameters**

        *iter_num*: ``int, default=1``. The number of iterations to perform.
        """
        for _ in range(iter_num):
            self.__landweber_one_iteration()

    def __update_iterative_matrices(self):
        """Update iterative quantities

        - The expectation of the estimator satisfies
        :math: `m_{k+1} = m_k + hA^{\\top}A(\\mu-m_k)`
        - Compute matrix power iteratively
        :math: `(I-hA^{\\top}A)^{k+1}=(I-hA^{\\top}A)^{k} * (I-hA^{\\top}A)`
        """
        self.expectation_estimator = self.expectation_estimator + self.learning_rate * (self.gram_matrix) @ (
            self.true_signal - self.expectation_estimator
        )

        self.perturbation_congruency_matrix_power = (
            self.perturbation_congruency_matrix_power @ self.perturbation_congruency_matrix
        )

    def __update_strong_bias2(self):
        """Update strong bias

        Given the expectation of the estimator the squared bias is given by :math: `b_k^{2} = \\Vert (I-hA^{\\top}A)(\\mu-m_{k-1}) \\Vert^{2}`
        """
        new_strong_bias2 = np.sum(
            np.square(self.perturbation_congruency_matrix @ (self.true_signal - self.expectation_estimator))
        )
        self.strong_bias2 = np.append(self.strong_bias2, new_strong_bias2)

    def __update_weak_bias2(self):
        """Update weak bias

        Given the expectation of the estimator the (weak)-squared bias is given by :math: `b_k^{2} = \\Vert A(I-hA^{\\top}A)(\\mu-m_{k-1}) \\Vert^{2}`
        """
        new_weak_bias2 = np.sum(
            np.square(self.weak_perturbation_congruency_matrix @ (self.true_signal - self.expectation_estimator))
        )
        self.weak_bias2 = np.append(self.weak_bias2, new_weak_bias2)

    def __update_strong_variance(self):
        """Update strong variance

        The strong variance in the m-th iteration is given by
        :math: `\\sigma**2 \\mathrm{tr}(h^{-1}(A^{\\top}A)^{-1}(I-(I-hA^{\\top}A)^{m})^{2})`
        """

        #presquare_temporary_matrix = self.identity - self.perturbation_congruency_matrix_power
        pretrace_temporary_matrix = (
            self.learning_rate ** (-1)
            * self.inverse_congruency_matrix
            @ (self.identity - self.perturbation_congruency_matrix_power) @ (self.identity - self.perturbation_congruency_matrix_power)
        )

        new_strong_variance = self.true_noise_level**2 * pretrace_temporary_matrix.trace()


        self.strong_variance = np.append(self.strong_variance, new_strong_variance)

    def __update_weak_variance(self):
        """Update weak variance

        The weak variance in the m-th iteration is given by
        :math: `\\sigma**2 \\mathrm{tr}((I-(I-hA^{\\top}A)^{m})^{2})`
        """
        pretrace_temporary_matrix = (self.identity - self.perturbation_congruency_matrix_power) @ (self.identity - self.perturbation_congruency_matrix_power)
        
        new_weak_variance = self.true_noise_level**2 * pretrace_temporary_matrix.trace()

        self.weak_variance = np.append(self.weak_variance, new_weak_variance)

    def __landweber_one_iteration(self):
        """Performs one iteration of the Landweber algorithm"""
        #print(f'The current iteration is {self.iteration}.')
        self.landweber_estimate = self.landweber_estimate + self.learning_rate * np.transpose(self.design) @ (
            self.response - self.design @ self.landweber_estimate
        )
        # Collect coefficients:
        self.landweber_estimate_collect.append(self.landweber_estimate)

        # self.landweber_estimate = (self.landweber_estimate
        #     + self.learning_rate * np.transpose(self.design)
        #     * (self.response - self.design * self.landweber_estimate))

        # Update estimation quantities
        self.__residual_vector = self.response - self.design @ self.landweber_estimate
        new_residuals = np.sum(self.__residual_vector**2)
        self.residuals = np.append(self.residuals, new_residuals)

        if (self.early_stopping_index is None) and (self.residuals[self.iteration] <= self.critical_value):
            self.early_stopping_index = self.iteration

        self.iteration = self.iteration + 1

        if (self.true_signal is not None) and (self.true_noise_level is not None):
            # update weak and strong bias and variance
            self.__update_strong_bias2()
            self.__update_weak_bias2()
            self.__update_strong_variance()
            self.__update_weak_variance()
            self.__update_iterative_matrices()

            # update MSE and weak MSE
            self.strong_error = self.strong_bias2 + self.strong_variance
            self.weak_error = self.weak_bias2 + self.weak_variance

            self.weak_balanced_oracle = (
                (np.argmax(self.weak_bias2 <= self.weak_variance))
                if np.any(self.weak_bias2 <= self.weak_variance)
                else None
            )

            self.strong_balanced_oracle = (
                (np.argmax(self.strong_bias2 <= self.strong_variance))
                if np.any(self.strong_bias2 <= self.strong_variance)
                else None
            )

    # def landweber_to_early_stop(self, max_iter):
    #     """Early stopping for the Landweber procedure

    #         Procedure is stopped when the residuals go below crit or iteration
    #         max_iter is reached.

    #     **Parameters**

    #     *max_iter*: ``int`` The maximum number of iterations to perform.
    #     """
    #     while self.residuals[self.iteration] > self.critical_value and self.iteration < max_iter:
    #         self.__landweber_one_iteration()
    #     self.early_stopping_index = self.iteration

    # def landweber_gather_all(self, max_iter):
    #     """Runs the algorithm till max_iter and gathers all relevant simulation data.
    #     The early stopping index is recorded.

    #     **Parameters**

    #     *max_iter*: ``int`` The maximum number of iterations to perform.
    #     """
    #     self.landweber_to_early_stop(max_iter)
    #     if max_iter > self.early_stopping_index:
    #         self.landweber(max_iter - self.early_stopping_index)
    #     if (self.true_signal is not None) and (self.true_noise_level is not None):
    #         self.weak_balanced_oracle = (
    #             (np.argmax(self.weak_bias2 <= self.weak_variance))
    #             if np.any(self.weak_bias2 <= self.weak_variance)
    #             else None
    #         )
    #         self.strong_balanced_oracle = (
    #             (np.argmax(self.strong_bias2 <= self.strong_variance))
    #             if np.any(self.strong_bias2 <= self.strong_variance)
    #             else None
    #         )
