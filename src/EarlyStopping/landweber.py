import numpy as np
import scipy
from scipy import sparse
import warnings


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

    *initial_value*: ``array, default = None``. Determines the zeroth step of the iterative procedure. Default is zero. ( :math:`\hat{f}_0` )

    *true_signal*: ``array, default = None``.  p-dim vector For simulation purposes only. For simulated data the true signal can be included to compute theoretical quantities such as the bias and the mse alongside the iterative procedure. ( :math:`f \in \mathbb{R}^{p}` )

    *true_noise_level*: ``float, default = None`` For simulation purposes only. Corresponds to the standard deviation of normally distributed noise contributing to the response variable. Allows the analytic computation of the strong and weak variance. ( :math:`\delta \geq 0` )

    **Attributes**

    *sample_size*: ``int``. Sample size of the linear model ( :math:`D \in \mathbb{N}` )

    *parameter_size*: ``int``. Parameter size of the linear model ( :math:`p \in \mathbb{N}` )

    *iteration*: ``int``. Current Landweber iteration of the algorithm ( :math:`m \in \mathbb{N}` )

    *residuals*: ``array``. Lists the sequence of the squared residuals between the observed data and the Landweber estimator.

    *strong_bias2*: ``array``. Only exists if true_signal was given. Lists the values of the strong squared bias up to the current Landweber iteration.

    .. math::
       B^{2}_{m} = \\Vert (I-A^{\\top}A)(f-\hat{f}_{m-1}) \\Vert^{2}

    *strong_variance*: ``array``. Only exists if true_signal was given. Lists the values of the strong variance up to the current Landweber iteration.

    .. math::
        V_m = \\delta^2 \\mathrm{tr}((A^{\\top}A)^{-1}(I-(I-A^{\\top}A)^{m})^{2})

    *strong_risk*: ``array``. Only exists if true_signal was given. Lists the values of the strong norm error between the Landweber estimator and the true signal up to the current Landweber iteration.

    .. math::
        E[\\Vert \hat{f}_{m} - f \\Vert^2] = B^{2}_{m} + V_m

    *weak_bias2*: ``array``. Only exists if true_signal was given. Lists the values of the weak squared bias up to the current Landweber iteration.

    .. math::
        B^{2}_{m,A} = \\Vert A(I-A^{\\top}A)(f-\hat{f}_{m-1}) \\Vert^{2}

    *weak_variance*: ``array``. Only exists if true_signal was given. Lists the values of the weak variance up to the current Landweber iteration.

    .. math::
       V_{m,A} = \\delta^2 \\mathrm{tr}((I-(I-A^{\\top}A)^{m})^{2})

    *weak_risk*: ``array``. Only exists if true_signal was given. Lists the values of the weak norm error between the Landweber estimator and the true signal up to the current Landweber iteration.

    .. math::
        E[\\Vert \hat{f}_{m} - f \\Vert_A^2] = B^{2}_{m,A} + V_{m,A}

    **Methods**

    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | iterate(``number_of_iterations=1``)                     | Performs a specified number of iterations of the Landweber algorithm.    |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | landweber_to_early_stop(``max_iter``)                   | Applies early stopping to the Landweber iterative procedure.             |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_estimate(``iteration``)                             | Returns the landweber estimator at iteration.                            |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_discrepancy_stop(``critical_value, max_iteration``) | Returns the early stopping index according to the discrepancy principle. |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_weak_balanced_oracle(``max_iteration``)             | Returns the weak balanced oracle if found up to max_iteration.           |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    | get_strong_balanced_oracle(``max_iteration``)           | Returns the strong balanced oracle if found up to max_iteration.         |
    +---------------------------------------------------------+--------------------------------------------------------------------------+
    """

    def __init__(
        self,
        design,
        response,
        learning_rate=1,
        initial_value=None,
        true_signal=None,
        true_noise_level=None,
    ):

        self.design = design
        self.design_T = np.transpose(design)
        self.response = response
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level

        # Parameters of the model
        self.sample_size = np.shape(design)[0]
        self.parameter_size = np.shape(design)[1]

        # Determine starting value for the procedure
        if initial_value is None:
            warnings.warn("No initial_value is given, using zero by default.", category=UserWarning)
            self.initial_value = np.zeros(self.parameter_size)
        else:
            if initial_value.size != self.parameter_size:
                raise ValueError("The dimension of the initial_value should match paramter size!")
            self.initial_value = initial_value

        # Estimation quantities
        self.iteration = 0
        self.landweber_estimate = self.initial_value

        # Collect coefficients:
        self.landweber_estimate_list = [self.initial_value]

        self.gram_matrix = np.transpose(self.design) @ self.design

        # Residual quantities
        self.__residual_vector = self.response - self.design @ self.initial_value
        self.residuals = np.array([np.sum(self.__residual_vector**2)])

        #  Initialilze theoretical quantities
        if (self.true_signal is not None) and (self.true_noise_level is not None):
            # initialize matrices required for computing the strong/weak bias and variance
            self.identity = sparse.dia_matrix(np.eye(self.parameter_size))

            self.illposed = False
            if scipy.sparse.issparse(self.gram_matrix):
                self.inverse_congruency_matrix = scipy.sparse.linalg.inv(self.gram_matrix)
            else:
                self.inverse_congruency_matrix = np.linalg.inv(self.gram_matrix)
                print(self.inverse_congruency_matrix)

                rank = np.linalg.matrix_rank(self.gram_matrix)

                if not (rank == self.gram_matrix.shape[0]):
                    warnings.warn(
                        "PARAMETER WARNING: The inverse problem is ill-posed, settings illposed flag. EXPERIMENTAL FEATURE",
                        category=UserWarning,
                    )

                    self.illposed = False # eventually True just bugged atm

            self.perturbation_congruency_matrix = (
                sparse.dia_matrix(np.eye(self.parameter_size)) - self.learning_rate * self.gram_matrix
            )
            self.weak_perturbation_congruency_matrix = self.design @ self.perturbation_congruency_matrix
            self.perturbation_congruency_matrix_power = self.perturbation_congruency_matrix
            
            if self.illposed:
                self.accomulated_perturbation_congruency_matrix_power = self.perturbation_congruency_matrix_power 

            # initialize strong/weak bias and variance
            self.expectation_estimator = self.initial_value
            self.strong_bias2 = np.array([np.sum((self.true_signal - self.initial_value) ** 2)])
            self.weak_bias2 = np.array([np.sum((self.design @ (self.true_signal - self.initial_value)) ** 2)])

            self.strong_variance = np.array([0])
            self.weak_variance = np.array([0])

            self.strong_risk = self.strong_bias2 + self.strong_variance
            self.weak_risk = self.weak_bias2 + self.weak_variance

            self.strong_empirical_risk = np.array([np.sum((self.initial_value - self.true_signal) ** 2)])
            self.weak_empirical_risk = np.array([np.sum((self.design @ (self.initial_value - self.true_signal)) ** 2)])

    def iterate(self, number_of_iterations):
        """Performs number_of_iterations iterations of the Landweber algorithm

        **Parameters**

        *number_of_iterations*: ``int``. The number of iterations to perform.
        """
        for _ in range(number_of_iterations):
            self.__landweber_one_iteration()

    def get_estimate(self, iteration):
        """Returns the Landweber estimate at iteration

        **Parameters**

        *iteration*: ``int``. The iteration at which the Landweber estimate is requested.

        **Returns**

        *landweber_estimate*: ``ndarray``. The Landweber estimate at iteration.
        """
        if iteration is None:
            raise ValueError("iteration is None. Potentially from querying the estimate at an oracle or stopping time that was not found until max_iteration.")

        if iteration > self.iteration:
            self.iterate(iteration - self.iteration)

        landweber_estimate = self.landweber_estimate_list[iteration]
        return landweber_estimate

    def get_discrepancy_stop(self, critical_value, max_iteration):
        """Returns early stopping index based on discrepancy principle up to max_iteration

        **Parameters**

        *critical_value*: ``float``. The critical value for the discrepancy principle. The algorithm stops when
        :math: `\\Vert Y - A \hat{f}^{(m)} \\Vert^{2} \leq \\kappa^{2},`
        where :math: `\\kappa` is the critical value.

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        **Returns**

        *early_stopping_index*: ``int``. The first iteration at which the discrepancy principle is satisfied.
        (None is returned if the stopping index is not found.)
        """
        if self.residuals[self.iteration] <= critical_value:
            # argmax takes the first instance of True in the true-false array
            early_stopping_index = np.argmax(self.residuals <= critical_value)
            return early_stopping_index

        if self.residuals[self.iteration] > critical_value:
            while self.residuals[self.iteration] > critical_value and self.iteration < max_iteration:
                self.__landweber_one_iteration()

        if self.residuals[self.iteration] <= critical_value:
            early_stopping_index = self.iteration
            return early_stopping_index
        else:
            warnings.warn("Early stopping index not found up to max_iteration. Returning None.", category=UserWarning)
            return None

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
            return weak_balanced_oracle

        if self.weak_bias2[self.iteration] > self.weak_variance[self.iteration]:
            while (
                self.weak_bias2[self.iteration] > self.weak_variance[self.iteration]
                and self.iteration <= max_iteration
            ):
                self.__landweber_one_iteration()

        if self.weak_bias2[self.iteration] <= self.weak_variance[self.iteration]:
            weak_balanced_oracle = self.iteration
            return weak_balanced_oracle
        else:
            warnings.warn(
                "Weakly balanced oracle not found up to max_iteration. Returning None.", category=UserWarning
            )
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
            return strong_balanced_oracle

        if self.strong_bias2[self.iteration] > self.strong_variance[self.iteration]:
            while (
                self.strong_bias2[self.iteration] > self.strong_variance[self.iteration]
                and self.iteration <= max_iteration
            ):
                self.__landweber_one_iteration()

        if self.strong_bias2[self.iteration] <= self.strong_variance[self.iteration]:
            strong_balanced_oracle = self.iteration
            return strong_balanced_oracle
        else:
            warnings.warn(
                "Strongly balanced oracle not found up to max_iteration. Returning None.", category=UserWarning
            )
            return None

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

        if self.illposed:
            self.accomulated_perturbation_congruency_matrix_power = self.accomulated_perturbation_congruency_matrix_power + self.perturbation_congruency_matrix_power
            
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
        if self.illposed:
            warnings.warn(
                        "PARAMETER WARNING: The inverse problem is ill-posed. Switching to longfrom variane computation.",
                        category=UserWarning,
                    )
            square_matrix = self.accomulated_perturbation_congruency_matrix_power @ self.accomulated_perturbation_congruency_matrix_power
            pretrace_temporary_matrix = square_matrix @ self.gram_matrix
            new_strong_variance = (self.true_noise_level ** 2) * (self.learning_rate**2) * pretrace_temporary_matrix.trace()
            
            self.strong_variance = np.append(self.strong_variance, new_strong_variance)
            # print(new_strong_variance)
        else: 
            # presquare_temporary_matrix = self.identity - self.perturbation_congruency_matrix_power
            pretrace_temporary_matrix = (
                # self.learning_rate ** (-1)
                self.inverse_congruency_matrix
                @ (self.identity - self.perturbation_congruency_matrix_power)
                @ (self.identity - self.perturbation_congruency_matrix_power)
            )
            new_strong_variance = self.true_noise_level**2 * pretrace_temporary_matrix.trace()
            # print(new_strong_variance)
            self.strong_variance = np.append(self.strong_variance, new_strong_variance)

    def __update_weak_variance(self):
        """Update weak variance

        The weak variance in the m-th iteration is given by
        :math: `\\sigma**2 \\mathrm{tr}((I-(I-hA^{\\top}A)^{m})^{2})`
        """
        if self.illposed:
            warnings.warn(
                        "PARAMETER WARNING: The inverse problem is ill-posed. Switching to longfrom variane computation.",
                        category=UserWarning,
                    )

            pretrace_temporary_matrix_presquare = self.design @ self.accomulated_perturbation_congruency_matrix_power @ self.design_T
            square_matrix = pretrace_temporary_matrix_presquare @ pretrace_temporary_matrix_presquare
            new_weak_variance = (self.true_noise_level ** 2) * (self.learning_rate**2) * square_matrix.trace()
            
            self.weak_variance = np.append(self.weak_variance, new_weak_variance)
        else:
            pretrace_temporary_matrix = (self.identity - self.perturbation_congruency_matrix_power) @ (
                self.identity - self.perturbation_congruency_matrix_power
            )

            new_weak_variance = self.true_noise_level**2 * pretrace_temporary_matrix.trace()

            self.weak_variance = np.append(self.weak_variance, new_weak_variance)

    def __update_strong_empirical_risk(self):
        """Update the strong empirical error"""
        strong_empirical_risk = np.sum((self.landweber_estimate - self.true_signal) ** 2)
        self.strong_empirical_risk = np.append(self.strong_empirical_risk, strong_empirical_risk)

    def __update_weak_empirical_risk(self):
        """Update the weak empirical error"""
        weak_empirical_risk = np.sum((self.design @ (self.landweber_estimate - self.true_signal)) ** 2)
        self.weak_empirical_risk = np.append(self.weak_empirical_risk, weak_empirical_risk)

    def __landweber_one_iteration(self):
        """Performs one iteration of the Landweber algorithm"""
        # Add residual_vector to the update step

        self.landweber_estimate = self.landweber_estimate + self.learning_rate * np.transpose(self.design) @ (
            self.response - self.design @ self.landweber_estimate
        )
        # Collect coefficients
        self.landweber_estimate_list.append(self.landweber_estimate)

        # Update estimation quantities
        # Add residual_vector to the update step
        self.__residual_vector = self.response - self.design @ self.landweber_estimate
        new_residuals = np.sum(self.__residual_vector**2)
        self.residuals = np.append(self.residuals, new_residuals)

        self.iteration = self.iteration + 1

        if (self.true_signal is not None) and (self.true_noise_level is not None):
            # update weak and strong bias and variance
            self.__update_strong_bias2()
            self.__update_weak_bias2()
            self.__update_strong_variance()
            self.__update_weak_variance()
            self.__update_iterative_matrices()

            # update MSE and weak MSE
            self.strong_risk = self.strong_bias2 + self.strong_variance
            self.weak_risk = self.weak_bias2 + self.weak_variance

            self.__update_strong_empirical_risk()
            self.__update_weak_empirical_risk()
