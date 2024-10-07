import numpy as np
import warnings


class ConjugateGradients:
    """
     `[Source] <https://github.com/ESFIEP/EarlyStopping/edit/main/src/EarlyStopping/conjugate_gradients.py>`_ A class to perform estimation using the conjugate gradients algorithm for the normal equation.

    **Description**

    Consider the *linear model*

    .. math::
        Y = Af + \\delta Z,

    where :math:`Z` is a :math:`n`-dimensional standard normal distribution.
    The conjugate gradient estimate :math:`\\hat{f}^{(m)}` at the integer iteration index :math:`m` is iteratively calculated by the *conjugate gradients for the normal equation* algorithm with initial value :math:`\\hat{f}_0`,
    see `Björck (1996, Algorithm 7.4.1) <https://doi.org/10.1137/1.9781611971484>`_.

    **Parameters**

    *design*: ``ndarray``. nxp design matrix of the linear model. ( :math:`A \\in \\mathbb{R}^{n \\times p}` )

    *response*: ``ndarray``. n-dim vector of the observed data in the linear model. ( :math:`Y \in \mathbb{R}^{n}` )

    *initial_value*: ``array, default = None``. Determines the zeroth step of the iterative procedure. Default is zero. ( :math:`\\hat{f}_0` )

    *true_signal*: ``ndarray, default = None``. p-dim vector for simulation purposes only. For simulated data the true signal can be included to compute additional quantities alongside the iterative procedure. ( :math:`f \\in \\mathbb{R}^{p}` )

    *true_noise_level*: ``float, default = None``. For simulation purposes only. Corresponds to the standard deviation of normally distributed noise contributing to the response variable. ( :math:`\\delta \\geq 0` )

    *computation_threshold*: ``float, default = 10 ** (-8)``. Threshold used to terminate the conjugate gradients algorithm.

    **Attributes**

    *sample_size*: ``int``. Sample size of the linear model. ( :math:`n \\in \\mathbb{N}` )

    *parameter_size*: ``int``. Parameter size of the linear model. ( :math:`p \\in \\mathbb{N}` )

    *iteration*: ``int``. Current conjugate gradient iteration of the algorithm. ( :math:`m \\in \\mathbb{N}` )

    *conjugate_gradient_estimate*: ``ndarray``. Conjugate gradient estimate at the current iteration for the data given in design and response. ( :math:`\\hat{f}^{(m)}` )

    *conjugate_gradient_estimate_list*: ``list``. List containing the conjugate gradient estimates at integer iteration indices up to the current conjugate gradient iteration.

    *residuals*: ``ndarray``. Lists the sequence of the squared residuals between the observed data and the conjugate gradient estimator.

    *strong_empirical_risk*: ``ndarray``. Only exists if true_signal was given. Lists the values of the strong empirical error between the conjugate gradient estimator and the true signal up to the current conjugate gradient iteration.

    *weak_empirical_risk*: ``ndarray``. Only exists if true_signal was given. Lists the values of the weak empirical error between the conjugate gradient estimator and the true signal up to the current conjugate gradient iteration.

    **Methods**

    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | iterate(``number_of_iterations = 1``)                                                 | Performs a specified number of iterations of the conjugate gradients algorithm.             |
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | get_estimate(``iteration``)                                                           | Returns the conjugate gradient estimator at iteration.                                      |
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | get_discrepancy_stop(``critical_value``, ``max_iteration``, ``interpolation = False``)| Returns the early stopping index according to the discrepancy principle with emergency stop.|
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | get_residual(``iteration``)                                                           | Returns the squared residual at iteration.                                                  |
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | get_strong_empirical_risk(``iteration``)                                              | Returns the strong empirical error at iteration.                                            |
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | get_weak_empirical_risk(``iteration``)                                                | Returns the weak empirical error at iteration.                                              |
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | get_strong_empirical_oracle(``max_iteration``, ``interpolation = False``)             | Returns the strong empirical oracle up to max_iteration.                                    |
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    | get_weak_empirical_oracle(``max_iteration``, ``interpolation = False``)               | Returns the weak empirical oracle up to max_iteration.                                      |
    +---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
    """

    def __init__(
        self,
        design,
        response,
        initial_value=None,
        true_signal=None,
        true_noise_level=None,
        computation_threshold=10 ** (-8),
    ):
        self.design = design
        self.response = response
        self.initial_value = initial_value
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level

        self.computation_threshold = computation_threshold

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
        self.conjugate_gradient_estimate = self.initial_value

        # Collect coefficients at integer iteration indices
        self.conjugate_gradient_estimate_list = [self.initial_value]

        # Residual quantities
        self.__residual_vector = self.response - self.design @ self.initial_value
        self.residuals = np.array([np.sum(self.__residual_vector**2)])
        self.__transformed_residuals = np.array([np.sum((np.transpose(self.design) @ self.__residual_vector) ** 2)])

        # Initialize theoretical quantities
        if self.true_signal is not None:
            self.__transformed_true_signal = self.design @ self.true_signal
            self.strong_empirical_risk = np.array([np.sum((self.initial_value - self.true_signal) ** 2)])
            self.weak_empirical_risk = np.array(
                [np.sum((self.design @ self.initial_value - self.__transformed_true_signal) ** 2)]
            )
            self.__strong_estimator_distances = np.array([0])
            self.__weak_estimator_distances = np.array([0])

        # Starting values for the algorithm
        self.__transposed_design = np.transpose(design)
        self.__transformed_residual_vector = self.__transposed_design @ self.__residual_vector
        self.__search_direction = self.__transformed_residual_vector

    def iterate(self, number_of_iterations=1):
        """Performs number_of_iterations iterations of the conjugate gradients algorithm

        **Parameters**

        *number_of_iterations*: ``int``. The number of iterations to perform.
        """
        for _ in range(number_of_iterations):
            if self.__transformed_residuals[self.iteration] <= self.computation_threshold:
                warnings.warn(
                    f"Algorithm terminates at iteration {self.iteration}: norm of transformed residual vector ({self.__transformed_residuals[self.iteration]}) <= computation_threshold ({self.computation_threshold}).",
                    category=UserWarning,
                )
                break
            self.__conjugate_gradients_one_iteration()

    def get_estimate(self, iteration):
        """Returns the conjugate gradient estimate at a possibly noninteger iteration

        **Parameters**

        *iteration*: ``int or float``. The (possibly noninteger) iteration at which the conjugate gradient estimate is requested.

        **Returns**

        *conjugate_gradient_estimate*: ``ndarray``. The conjugate gradient estimate at iteration.
        """

        iteration_ceil = np.ceil(iteration).astype("int")

        if iteration_ceil > self.iteration:
            self.iterate(iteration_ceil - self.iteration)

        if iteration_ceil > self.iteration:
            raise ValueError(
                "Algorithm terminated due to computation_threshold before (ceiling of) requested iteration."
            )

        if iteration % 1 == 0:
            conjugate_gradient_estimate = self.conjugate_gradient_estimate_list[int(iteration)]
        else:
            iteration_floor = np.floor(iteration).astype("int")
            alpha = iteration - iteration_floor
            conjugate_gradient_estimate = (1 - alpha) * self.conjugate_gradient_estimate_list[
                iteration_floor
            ] + alpha * self.conjugate_gradient_estimate_list[iteration_ceil]

        return conjugate_gradient_estimate

    def get_discrepancy_stop(self, critical_value, max_iteration, interpolation=False):
        """Returns early stopping index based on discrepancy principle up to max_iteration

        **Parameters**

        *critical_value*: ``float``. The critical value for the discrepancy principle. The algorithm stops when
        :math:`\\Vert Y - A \\hat{f}^{(m)} \\Vert^{2} \\leq \\kappa^{2},`
        where :math:`\\kappa` is the critical value.

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        *interpolation*: ``boolean, default = False``. If interpolation is set to ``True``, the early stopping index can be noninteger valued.

        **Returns**

        *early_stopping_index*: ``int or float``. The first iteration at which the discrepancy principle is satisfied or
        :math:`\\Vert A^{\\top}(Y - A \\hat{f}^{(m)}) \\Vert^{2} \\leq C,` where :math:`C` is the computation_threshold
        for the emergency stop.
        (``None`` is returned if the stopping index is not found.)
        """

        # Discrepancy stop on existing estimators
        if np.any(self.residuals <= critical_value):
            iteration = np.argmax(self.residuals <= critical_value)
            if interpolation is True and iteration > 0:
                alpha = 1 - np.sqrt(
                    1
                    - (self.residuals[iteration - 1] - critical_value)
                    / (self.residuals[iteration - 1] - self.residuals[iteration])
                )
                early_stopping_index = iteration - 1 + alpha
            else:
                early_stopping_index = iteration
            return early_stopping_index
        # Further iteration if necessary and possible
        elif self.__transformed_residuals[self.iteration] > self.computation_threshold:
            while (
                self.residuals[self.iteration] > critical_value
                and self.__transformed_residuals[self.iteration] > self.computation_threshold
                and self.iteration < max_iteration
            ):
                self.__conjugate_gradients_one_iteration()

        # Emergency stop
        if self.__transformed_residuals[self.iteration] <= self.computation_threshold:
            early_stopping_index = self.iteration
            return early_stopping_index

        # Discrepancy stop
        if self.residuals[self.iteration] <= critical_value:
            if interpolation is True and self.iteration > 0:
                alpha = 1 - np.sqrt(
                    1
                    - (self.residuals[self.iteration - 1] - critical_value)
                    / (self.residuals[self.iteration - 1] - self.residuals[self.iteration])
                )
                early_stopping_index = self.iteration - 1 + alpha
            else:
                early_stopping_index = self.iteration
            return early_stopping_index
        else:
            warnings.warn("Early stopping index not found up to max_iteration. Returning None.", category=UserWarning)
            return None

    def get_residual(self, iteration):
        """Returns the squared residual at a possibly noninteger iteration index

        **Parameters**

        *iteration*: ``int or float``. Iteration index where the squared residual should be calculated.

        **Returns**

        *residual*: ``float``. The squared residual at the requested iteration index.
        """

        iteration_ceil = np.ceil(iteration).astype("int")

        if iteration_ceil > self.iteration:
            self.iterate(iteration_ceil - self.iteration)

        if iteration_ceil > self.iteration:
            raise ValueError(
                "Algorithm terminated due to computation_threshold before (ceiling of) requested iteration."
            )

        if iteration % 1 == 0:
            residual = self.residuals[int(iteration)]
        else:
            iteration_floor = np.floor(iteration).astype("int")
            alpha = iteration - iteration_floor
            residual = (1 - alpha) ** 2 * self.residuals[iteration_floor] + (1 - (1 - alpha) ** 2) * self.residuals[
                iteration_ceil
            ]
        return residual

    def get_strong_empirical_risk(self, iteration):
        """Returns the strong empirical error at a possibly noninteger iteration index

        **Parameters**

        *iteration*: ``int or float``. Iteration index where the error should be calculated.

        **Returns**

        *strong_empirical_risk*: ``float``. The strong empirical error at the requested iteration index.
        """

        if self.true_signal is None:
            raise ValueError("No true signal given.")

        conjugate_gradient_estimate = self.get_estimate(iteration)

        strong_empirical_risk = np.sum((conjugate_gradient_estimate - self.true_signal) ** 2)
        return strong_empirical_risk

    def get_weak_empirical_risk(self, iteration):
        """Returns the weak empirical error at a possibly noninteger iteration index

        **Parameters**

        *iteration*: ``int or float``. Iteration index where the error should be calculated.

        **Returns**

        *weak_empirical_risk*: ``float``. The weak empirical error at the requested iteration index.
        """

        if self.true_signal is None:
            raise ValueError("No true signal given.")

        conjugate_gradient_estimate = self.get_estimate(iteration)

        weak_empirical_risk = np.sum((self.design @ conjugate_gradient_estimate - self.__transformed_true_signal) ** 2)
        return weak_empirical_risk

    def get_strong_empirical_oracle(self, max_iteration, interpolation=False):
        """Returns the strong empirical oracle up to max_iteration

        **Parameters**

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        *interpolation*: ``boolean, default = False``. If interpolation is set to ``True``, the strong empirical oracle can be noninteger valued.

        **Returns**

        *strong_empirical_oracle*: ``int or float``. The iteration index at which the strong empirical error is minimal along the iteration path up to max_iteration. If not unique, the smallest one is returned.
        """

        if self.true_signal is None:
            raise ValueError("No true signal given.")

        if max_iteration > self.iteration:
            self.iterate(max_iteration - self.iteration)

        if max_iteration > self.iteration:
            warnings.warn(
                "Algorithm terminated due to computation_threshold before max_iteration. max_iteration is set to terminal iteration index.",
                category=UserWarning,
            )
            max_iteration = self.iteration

        if interpolation is True:
            optimal_index_list = []
            empirical_risk_at_optimal_index_list = []
            empirical_risk = self.strong_empirical_risk
            estimator_distances = self.__strong_estimator_distances
            for index in np.arange(max_iteration):
                if np.sqrt(estimator_distances[index + 1]) <= self.computation_threshold:
                    alpha = 0
                else:
                    alpha = (empirical_risk[index] - empirical_risk[index + 1] + estimator_distances[index + 1]) / (
                        2 * estimator_distances[index + 1]
                    )
                if alpha < 0:
                    optimal_index_candidate = index
                elif alpha > 1:
                    optimal_index_candidate = index + 1
                else:
                    optimal_index_candidate = index + alpha
                optimal_index_list = np.append(optimal_index_list, optimal_index_candidate)
                empirical_risk_at_optimal_index_list = np.append(
                    empirical_risk_at_optimal_index_list,
                    self.get_strong_empirical_risk(optimal_index_candidate),
                )
            strong_empirical_oracle = optimal_index_list[np.argmin(empirical_risk_at_optimal_index_list)]
        else:
            strong_empirical_oracle = np.argmin(self.strong_empirical_risk)

        return strong_empirical_oracle

    def get_weak_empirical_oracle(self, max_iteration, interpolation=False):
        """Returns the weak empirical oracle up to max_iteration

        **Parameters**

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        *interpolation*: ``boolean, default = False``. If interpolation is set to ``True``, the weak empirical oracle can be noninteger valued.

        **Returns**

        *weak_empirical_oracle*: ``int or float``. The iteration index at which the weak empirical error is minimal along the iteration path up to max_iteration. If not unique, the smallest one is returned.
        """

        if self.true_signal is None:
            raise ValueError("No true signal given.")

        if max_iteration > self.iteration:
            self.iterate(max_iteration - self.iteration)

        if max_iteration > self.iteration:
            warnings.warn(
                "Algorithm terminated due to computation_threshold before max_iteration. max_iteration is set to terminal iteration index.",
                category=UserWarning,
            )
            max_iteration = self.iteration

        if interpolation is True:
            optimal_index_list = []
            empirical_risk_at_optimal_index_list = []
            empirical_risk = self.weak_empirical_risk
            estimator_distances = self.__weak_estimator_distances
            for index in np.arange(max_iteration):
                if np.sqrt(estimator_distances[index + 1]) <= self.computation_threshold:
                    alpha = 0
                else:
                    alpha = (empirical_risk[index] - empirical_risk[index + 1] + estimator_distances[index + 1]) / (
                        2 * estimator_distances[index + 1]
                    )
                if alpha < 0:
                    optimal_index_candidate = index
                elif alpha > 1:
                    optimal_index_candidate = index + 1
                else:
                    optimal_index_candidate = index + alpha
                optimal_index_list = np.append(optimal_index_list, optimal_index_candidate)
                empirical_risk_at_optimal_index_list = np.append(
                    empirical_risk_at_optimal_index_list,
                    self.get_weak_empirical_risk(optimal_index_candidate),
                )
            weak_empirical_oracle = optimal_index_list[np.argmin(empirical_risk_at_optimal_index_list)]
        else:
            weak_empirical_oracle = np.argmin(self.weak_empirical_risk)
        return weak_empirical_oracle

    def __update_strong_empirical_risk(self):
        """Update the strong empirical error"""
        strong_empirical_risk = np.sum((self.conjugate_gradient_estimate - self.true_signal) ** 2)
        self.strong_empirical_risk = np.append(self.strong_empirical_risk, strong_empirical_risk)

    def __update_weak_empirical_risk(self):
        """Update the weak empirical error"""
        weak_empirical_risk = np.sum(
            (self.design @ self.conjugate_gradient_estimate - self.__transformed_true_signal) ** 2
        )
        self.weak_empirical_risk = np.append(self.weak_empirical_risk, weak_empirical_risk)

    def __conjugate_gradients_one_iteration(self):
        """Performs one iteration of the conjugate gradients algorithm"""
        old_transformed_residual_vector = self.__transformed_residual_vector
        squared_norm_old_transformed_residual_vector = np.sum(old_transformed_residual_vector**2)
        transformed_search_direction = self.design @ self.__search_direction
        learning_rate = squared_norm_old_transformed_residual_vector / np.sum(transformed_search_direction**2)
        conjugate_gradient_estimates_distance = learning_rate * self.__search_direction
        transformed_conjugate_gradient_estimates_distance = learning_rate * transformed_search_direction
        self.conjugate_gradient_estimate = self.conjugate_gradient_estimate + conjugate_gradient_estimates_distance
        self.conjugate_gradient_estimate_list.append(self.conjugate_gradient_estimate)
        self.__residual_vector = self.__residual_vector - transformed_conjugate_gradient_estimates_distance
        self.__transformed_residual_vector = self.__transposed_design @ self.__residual_vector
        transformed_residual_ratio = (
            np.sum(self.__transformed_residual_vector**2) / squared_norm_old_transformed_residual_vector
        )
        self.__search_direction = (
            self.__transformed_residual_vector + transformed_residual_ratio * self.__search_direction
        )
        self.residuals = np.append(self.residuals, np.sum(self.__residual_vector**2))
        self.__transformed_residuals = np.append(
            self.__transformed_residuals, np.sum(self.__transformed_residual_vector**2)
        )

        self.iteration = self.iteration + 1

        if self.true_signal is not None:
            self.__update_strong_empirical_risk()
            self.__update_weak_empirical_risk()
            self.__strong_estimator_distances = np.append(
                self.__strong_estimator_distances,
                np.sum(conjugate_gradient_estimates_distance**2),
            )
            self.__weak_estimator_distances = np.append(
                self.__weak_estimator_distances,
                np.sum(transformed_conjugate_gradient_estimates_distance**2),
            )
