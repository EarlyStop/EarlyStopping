import numpy as np


class ConjugateGradients:
    """
    `[Source] <https://github.com/ESFIEP/EarlyStopping/edit/main/src/EarlyStopping/conjugate_gradients.py>`_
    Conjugate gradients algorithm applied to the normal equation of a linear model

    **Parameters**

    *design*: ``array``. nxp design matrix of the linear model.

    *response*: ``array``. n-dim vector of the observed data in the linear model.

    *true_signal*: ``array or None, default = None``. p-dim vector. For simulation purposes only. For simulated data the true signal can be included to compute additional quantities.

    *true_noise_level*: ``float or None, default = None``. For simulation purposes only. Corresponds to the standard deviation of normally distributed noise contributing to the response variable.

    *critical_value*: ``array or None, default = None``. Critical value for the early stopping rule.

    *starting_value*: ``array or None, default = None``. Determines the zeroth step of the iterative procedure. Defaults to the zero vector.

    *interpolation*: ``boolean, default = False``. If interpolation is set to ``True``, the early stopping iteration index can be noninteger valued.

    **Attributes**

    *sample_size*: ``int``. Sample size of the linear model.

    *parameter_size*: ``int``. Parameter size of the linear model.

    *iter*: ``int``. Current conjugate gradient iteration of the algorithm.

    *conjugate_gradient_estimate*: ``array``. Conjugate gradient estimate at the current iteration for the data given in design and response.

    *early_stopping_index*: ``int or None``. Early Stopping iteration index. Is set to ``None`` if no early stopping is performed.

    *residuals*: ``array``. Lists the sequence of the squared residuals between the observed data and the conjugate gradient estimator.

    *strong_empirical_errors*: ``array``. Only exists if true_signal was given. Lists the values of the strong empirical error between the conjugate gradient estimator and the true signal up to the current conjugate gradient iteration.

    *weak_empirical_errors*: ``array``. Only exists if true_signal was given. Lists the values of the weak empirical error between the conjugate gradient estimator and the true signal up to the current conjugate gradient iteration.

    **Methods**

    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | iterate(``number_of_iterations = 1``)                   | Performs number_of_iterations of the conjugate gradients algorithm.                               |
    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | discrepancy_stop(``max_iter``)                          | Stops the conjugate gradients algorithm based on the discrepancy principle.                       |
    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | gather_all(``max_iter``)                                | Gathers all relevant simulation data.                                                             |
    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | calculate_interpolated_residual(``index``)              | Calculates the interpolated squared residual(s) at a(n array of) noninteger index (indices).      |
    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | calculate_interpolated_strong_empirical_error(``index``)| Calculates the interpolated strong empirical error(s) at a(n array of) noninteger index (indices).|
    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | calculate_interpolated_weak_empirical_error(``index``)  | Calculates the interpolated weak empirical error(s) at a(n array of) noninteger index (indices).  |
    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | calculate_empirical_oracles(``max_iter``)               | Calculates the strong and weak empirical oracle indices and errors.                               |
    +---------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    """

    def __init__(
        self,
        design,
        response,
        critical_value=None,
        starting_value=None,
        true_signal=None,
        true_noise_level=None,
        interpolation=False,
    ):
        self.design = design
        self.response = response
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.critical_value = critical_value
        self.interpolation = interpolation

        # Parameters of the model
        self.sample_size = np.shape(design)[0]
        self.parameter_size = np.shape(design)[1]

        # Determine starting value for the procedure
        if starting_value is None:
            self.starting_value = np.zeros(self.parameter_size)
        else:
            self.starting_value = starting_value

        # Estimation quantities
        self.iter = 0
        self.conjugate_gradient_estimate = self.starting_value
        self.early_stopping_index = None

        # Residual quantities
        self.residual_vector = response - design @ self.conjugate_gradient_estimate
        self.residuals = np.array([np.sum(self.residual_vector**2)])

        if self.true_signal is not None:
            self.transformed_true_signal = self.design @ self.true_signal
            self.strong_empirical_errors = np.array(
                [np.sum((self.conjugate_gradient_estimate - self.true_signal) ** 2)]
            )
            self.weak_empirical_errors = np.array(
                [np.sum((self.design @ self.conjugate_gradient_estimate - self.transformed_true_signal) ** 2)]
            )
            self.strong_estimator_distances = np.array([None])
            self.weak_estimator_distances = np.array([None])

        # Starting values for the algorithm
        self.transposed_design = np.transpose(design)
        self.transformed_residual_vector = self.transposed_design @ self.residual_vector
        self.search_direction = self.transformed_residual_vector

        # Critical value
        if self.critical_value is None and self.true_noise_level is None:
            raise ValueError("Neither the critical value nor the true noise level is specified.")
        elif self.critical_value is None and self.true_noise_level is not None:
            self.critical_value = self.true_noise_level**2 * self.sample_size

        # Vectorized functions
        self.calculate_interpolated_residual = np.vectorize(self.calculate_interpolated_residual, excluded="self")
        self.calculate_interpolated_strong_empirical_error = np.vectorize(
            self.calculate_interpolated_strong_empirical_error, excluded="self"
        )
        self.calculate_interpolated_weak_empirical_error = np.vectorize(
            self.calculate_interpolated_weak_empirical_error, excluded="self"
        )

    def iterate(self, number_of_iterations=1):
        """Performs number_of_iterations of the conjugate gradients algorithm.

        **Parameters**

        *number_of_iterations*: ``int, default = 1``. Number of conjugate gradients iterations to be performed.
        """
        for _ in range(number_of_iterations):
            if np.sum(self.transformed_residual_vector**2) == 0:
                print(f"Transformed residual vector is zero. Algorithm terminates at iteration {self.iter}.")
                break
            self.__conjugate_gradients_one_iteration()

    def __conjugate_gradients_one_iteration(self):
        """Performs one iteration of the conjugate gradients algorithm"""
        old_transformed_residual_vector = self.transformed_residual_vector
        squared_norm_old_transformed_residual_vector = np.sum(old_transformed_residual_vector**2)
        transformed_search_direction = self.design @ self.search_direction
        learning_rate = squared_norm_old_transformed_residual_vector / np.sum(transformed_search_direction**2)
        conjugate_gradient_estimates_distance = learning_rate * self.search_direction
        transformed_conjugate_gradient_estimates_distance = learning_rate * transformed_search_direction
        self.conjugate_gradient_estimate = self.conjugate_gradient_estimate + conjugate_gradient_estimates_distance
        self.residual_vector = self.residual_vector - transformed_conjugate_gradient_estimates_distance
        self.transformed_residual_vector = self.transposed_design @ self.residual_vector
        transformed_residual_ratio = (
            np.sum(self.transformed_residual_vector**2) / squared_norm_old_transformed_residual_vector
        )
        self.search_direction = self.transformed_residual_vector + transformed_residual_ratio * self.search_direction
        self.residuals = np.append(self.residuals, np.sum(self.residual_vector**2))

        self.iter = self.iter + 1

        if self.true_signal is not None:
            self.strong_empirical_errors = np.append(
                self.strong_empirical_errors, np.sum((self.conjugate_gradient_estimate - self.true_signal) ** 2)
            )
            self.weak_empirical_errors = np.append(
                self.weak_empirical_errors,
                np.sum((self.design @ self.conjugate_gradient_estimate - self.transformed_true_signal) ** 2),
            )
            self.strong_estimator_distances = np.append(
                self.strong_estimator_distances, np.sum((conjugate_gradient_estimates_distance) ** 2)
            )
            self.weak_estimator_distances = np.append(
                self.weak_estimator_distances, np.sum((transformed_conjugate_gradient_estimates_distance) ** 2)
            )

    def discrepancy_stop(self, max_iter):
        """Early stopping for the conjugate gradient procedure based on the discrepancy principle. Procedure is stopped when the squared residuals go below critical_value or iteration max_iter is reached.

        **Parameters**

        *max_iter*: ``int``. The maximum number of iterations to be performed.
        """
        while self.residuals[self.iter] > self.critical_value and self.iter < max_iter:
            if self.interpolation is True:
                old_conjugate_gradient_estimate = self.conjugate_gradient_estimate
            self.__conjugate_gradients_one_iteration()
        if (self.interpolation is True) and (self.iter > 0) and (self.residuals[self.iter] <= self.critical_value):
            alpha = 1 - np.sqrt(
                1
                - (self.residuals[self.iter - 1] - self.critical_value)
                / (self.residuals[self.iter - 1] - self.residuals[self.iter])
            )
            self.early_stopping_index = self.iter - 1 + alpha
            self.conjugate_gradient_estimate = (
                1 - alpha
            ) * old_conjugate_gradient_estimate + alpha * self.conjugate_gradient_estimate
        else:
            self.early_stopping_index = self.iter

    def gather_all(self, max_iter):
        """Gathers all relevant simulation data (runs the algorithm till max_iter) but tracks the early stopping index and the associated conjugate gradient estimate.

        **Parameters**

        *max_iter*: ``int``. The maximum number of iterations to be performed.
        """
        self.discrepancy_stop(max_iter)
        conjugate_gradient_estimate = self.conjugate_gradient_estimate
        if max_iter > int(np.ceil(self.early_stopping_index)):
            self.iterate(max_iter - int(np.ceil(self.early_stopping_index)))
            self.conjugate_gradient_estimate = conjugate_gradient_estimate

    def calculate_interpolated_residual(self, index):
        """Calculates the interpolated squared residual at a possibly noninteger iteration index. The function is vectorized such that arrays of indices can be inserted.

        **Parameters**

        *index*: ``array or float``. Index or array of indices where the interpolated squared residual(s) should be calculated.
        """
        index_ceil = int(np.ceil(index))
        index_floor = int(np.floor(index))
        alpha = index - index_floor
        interpolated_residual = (1 - alpha) ** 2 * self.residuals[index_floor] + (
            1 - (1 - alpha) ** 2
        ) * self.residuals[index_ceil]
        return interpolated_residual

    def calculate_interpolated_strong_empirical_error(self, index):
        """Calculates the interpolated strong empirical error at a possibly noninteger iteration index. The function is vectorized such that arrays of indices can be inserted.

        **Parameters**

        *index*: ``array or float``. Index or array of indices where the interpolated error(s) should be calculated.
        """
        if index == 0:
            interpolated_strong_empirical_error = self.strong_empirical_errors[0]
        else:
            index_ceil = int(np.ceil(index))
            index_floor = int(np.floor(index))
            alpha = index - index_floor
            interpolated_strong_empirical_error = (
                (1 - alpha) * self.strong_empirical_errors[index_floor]
                + alpha * self.strong_empirical_errors[index_ceil]
                - (1 - alpha) * alpha * self.strong_estimator_distances[index_ceil]
            )
        return interpolated_strong_empirical_error

    def calculate_interpolated_weak_empirical_error(self, index):
        """Calculates the interpolated weak empirical error at a possibly noninteger iteration index. The function is vectorized such that arrays of indices can be inserted.

        **Parameters**

        *index*: ``array or float``. Index or array of indices where the interpolated error(s) should be calculated.
        """
        if index == 0:
            interpolated_weak_empirical_error = self.weak_empirical_errors[0]
        else:
            index_ceil = int(np.ceil(index))
            index_floor = int(np.floor(index))
            alpha = index - index_floor
            interpolated_weak_empirical_error = (
                (1 - alpha) * self.weak_empirical_errors[index_floor]
                + alpha * self.weak_empirical_errors[index_ceil]
                - (1 - alpha) * alpha * self.weak_estimator_distances[index_ceil]
            )
        return interpolated_weak_empirical_error

    def calculate_empirical_oracles(self, max_iter):
        """Calculates the strong and weak empirical oracles. Returns a vector, where the first (third) entry is the strong (weak) empirical oracle and the second (fourth) entry
        is the corresponding strong (weak) empirical error.

        **Parameters**

        *max_iter*: ``int``. The maximum number of iterations to be performed.
        """
        if self.interpolation is True:
            empirical_errors_list = [self.strong_empirical_errors, self.weak_empirical_errors]
            estimator_distances_list = [
                self.strong_estimator_distances,
                self.weak_estimator_distances,
            ]
            empirical_oracles = []
            for error_type in np.arange(2):
                optimal_index_list = []
                empirical_errors = empirical_errors_list[error_type]
                estimator_distances = estimator_distances_list[error_type]
                for index in np.arange(max_iter):
                    if estimator_distances[index + 1] > 10 ** (-7):
                        alpha = (
                            empirical_errors[index] - empirical_errors[index + 1] + estimator_distances[index + 1]
                        ) / (2 * estimator_distances[index + 1])
                        if alpha < 0:
                            optimal_index_list = np.append(optimal_index_list, index)
                        elif alpha > 1:
                            optimal_index_list = np.append(optimal_index_list, index + 1)
                        else:
                            optimal_index_list = np.append(optimal_index_list, index + alpha)
                    else:
                        optimal_index_list = np.append(optimal_index_list, index)
                if error_type == 0:
                    empirical_errors_at_optimal_index_list = self.calculate_interpolated_strong_empirical_error(
                        optimal_index_list
                    )
                else:
                    empirical_errors_at_optimal_index_list = self.calculate_interpolated_weak_empirical_error(
                        optimal_index_list
                    )
                empirical_oracle_error = np.min(empirical_errors_at_optimal_index_list)
                empirical_oracle_index = optimal_index_list[np.argmin(empirical_errors_at_optimal_index_list)]
                empirical_oracles = np.append(empirical_oracles, [empirical_oracle_index, empirical_oracle_error])
        else:
            empirical_oracles = [
                np.argmin(self.strong_empirical_errors),
                np.min(self.strong_empirical_errors),
                np.argmin(self.weak_empirical_errors),
                np.min(self.weak_empirical_errors),
            ]
        return empirical_oracles
