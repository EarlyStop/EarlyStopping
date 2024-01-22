import numpy as np


class ConjugateGradients:
    """A class to perform estimation using the conjugate gradients iterative method.

    Parameters
    ----------
    design_matrix: array
        nxp design matrix of the linear model.

    response_variable: array
        n-dim vector of the observed data in the linear model.

    critical_value: array, default = None
        Critical value for the early stopping rule.

    starting_value: array, default = None
        Determines the zeroth step of the iterative procedure. (Defaults to zero).

    true_signal: array, default = None
        p-dim vector
        For simulation purposes only. For simulated data the true signal can be
        included to compute additional quantities.

    true_noise_level: float, default = None
        For simulation purposes only. Corresponds to the standard deviation
        of normally distributed noise contributing to the response variable.

    interpolation: boolean, default = False
        If interpolation is set to True, the early stopping iteration index can be
        noninteger valued. (Defaults to False.)

    Attributes
    ----------
    sample_size: int
        Sample size of the linear model

    para_size: int
        Parameter size of the linear model

    iter: int
        Current conjugate gradient iteration of the algorithm

    early_stopping_index: int
        Early Stopping iteration index (Is set to None if no early stopping is performed)

    conjugate_gradient_estimate: array
        Conjugate gradient estimate at the current iteration for the data given in
        design_matrix

    residuals: array
        Lists the sequence of the squared residuals between the observed data and
        the conjugate gradient estimator.

    strong_empirical_error: array
        Only exists if true_signal was given. Lists the values of the strong empirical error
        between the conjugate gradient estimator and the true signal up to the
        current conjugate gradient iteration.

    weak_empirical_error: array
        Only exists if true_signal was given. Lists the values of the weak empirical error
        between the conjugate gradient estimator and the true signal up to the
        current conjugate gradient iteration.

    Methods
    -------
    conjugate_gradients(iterations=1)
        Performs a specified number of iterations of the conjugate gradients algorithm.

    conjugate_gradients_to_early_stop(crit, max_iter)
        Applies early stopping to the conjugate gradients iterative procedure.
    """

    def __init__(
        self,
        design_matrix,
        response_variable,
        critical_value=None,
        starting_value=None,
        true_signal=None,
        true_noise_level=None,
        interpolation=False,
    ):
        self.design_matrix = design_matrix
        self.response_variable = response_variable
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.critical_value = critical_value
        self.interpolation = interpolation

        # Parameters of the model
        self.sample_size = np.shape(design_matrix)[0]
        self.para_size = np.shape(design_matrix)[1]

        # Determine starting value for the procedure
        if starting_value is None:
            self.starting_value = np.zeros(self.para_size)
        else:
            self.starting_value = starting_value

        # Estimation quantities
        self.iter = 0
        self.conjugate_gradient_estimate = self.starting_value
        self.early_stopping_index = None

        # Residual quantities
        self.residual_vector = response_variable - design_matrix @ self.conjugate_gradient_estimate
        self.residuals = np.array([np.sum(self.residual_vector**2)])

        self.transposed_design_matrix = np.transpose(design_matrix)
        self.transformed_residual_vector = self.transposed_design_matrix @ self.residual_vector
        self.search_direction = self.transformed_residual_vector

        if self.critical_value is None and self.true_noise_level is None:
            raise ValueError("True noise level is not specified.")
        elif self.critical_value is None and self.true_noise_level is not None:
            # if the true noise level is given, it does not need to be estimated
            self.critical_value = self.true_noise_level**2 * self.sample_size

        if self.true_signal is not None:
            self.transformed_true_signal = self.design_matrix @ self.true_signal
            self.strong_empirical_error = np.array(
                [np.sum((self.conjugate_gradient_estimate - self.true_signal) ** 2)]
            )
            self.weak_empirical_error = np.array(
                [np.sum((self.design_matrix @ self.conjugate_gradient_estimate - self.transformed_true_signal) ** 2)]
            )
            if interpolation:
                self.strong_empirical_error_inner_products = np.array([None])
                self.weak_empirical_error_inner_products = np.array([None])

        # Vectorize functions
        self.calculate_interpolated_residual = np.vectorize(self.calculate_interpolated_residual, excluded="self")
        self.calculate_interpolated_strong_empirical_error = np.vectorize(
            self.calculate_interpolated_strong_empirical_error, excluded="self"
        )
        self.calculate_interpolated_weak_empirical_error = np.vectorize(
            self.calculate_interpolated_weak_empirical_error, excluded="self"
        )

    def conjugate_gradients(self, iterations=1):
        """Performs iterations of the conjugate gradients algorithm

        Parameters
        ----------
        iterations: int, default = 1
            The number of iterations to perform.
        """
        for _ in range(iterations):
            if np.sum(self.transformed_residual_vector**2) == 0:
                print("Transformed residual vector is zero. Algorithm terminates.")
                break
            self.__conjugate_gradients_one_iteration()

    def __conjugate_gradients_one_iteration(self):
        """Performs one iteration of the conjugate gradients algorithm"""
        old_transformed_residual_vector = self.transformed_residual_vector
        squared_norm_old_transformed_residual_vector = np.sum(old_transformed_residual_vector**2)
        transformed_search_direction = self.design_matrix @ self.search_direction
        learning_rate = squared_norm_old_transformed_residual_vector / np.sum(transformed_search_direction**2)
        if self.true_signal is not None and self.interpolation:
            strong_empirical_error_inner_product = (
                self.strong_empirical_error[-1]
                + learning_rate
                * np.transpose(self.conjugate_gradient_estimate - self.true_signal)
                @ self.search_direction
            )
            self.strong_empirical_error_inner_products = np.append(
                self.strong_empirical_error_inner_products, strong_empirical_error_inner_product
            )
            weak_empirical_error_inner_product = self.weak_empirical_error[-1] + learning_rate * np.transpose(
                self.design_matrix @ (self.conjugate_gradient_estimate - self.true_signal)
            ) @ (self.design_matrix @ self.search_direction)
            self.weak_empirical_error_inner_products = np.append(
                self.weak_empirical_error_inner_products, weak_empirical_error_inner_product
            )

        self.conjugate_gradient_estimate = self.conjugate_gradient_estimate + learning_rate * self.search_direction
        self.residual_vector = self.residual_vector - learning_rate * transformed_search_direction
        self.transformed_residual_vector = self.transposed_design_matrix @ self.residual_vector
        transformed_residual_ratio = (
            np.sum(self.transformed_residual_vector**2) / squared_norm_old_transformed_residual_vector
        )
        self.search_direction = self.transformed_residual_vector + transformed_residual_ratio * self.search_direction
        self.residuals = np.append(self.residuals, np.sum(self.residual_vector**2))

        self.iter = self.iter + 1

        if self.true_signal is not None:
            self.strong_empirical_error = np.append(
                self.strong_empirical_error, np.sum((self.conjugate_gradient_estimate - self.true_signal) ** 2)
            )
            self.weak_empirical_error = np.append(
                self.weak_empirical_error,
                np.sum((self.design_matrix @ self.conjugate_gradient_estimate - self.transformed_true_signal) ** 2),
            )

    def conjugate_gradients_to_early_stop(self, max_iter):
        """Early stopping for the conjugate gradient procedure

            Procedure is stopped when the residuals go below critical_value or iteration
            max_iter is reached.

        Parameters
        ----------
        max_iter: int
            The maximum number of iterations to perform.
        """
        while self.residuals[self.iter] > self.critical_value and self.iter < max_iter:
            if self.interpolation is True:
                old_conjugate_gradient_estimate = self.conjugate_gradient_estimate
            self.__conjugate_gradients_one_iteration()
        if self.interpolation is True:
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

    def conjugate_gradients_gather_all(self, max_iter):
        """Gathers all relevant simulation data (runs the algorithm till max_iter) but tracks the early stopping index and the associated conjugate gradient estimate."""
        self.conjugate_gradients_to_early_stop(max_iter)
        conjugate_gradient_estimate = self.conjugate_gradient_estimate
        if max_iter > int(np.ceil(self.early_stopping_index)):
            self.conjugate_gradients(max_iter - int(np.ceil(self.early_stopping_index)))
            self.conjugate_gradient_estimate = conjugate_gradient_estimate

    def calculate_interpolated_residual(self, index):
        """Calculates the interpolated squared residual at a possibly noninteger iteration index. The function is vectorized such that arrays of indices can be inserted."""
        index_ceil = int(np.ceil(index))
        index_floor = int(np.floor(index))
        alpha = index - index_floor
        interpolated_residual = (1 - alpha) ** 2 * self.residuals[index_floor] + (
            1 - (1 - alpha) ** 2
        ) * self.residuals[index_ceil]
        return interpolated_residual

    def calculate_interpolated_strong_empirical_error(self, index):
        """Calculates the interpolated strong empirical error at a possibly noninteger iteration index. The function is vectorized such that arrays of indices can be inserted."""
        index_ceil = int(np.ceil(index))
        index_floor = int(np.floor(index))
        alpha = index - index_floor
        if index == 0:
            interpolated_strong_empirical_error = self.strong_empirical_error[0]
        else:
            interpolated_strong_empirical_error = (
                (1 - alpha) ** 2 * self.strong_empirical_error[index_floor]
                + alpha**2 * self.strong_empirical_error[index_ceil]
                + 2 * (1 - alpha) * alpha * self.strong_empirical_error_inner_products[index_ceil]
            )
        return interpolated_strong_empirical_error

    def calculate_interpolated_weak_empirical_error(self, index):
        """Calculates the interpolated weak empirical error at a possibly noninteger iteration index. The function is vectorized such that arrays of indices can be inserted."""
        index_ceil = int(np.ceil(index))
        index_floor = int(np.floor(index))
        alpha = index - index_floor
        if index == 0:
            interpolated_weak_empirical_error = self.weak_empirical_error[0]
        else:
            interpolated_weak_empirical_error = (
                (1 - alpha) ** 2 * self.weak_empirical_error[index_floor]
                + alpha**2 * self.weak_empirical_error[index_ceil]
                + 2 * (1 - alpha) * alpha * self.weak_empirical_error_inner_products[index_ceil]
            )
        return interpolated_weak_empirical_error
