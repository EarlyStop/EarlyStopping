import numpy as np
from sklearn import linear_model
import warnings


class L2_boost:
    """
    `[Source] <https://github.com/ESFIEP/EarlyStopping/blob/main/src/EarlyStopping/L2_boost.py>`_
    L2-boosting algorithm for high dimensional linear models.

    **Parameters**

    *design*: ``array``. nxp-Design matrix of the linear model.

    *response*: ``array``. n-dim vector of the observed data in the linear model.

    *true_signal*: ``array or None, default = None``. For simulation purposes only. For simulated data the true signal can be included to compute theoretical quantities such as the bias and the risk alongside the boosting procedure.

    **Attributes**

    *sample_size:* ``int``. Sample size of the linear model.

    *parameter_size*: ``int``. Parameter size of the linear model.

    *iteration*: ``int``. Current boosting iteration of the algorithm.

    *boost_estimate*: ``array``. Boosting estimate at the current iteration for the data given in design.

    *residuals*: ``array``. Lists the sequence of the residual mean of squares betwean the data and the boosting estimator.

    *bias2*: ``array``. Only exists if true_signal was given. Lists the values of the squared bias up to current boosting iteration.

    *stochastic_error*: ``array``. Only exists if true_signal was given. Lists the values of a stochastic error term up to current boosting iteration.

    *risk*: ``array``. Only exists if true_signal was given. Lists the values of the mean squared error between the boosting estimator and the true signal up to current boosting iteration.

    **Methods**

    +-----------------------------------------------------------------+------------------------------------------------------------------+
    | iterate( ``number_of_iterations=1`` )                           | Performs number of iterations of the boosting algorithm.         |
    +-----------------------------------------------------------------+------------------------------------------------------------------+
    | predict( ``design_observation`` )                               | Predicts the response based on the current boosting estimate.    |
    +-----------------------------------------------------------------+------------------------------------------------------------------+
    | get_discrepancy_stop( ``critical_value, max_iteration`` )       | Stops the boosting algorithm based on the discrepancy principle. |
    +-----------------------------------------------------------------+------------------------------------------------------------------+
    | get_residual_ratio_stop( ``max_iteration, alpha=0.05, K=1.2`` ) | Stops the boosting algorithm based on the residual ratios.       |
    +-----------------------------------------------------------------+------------------------------------------------------------------+
    | get_noise_estimate(``K=1``)                                     | Computes a noise estimate for the model via the scaled lasso.    |
    +-----------------------------------------------------------------+------------------------------------------------------------------+
    | get_aic_iteration(``K=2``)                                      | Computes the minimizer of a high dimensional Akaike criterion.   |
    +-----------------------------------------------------------------+------------------------------------------------------------------+
    | get_balanced_oracle( ``max_iteration`` )                        | Iterates the boosting algorithm up to the balanced oracle.       |
    +-----------------------------------------------------------------+------------------------------------------------------------------+
    """

    def __init__(self, design, response, true_signal=None):
        self.design = design
        self.response = response
        self.true_signal = true_signal

        # Parameters of the model
        self.sample_size = np.shape(design)[0]
        self.parameter_size = np.shape(design)[1]

        # Estimation quantities
        self.iteration = 0
        self.selected_components = np.array([])
        self.orth_directions = []
        self.coefficients_list = [np.zeros(self.parameter_size)]
        self.boost_estimate_list = [np.zeros(self.sample_size)]

        # Residual quantities
        self.__residual_vector = response
        self.residuals = np.array([np.mean(self.__residual_vector**2)])
        self.residual_ratios = np.array([0])

        if self.true_signal is not None:
            self.noiseless_response = self.design @ self.true_signal
            self.__error_vector = self.response - self.noiseless_response
            self.__bias2_vector = self.noiseless_response
            self.__stochastic_error_vector = np.zeros(self.sample_size)

            self.bias2 = np.array([np.mean(self.__bias2_vector**2)])
            self.stochastic_error = np.array([0])
            self.risk = np.array([np.mean(self.noiseless_response**2)])

    def iterate(self, number_of_iterations=1):
        """Performs number_of_iterations iterations of the orthogonal boosting algorithm.

        **Parameters**

        *number_of_iterations*: ``int``. Number of boosting iterations to be performed.
        """
        for _ in range(number_of_iterations):
            self.__boost_one_iteration()

    def predict(self, design_observation):
        """Predicts the output variable based on the current boosting estimate.

        **Parameters**

        *input_variable*: ``array``. The size of input_variable has to match parameter_size.
        """
        return np.dot(design_observation, self.coefficients)
        # 2025-03-10-TODO-BS: adjust coefficients.

    def get_discrepancy_stop(self, critical_value, max_iteration):
        """Early stopping for the boosting procedure based on the discrepancy principle.
        Procedure is stopped when the residuals go below critical_value or iteration
        max_iteration is reached.

        **Parameters**

        *critical_value*: ``float``. Critical value for the early stopping procedure.

        *max_iteration*: ``int``. Maximal number of iterations to be performed.
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
            return int(early_stopping_index)
        else:
            warnings.warn("Stopping index not found up to max_iteration. Returning None.", category=UserWarning)
            return None

    def get_residual_ratio_stop(self, max_iteration, alpha=0.05, K=1.2):
        """Stops the algorithm based on a residual ration criterion.

        **Parameters**

        *alpha*: ``float``. accuracy level

        *K*: ``float``. Constant for the definition.

        *max_iteration*: ``int``. Maximal number of iterations to be performed.
        """
        critical_value = 1 - 4 * K * np.log(self.parameter_size / alpha) / self.sample_size
        if self.residual_ratios[self.iteration] >= critical_value:
            # argmax takes the first instance of True in the true-false array
            early_stopping_index = np.argmax(self.residual_ratios >= critical_value)
            return int(early_stopping_index)

        if self.residual_ratios[self.iteration] < critical_value:
            while self.residual_ratios[self.iteration] < critical_value and self.iteration <= max_iteration:
                self.iterate(1)

        if self.residual_ratios[self.iteration] >= critical_value:
            early_stopping_index = self.iteration
            return int(early_stopping_index)
        else:
            warnings.warn("Stopping index not found up to max_iteration. Returning None.", category=UserWarning)
            return None

    def get_balanced_oracle(self, max_iteration):
        """Performs iterations of the orthogonal boosting algorithm until the
        balanced oracle index at which the squared bias is smaller than the
        stochastic error is reached.

        **Parameters**

        *max_iteration*: ``int``. Maximal number of iterations to be performed.
        """
        if self.true_signal is None:
            return "This method is only available when the true signal is known"
        else:
            if self.bias2[self.iteration] <= self.stochastic_error[self.iteration]:
                # argmax takes the first instance of True in the true-false array
                balanced_oracle = np.argmax(self.bias2 <= self.stochastic_error)
                return int(balanced_oracle)

            if self.bias2[self.iteration] > self.stochastic_error[self.iteration]:
                while (
                    self.bias2[self.iteration] > self.stochastic_error[self.iteration]
                    and self.iteration <= max_iteration
                ):
                    self.iterate(1)

            if self.bias2[self.iteration] <= self.stochastic_error[self.iteration]:
                balanced_oracle = self.iteration
                return int(balanced_oracle)
            else:
                warnings.warn("Balanced oracle not found up to max_iteration. Returning None.", category=UserWarning)
                return None

    def get_noise_estimate(self, K=1):
        """Computes an estimator for the noise level sigma^2 of the model via the scaled Lasso.

        **Parameters**

        *K*: ``float``. Constant in the definition. Defaults to 1, which is the choice from the scaled Lasso paper.
        """
        iter = 0
        max_iter = 50
        tolerance = 1 / self.sample_size
        estimation_difference = 2 * tolerance
        alpha_0 = np.sqrt(K * np.log(self.parameter_size) / self.sample_size)

        lasso = linear_model.Lasso(alpha_0, fit_intercept=False)
        lasso.fit(self.design, self.response)
        noise_estimate = np.mean((self.response - lasso.predict(self.design)) ** 2)

        while estimation_difference > tolerance and iter <= max_iter:
            alpha = np.sqrt(noise_estimate) * alpha_0
            lasso = linear_model.Lasso(alpha, fit_intercept=False)
            lasso.fit(self.design, self.response)

            new_noise_estimate = np.mean((self.response - lasso.predict(self.design)) ** 2)
            estimation_difference = np.abs(new_noise_estimate - noise_estimate)
            noise_estimate = new_noise_estimate

            iter = iter + 1

        return noise_estimate

    def get_aic_iteration(self, K=2, max_iteration=None):
        """Computes the iteration index minimizing a high dimensional Akaike criterion.

        **Parameters**

        *K*: ``float``. Constant in the definition. Defaults to 2, which is common in the literature.
        """
        if max_iteration == None:
            max_iteration = self.iteration

        noise_estimate = self.get_noise_estimate()
        dim_penalty = (
            np.arange(0, max_iteration + 1) * K * noise_estimate * np.log(self.parameter_size) / self.sample_size
        )
        aic = self.residuals[0 : max_iteration + 1] + dim_penalty
        return np.argmin(aic)

    def __boost_one_iteration(self):
        """Performs one iteration of the orthogonal boosting algorithm"""
        # Compute weak learner index and check for repetition
        weak_learner_index = self.__compute_weak_learner_index()
        component_selected_repeatedly = False
        for iter_num in range(self.iteration):
            if weak_learner_index == self.selected_components[iter_num]:
                component_selected_repeatedly = True

        if component_selected_repeatedly:
            print("Algorithm terminated")
        else:
            # Update selected variables
            self.selected_components = np.append(self.selected_components, weak_learner_index)
            self.__update_orth_directions(self.design[:, weak_learner_index])
            weak_learner = self.orth_directions[-1]

            # Update estimation quantities
<<<<<<< HEAD
            coefficient_entry                    = np.dot(self.response, weak_learner) / self.sample_size
            new_coefficients                     = np.copy(self.coefficients_list[self.iteration])
=======
            coefficient_entry = np.dot(self.response, weak_learner) / self.sample_size
            new_coefficients = self.coefficients_list[self.iteration]
>>>>>>> 803b92c518343593d07a7dcad388e7294b7bf0db
            new_coefficients[weak_learner_index] = coefficient_entry
            self.coefficients_list.append(new_coefficients)

            new_boost_estimate = self.boost_estimate_list[self.iteration] + coefficient_entry * weak_learner
            self.boost_estimate_list.append(new_boost_estimate)

            self.__residual_vector = self.response - new_boost_estimate
            new_residuals = np.mean(self.__residual_vector**2)
            new_residual_ratio = new_residuals / self.residuals[self.iteration]
            self.residuals = np.append(self.residuals, new_residuals)
            self.residual_ratios = np.append(self.residual_ratios, new_residual_ratio)

            self.iteration = self.iteration + 1

            # Update theoretical quantities
            if self.true_signal is not None:
                self.__update_risk()
                self.__update_bias2(weak_learner)
                self.__update_stochastic_error(weak_learner)

    def __compute_weak_learner_index(self):
        """Computes the column index of the design matrix which reduces the
        resiudals the most
        """
        decreased_residuals = np.zeros(self.parameter_size)
        for j in range(self.parameter_size):
            direction = self.design[:, j]
            direction_norm = np.sqrt(np.mean(direction**2))
            direction = direction / direction_norm
            step_size = np.dot(self.__residual_vector, direction) / self.sample_size
            decreased_residuals[j] = np.mean((self.__residual_vector - step_size * direction) ** 2)
        weak_learner_index = np.argmin(decreased_residuals)
        return weak_learner_index

    def __update_orth_directions(self, direction):
        """Updates the list of orthogonal directions"""
        if self.iteration == 0:
            direction_norm = np.sqrt(np.mean(direction**2))
            direction = direction / direction_norm
        else:
            for orth_direction in self.orth_directions:
                dot_product = np.dot(direction, orth_direction) / self.sample_size
                direction = direction - dot_product * orth_direction
            direction_norm = np.sqrt(np.mean(direction**2))
            direction = direction / direction_norm
        self.orth_directions.append(direction)

    def __update_risk(self):
        new_risk = np.mean((self.noiseless_response - self.boost_estimate_list[self.iteration]) ** 2)
        self.risk = np.append(self.risk, new_risk)

    def __update_bias2(self, weak_learner):
        coefficient = np.dot(self.noiseless_response, weak_learner) / self.sample_size
        self.__bias2_vector = self.__bias2_vector - coefficient * weak_learner
        new_bias2 = np.mean(self.__bias2_vector**2)
        self.bias2 = np.append(self.bias2, new_bias2)

    def __update_stochastic_error(self, weak_learner):
        coefficient = np.dot(self.__error_vector, weak_learner) / self.sample_size
        self.__stochastic_error_vector = self.__stochastic_error_vector + coefficient * weak_learner
        new_stochastic_error = np.mean(self.__stochastic_error_vector**2)
        self.stochastic_error = np.append(self.stochastic_error, new_stochastic_error)

    def __str__(self):
        """Return a string representation of the L2_boost model"""
        lines = []
        lines.append("L2_boost class")
        lines.append("=" * 30)
        lines.append(f"Problem dimensions: {self.sample_size} × {self.parameter_size}")
        lines.append(f"Current iteration: {self.iteration}")

        if self.iteration > 0:
            lines.append(f"Current residual: {self.residuals[self.iteration]:.6f}")
            lines.append(f"Selected components: {len(self.selected_components)}")
            if len(self.residual_ratios) > self.iteration:
                lines.append(f"Current residual ratio: {self.residual_ratios[self.iteration]:.6f}")

        # Check if theoretical quantities are available
        has_true_signal = self.true_signal is not None
        lines.append(f"True signal available: {'Yes' if has_true_signal else 'No'}")

        if has_true_signal and self.iteration > 0:
            lines.append(f"Bias²: {self.bias2[self.iteration]:.6f}")
            lines.append(f"Stochastic error: {self.stochastic_error[self.iteration]:.6f}")
            lines.append(f"Risk: {self.risk[self.iteration]:.6f}")

        return "\n".join(lines)

    def __repr__(self):
        """Return a technical representation of the L2_boost model"""
        has_true_signal = self.true_signal is not None

        # Build constructor-like representation
        args = [
            f"design=array({self.sample_size}x{self.parameter_size})",
            f"response=array({self.sample_size},)",
            f"true_signal={'array' if self.true_signal is not None else 'None'}",
        ]

        base_repr = f"L2_boost({', '.join(args)})"

        # Add current state information
        state_info = f" [iteration={self.iteration}"
        if self.iteration > 0:
            state_info += f", residual={self.residuals[self.iteration]:.6f}"
            state_info += f", selected_components={len(self.selected_components)}"
        if has_true_signal and self.iteration > 0:
            state_info += f", bias2={self.bias2[self.iteration]:.6f}"
            state_info += f", stochastic_error={self.stochastic_error[self.iteration]:.6f}"
            state_info += f", risk={self.risk[self.iteration]:.6f}"
        state_info += "]"

        return base_repr + state_info
