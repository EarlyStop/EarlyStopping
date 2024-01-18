import numpy as np

class L2_boost():
    """ L2-boosting algorithm for high dimensional linear models.

    Parameters
    ----------
    input_matrix: array
        nxp-Design matrix of the linear model.

    output_variable: array
        n-dim vector of the observed data in the linear model.

    true_signal: array or None, default = None 
        For simulation purposes only. For simulated data the true signal can be
        included to compute theoretical quantities such as the bias and the mse
        alongside the boosting procedure.

    Attributes
    ----------
    sample_size: int
        Sample size of the linear model.
    
    para_size: int
        Parameter size of the linear model.

    iter: int
        Current boosting iteration of the algorithm.

    boost_estimate: array
        Boosting estimate at the current iteration for the data given in
        input_matrix.

    residuals: array
        Lists the sequence of the residual mean of squares betwean the data and
        the boosting estimator.

    bias2: array
        Only exists if true_signal was given. Lists the values of the squared
        bias up to current boosting iteration.

    stoch_error: array
        Only exists if true_signal was given. Lists the values of a stochastic
        error term up to current boosting iteration.

    mse: array
        Only exists if true_signal was given. Lists the values of the mean
        squared error betwean the boosting estimator and the true signal up to
        current boosting iteration.

    Methods
    -------
    +-------------------------------------+-------------------------------------------------------------------+
    | boost(iter_num = 1)                 | Performs iter_num iterations of the boosting algorithm.           |
    +-------------------------------------+-------------------------------------------------------------------+
    | boost_to_early_stop(crit, max_iter) | Early stops the boosting algorithm.                               | 
    +--------------------------------+----+-------------------------------------------------------------------+
    | boost_to_balanced_oracle(self)      | Iterates the boosting algorithm up to the balanced oracle.        |
    +------+------------------------------+-------------------------------------------------------------------+
    | predict(input_variable)             | Predicts the output based on the current boosting estimate.       |
    +------+------------------------------+-------------------------------------------------------------------+
    """

    def __init__(self, input_matrix, output_variable, true_signal = None):
        self.input_matrix    = input_matrix
        self.output_variable = output_variable
        self.true_signal     = true_signal

        # Parameters of the model
        self.sample_size = np.shape(input_matrix)[0]
        self.para_size   = np.shape(input_matrix)[1]

        # Estimation quantities
        self.iter               = 0
        self.selected_components = np.array([])
        self.orth_directions     = []
        self.coefficients       = np.zeros(self.para_size)
        self.boost_estimate      = np.zeros(self.sample_size)

        # Residual quantities
        self.__residual_vector = output_variable
        self.residuals        = np.array([np.mean(self.__residual_vector**2)])

        if self.true_signal is not None:
            self.__error_vector      = self.output_variable - self.true_signal
            self.__bias2_vector      = self.true_signal
            self.__stoch_error_vector = np.zeros(self.sample_size)

            self.bias2      = np.array([np.mean(self.__bias2_vector**2)])
            self.stoch_error = np.array([0])
            self.mse        = np.array([np.mean(self.true_signal**2)])

    def boost(self, iter_num = 1):
        """ Performs iter_num iterations of the orthogonal boosting algorithm.

            Parameters
            ----------
            iter_num: int
                Number of boosting iterations to be performed.
                
        """
        for index in range(iter_num):
            self.__boost_one_iteration()

    def boost_to_early_stop(self, crit, max_iter):
        """ Early stopping for the boosting procedure.
            Procedure is stopped when the residuals go below crit or iteration
            max_iter is reached.

            Parameters
            ----------
            crit: float
                Critical value for the early stopping procedure.
            max_iter: int
                Maximal number of iterations to be performed.
                
        """
        while self.residuals[self.iter] > crit and self.iter <= max_iter:
            self.__boost_one_iteration()

    def boost_to_balanced_oracle(self):
        """ Performs iterations of the orthogonal boosting algorithm until the
            balanced oracle index at which the squared bias is smaller than the
            stochastic error is reached.
        """
        if self.true_signal is None:
            return "This method is only available when the true signal is known"
        else:
            while self.bias2[self.iter] > self.stoch_error[self.iter]:
                self.__boost_one_iteration()

    def predict(self, input_variable):
        """ Predicts the output variable based on the current boosting estimate

            Parameters
            ----------
            input_variable: array
                The size of input_variable has to match para_size.
        """
        return np.dot(input_variable, self.coefficients)

    def __boost_one_iteration(self):
        """Performs one iteration of the orthogonal boosting algorithm"""
        # Compute weak learner index and check for repetition
        weak_learner_index            = self.__compute_weak_learner_index()
        component_selected_repeatedly = False
        for iter_num in range(self.iter):
            if weak_learner_index == self.selected_components[iter_num]:
                component_selected_repeatedly = True

        if component_selected_repeatedly:
            print("Algorithm terminated")
        else:
            # Update selected variables
            self.selected_components = np.append(self.selected_components,
                                                weak_learner_index)
            self.__update_orth_directions(self.input_matrix[:, weak_learner_index])
            weak_learner = self.orth_directions[-1]

            # Update estimation quantities
            coefficient           = np.dot(self.output_variable, weak_learner) / \
                                    self.sample_size
            self.coefficients[weak_learner_index] = coefficient
            self.boost_estimate    = self.boost_estimate + coefficient * weak_learner
            self.__residual_vector = self.output_variable - self.boost_estimate
            new_residuals          = np.mean(self.__residual_vector**2)
            self.residuals        = np.append(self.residuals, new_residuals)
            self.iter             = self.iter + 1

            # Update theoretical quantities
            if self.true_signal is not None:
                self.__update_mse()
                self.__update_bias2(weak_learner)
                self.__update_stochastic_error(weak_learner)

    def __compute_weak_learner_index(self):
        """ Computes the column index of the design matrix which reduces the
            resiudals the most
        """
        decreased_residuals = np.zeros(self.para_size)
        for j in range(self.para_size):
            direction             = self.input_matrix[:, j]
            direction_norm         = np.sqrt(np.mean(direction**2))
            direction             = direction / direction_norm
            step_size              = np.dot(self.__residual_vector, direction) / \
                                    self.sample_size
            decreased_residuals[j] = np.mean((self.__residual_vector -
                                             step_size * direction)**2)
        weak_learner_index = np.argmin(decreased_residuals)
        return weak_learner_index

    def __update_orth_directions(self, direction):
        """Updates the list of orthogonal directions"""
        if self.iter == 0:
            direction_norm = np.sqrt(np.mean(direction**2))
            direction     = direction / direction_norm
        else:
            for orth_direction in self.orth_directions:
                dot_product = np.dot(direction, orth_direction) / self.sample_size
                direction  = direction -  dot_product * orth_direction
            direction_norm = np.sqrt(np.mean(direction**2))
            direction     = direction / direction_norm
        self.orth_directions.append(direction)

    def __update_mse(self):
        new_mse   = np.mean((self.true_signal - self.boost_estimate)**2)
        self.mse = np.append(self.mse, new_mse)

    def __update_bias2(self, weak_learner):
        coefficient        = np.dot(self.true_signal, weak_learner) / \
                             self.sample_size
        self.__bias2_vector = self.__bias2_vector - coefficient * weak_learner
        new_bias2           = np.mean(self.__bias2_vector**2)
        self.bias2         = np.append(self.bias2, new_bias2)

    def __update_stochastic_error(self, weak_learner):
        coefficient             = np.dot(self.__error_vector, weak_learner) / \
                                 self.sample_size
        self.__stoch_error_vector = self.__stoch_error_vector + \
                                  coefficient * weak_learner
        new_stoch_error           = np.mean(self.__stoch_error_vector**2)
        self.stoch_error         = np.append(self.stoch_error, new_stoch_error)
