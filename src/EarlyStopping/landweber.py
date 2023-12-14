import numpy as np
import scipy
from scipy import sparse

class Landweber:
    """ A class to perform estimation using the Landweber iterative method.

    Parameters
    ----------
    input_matrix: array
        nxp design matrix of the linear model.

    response_variable: array
        n-dim vector of the observed data in the linear model.

    starting_value: array, default: None
        Determines the zeroth step of the iterative procedure. (Defaults to zero).

    true_signal: array, default = None 
        d-dim vector
        For simulation purposes only. For simulated data the true signal can be
        included to compute theoretical quantities such as the bias and the mse
        alongside the iterative procedure.

    true_noise_level: float, default = None
        For simulation purposes only. Corresponds to the standard deviation 
        of normally distributed noise contributing to the response variable.
        Allows the analytic computation of the strong and weak variance.

    Attributes
    ----------
    sample_size: int
        Sample size of the linear model
    
    para_size: int
        Parameter size of the linear model

    iter: int
        Current Landweber iteration of the algorithm

    early_stopping_index: int
        Early Stopping iteration index (Is set to None if no early stopping is performed)

    landweber_estimate: array
        Landweber estimate at the current iteration for the data given in
        inputMatrix

    residuals: array
        Lists the sequence of the squared residuals between the observed data and
        the Landweber estimator.

    strong_bias2: array
        Only exists if trueSignal was given. Lists the values of the strong squared
        bias up to current Landweber iteration.

    strong_variance: array
        Only exists if trueSignal was given. Lists the values of the strong variance 
        up to current Landweber iteration.

    strong_error: array
        Only exists if trueSignal was given. Lists the values of the strong norm error 
        between the Landweber estimator and the true signal up to
        current Landweber iteration.
    
    weak_bias2: array
        Only exists if trueSignal was given. Lists the values of the weak squared
        bias up to current Landweber iteration.

    weak_variance: array
        Only exists if trueSignal was given. Lists the values of the weak variance 
        up to current Landweber iteration.

    weak_error: array
        Only exists if trueSignal was given. Lists the values of the weak norm error 
        between the Landweber estimator and the true signal up to
        current Landweber iteration.

    Methods
    -------
    landweber(iter_num=1)
        Performs a specified number of iterations of the Landweber algorithm.

    landweber_to_early_stop(crit, max_iter)
        Applies early stopping to the Landweber iterative procedure.
    """
    def __init__(self, input_matrix,
                 response_variable,
                 learning_rate = 1,
                 critical_value = None,
                 starting_value = None,
                 true_signal = None,
                 true_noise_level = None):
        self.input_matrix = input_matrix
        self.response_variable = response_variable
        self.learning_rate = learning_rate
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level #sigma
        self.critical_value = critical_value

        # Parameters of the model
        self.sample_size = np.shape(input_matrix)[0]
        self.para_size   = np.shape(input_matrix)[1]
        
        # Determine starting value for the procedure
        if starting_value is None:
            self.starting_value = np.zeros(self.para_size)
        else:
            self.starting_value = starting_value

        # Estimation quantities
        self.iter = 0
        self.landweber_estimate = self.starting_value
        self.early_stopping_index = None

        self.congruency_matrix = np.transpose(self.input_matrix) @ self.input_matrix
        
        if self.critical_value is None and self.true_noise_level is None:
            # maximum likelihood estimator
            # least_squares_estimator = np.linalg.solve(self.congruency_matrix, np.transpose(self.input_matrix) @ self.response_variable)
            sparse_least_squares_estimator, istop = sparse.linalg.lsqr(self.congruency_matrix, np.transpose(self.input_matrix) @ self.response_variable)[:2]
            noise_level_estimate = np.sum((self.response_variable - self.input_matrix @ sparse_least_squares_estimator)**2)/self.sample_size
            self.critical_value = noise_level_estimate * self.sample_size
        elif self.true_noise_level is not None:
            # if the true noise level is given, it does not need to be estimated
            self.critical_value = self.true_noise_level**2 * self.sample_size

        # Residual quantities
        self.__residual_vector = response_variable
        self.residuals = np.array([np.sum(self.__residual_vector**2)])

        if (self.true_signal is not None) and (self.true_noise_level is not None):            
            # initialize matrices required for computing the strong/weak bias and variance

            self.inverse_congruency_matrix = np.linalg.inv(self.congruency_matrix)
            self.perturbation_congruency_matrix = (np.eye(self.para_size) - self.learning_rate * self.congruency_matrix)
            self.weak_perturbation_congruency_matrix = (self.input_matrix @ self.perturbation_congruency_matrix)
            self.perturbation_congruency_matrix_power = self.perturbation_congruency_matrix
            
            # initialize strong/weak bias and variance
            self.expectation_estimator = self.starting_value
            self.strong_bias2 = np.array([np.sum(self.true_signal**2)])
            self.weak_bias2 = np.array([np.sum((self.input_matrix @ self.true_signal)**2)])

            self.strong_variance = np.array([0])
            self.weak_variance = np.array([0])

            self.strong_error = self.strong_bias2 + self.strong_variance
            self.weak_error = self.weak_bias2 + self.weak_variance

    def landweber(self, iter_num = 1):
        """Performs iter_num iterations of the Landweber algorithm
        
        Parameters
        ----------
        iter_num: int, default=1
            The number of iterations to perform.
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
        self.expectation_estimator = (self.expectation_estimator
            + self.learning_rate * (self.congruency_matrix)
            @ (self.true_signal - self.expectation_estimator))
        
        self.perturbation_congruency_matrix_power = (self.perturbation_congruency_matrix_power
            @ self.perturbation_congruency_matrix)

    def __update_strong_bias2(self):
        """Update strong bias

        Given the expectation of the estimator the squared bias is given by :math: `b_k^{2} = \\Vert (I-hA^{\\top}A)(\\mu-m_{k-1}) \\Vert^{2}` 
        """
        new_strong_bias2 = np.sum((self.perturbation_congruency_matrix
            @ (self.true_signal - self.expectation_estimator))**2)
        self.strong_bias2 = np.append(self.strong_bias2, new_strong_bias2)

    def __update_weak_bias2(self):
        """Update weak bias

        Given the expectation of the estimator the (weak)-squared bias is given by :math: `b_k^{2} = \\Vert A(I-hA^{\\top}A)(\\mu-m_{k-1}) \\Vert^{2}` 
        """
        new_weak_bias2 = np.sum((self.weak_perturbation_congruency_matrix
            @ (self.true_signal - self.expectation_estimator))**2)
        self.weak_bias2 = np.append(self.weak_bias2, new_weak_bias2)

    def __update_strong_variance(self):
        """update strong variance
        
        The strong variance in the m-th iteration is given by 
        :math: `\\sigma**2 \\mathrm{tr}(h^{-1}(A^{\\top}A)^{-1}(I-(I-hA^{\\top}A)^{m}))`
        """
        new_strong_variance = (self.true_noise_level**2 *
            np.trace(self.learning_rate**(-1) * self.inverse_congruency_matrix
            @ np.square(np.eye(self.para_size) - self.perturbation_congruency_matrix_power)))
        self.strong_variance = np.append(self.strong_variance, new_strong_variance)
        
    def __update_weak_variance(self):
        """update weak variance
        The weak variance in the m-th iteration is given by 
        :math: `\\sigma**2 \\mathrm{tr}((I-(I-hA^{\\top}A)^{m}))`
        """
        new_weak_variance = (self.true_noise_level**2 *
            np.trace(np.square(np.eye(self.para_size) - self.perturbation_congruency_matrix_power)))
        self.weak_variance = np.append(self.weak_variance, new_weak_variance)

    def __landweber_one_iteration(self):
        """Performs one iteration of the Landweber algorithm"""
        self.landweber_estimate = (self.landweber_estimate
            + self.learning_rate * np.matmul(np.transpose(self.input_matrix),
            self.response_variable - np.matmul(self.input_matrix,self.landweber_estimate)))

        # Update estimation quantities
        self.__residual_vector  = (self.response_variable
            - np.matmul(self.input_matrix,self.landweber_estimate))
        new_residuals = np.sum(self.__residual_vector**2)
        self.residuals = np.append(self.residuals, new_residuals)
        self.iter = self.iter + 1

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

    def landweber_to_early_stop(self, max_iter):
        """Early stopping for the Landweber procedure

            Procedure is stopped when the residuals go below crit or iteration
            max_iter is reached.

        Parameters
        ----------
        crit: float
            The criterion for stopping. The procedure stops when the residual is below this value.

        max_iter: int
            The maximum number of iterations to perform.
        """
        while self.residuals[self.iter] > self.critical_value and self.iter < max_iter:
            self.__landweber_one_iteration()
        self.early_stopping_index = self.iter

    def landweber_gather_all(self, max_iter):
        """Gather all relevant simulation data (Runs the algorithm till max_iter) but tracks the early stopping index."""
        self.landweber_to_early_stop(max_iter)
        if max_iter > self.early_stopping_index:
            self.landweber(max_iter - self.early_stopping_index)
