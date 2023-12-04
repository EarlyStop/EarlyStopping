import numpy as np

class landweber:
    """ A class to perform estimation using the Landweber iterative method.

    Parameters
    ----------
    input_matrix: array
        nxp design matrix of the linear model.

    response_variable: array
        n-dim vector of the observed data in the linear model.

    true_signal: array or None, default = None 
        d-dim vector
        For simulation purposes only. For simulated data the true signal can be
        included to compute theoretical quantities such as the bias and the mse
        alongside the iterative procedure.

    Attributes
    ----------
    sample_size: int
        Sample size of the linear model
    
    para_size: int
        Parameter size of the linear model

    iter: int
        Current Landweber iteration of the algorithm

    early_stopping_iter: int
        Early Stopping iteration index

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

    def __init__(self, input_matrix, response_variable, learning_rate = 1, true_signal = None):
        self.input_matrix       = input_matrix
        self.response_variable  = response_variable
        self.learning_rate      = learning_rate
        self.true_signal        = true_signal
 
        # Parameters of the model
        self.sample_size = np.shape(input_matrix)[0]
        self.para_size   = np.shape(input_matrix)[1]

        # Estimation quantities
        self.iter               = 0
 #       self.early_stopping_iter = 
        self.landweber_estimate     = np.zeros(self.para_size)

        # Residual quantities
        self.__residual_vector = response_variable
        self.residuals         = np.array([np.sum(self.__residual_vector**2)])

#        if self.true_signal is not None:
#            self.mse = np.array([])
#            self.mse_weak = np.array([])
   
#        if self.true_signal is not None:
#            self.__error_vector     = self.response_variable - np.dot(self.input_matrix, self.true_signal) 
#            self.__strong_bias2_vector     = self.true_signal
#            self.__strong_variance_vector  = np.zeros(self.para_size)
#            self.__weak_bias2_vector     = np.dot(self.input_matrix,self.true_signal)
#            self.__weak_variance_vector  = np.zeros(self.sample_size)#

#            self.strong_bias2      = np.array([np.sum(self.__strong_bias2_vector**2)])
#            self.strong_variance   = np.array([0])
#            self.strong_error      = self.strong_bias2

#            self.weak_bias2        = np.array([np.sum(self.__weak_bias2_vector**2)])
#            self.weak_variance     = np.array([0])
#            self.weak_error        = self.weak_bias2

    def landweber(self, iter_num = 1):
        """Performs iter_num iterations of the Landweber algorithm
        
        Parameters
        ----------
        iter_num: int, default=1
            The number of iterations to perform.
        """
        for _ in range(iter_num):
            self.__landweber_one_iteration()
        
    def __landweber_one_iteration(self):
        """Performs one iteration of the Landweber algorithm"""
        
        self.landweber_estimate = self.landweber_estimate + self.learning_rate * np.matmul(np.transpose(self.input_matrix),self.response_variable - np.matmul(self.input_matrix,self.landweber_estimate))

        # Update estimation quantities
        self.__residual_vector  = self.response_variable - np.matmul(self.input_matrix,self.landweber_estimate)
        new_residuals           = np.sum(self.__residual_vector**2)
        self.residuals          = np.append(self.residuals, new_residuals)
        self.iter               = self.iter + 1

    def landweber_to_early_stop(self, crit, max_iter):
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
        while self.residuals[self.iter] > crit and self.iter <= max_iter:
            self.__landweber_one_iteration()

 #       # Update theoretical quantities
 #       if self.true_signal is not None:
 #            self.__update_strong_error()
#            self.__update_strong_bias2()
#            self.__update_strong_variance()
 #            self.__update_weak_error()
#            self.__update_weak_bias2()
#            self.__update_weak_variance()
        
    # def __update_strong_error(self):
    #     new_mse   = np.mean((self.true_signal - self.landweber_estimate)**2)
    #     self.mse = np.append(self.mse, new_mse)

    # def __update_weak_error(self): 
    #     new_mse_weak   = np.mean(( np.dot(self.input_matrix,self.true_signal) -  np.dot(self.input_matrix,self.landweber_estimate))**2)
    #     self.mse_weak = np.append(self.mse_weak, new_mse_weak)

#    def __update_bias2(self, weak_learner):
#        coefficient        = np.dot(self.true_signal, weak_learner) / \
#                             self.sample_size
#        self.__bias2_vector = self.__bias2_vector - coefficient * weak_learner
#        new_bias2           = np.mean(self.__bias2_vector**2)
#        self.bias2         = np.append(self.bias2, new_bias2)

#    def __update_stochastic_error(self, weak_learner):
#        coefficient             = np.dot(self.__error_vector, weak_learner) / \
#                                 self.sample_size
#        self.__stoch_error_vector = self.__stoch_error_vector + \
#                                  coefficient * weak_learner
#        new_stoch_error           = np.mean(self.__stoch_error_vector**2)
#        self.stoch_error         = np.append(self.stoch_error, new_stoch_error)