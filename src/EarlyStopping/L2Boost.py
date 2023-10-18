import numpy as np

class L2Boost(object):
    """ L2-boosting algorithm for high dimensional linear models.

    Parameters
    ----------
    inputMatrix: array
        nxp-Design matrix of the linear model.

    outputVariable: array
        n-dim vector of the observed data in the linear model.

    trueSignal: array or None, default = None 
        For simulation purposes only. For simulated data the true signal can be
        included to compute theoretical quantities such as the bias and the mse
        alongside the boosting procedure.

    Attributes
    ----------
    sampleSize: int
        Sample size of the linear model
    
    paraSize: int
        Parameter size of the linear model

    iter: int
        Current boosting iteration of the algorithm

    boostEstimate: array
        Boosting estimate at the current iteration for the data given in
        inputMatrix

    residuals: array
        Lists the sequence of the residual mean of squares betwean the data and
        the boosting estimator.

    bias2: array
        Only exists if trueSignal was given. Lists the values of the squared
        bias up to current boosting iteration.

    stochError: array
        Only exists if trueSignal was given. Lists the values of a stochastic
        error term up to current boosting iteration.

    mse: array
        Only exists if trueSignal was given. Lists the values of the mean
        squared error betwean the boosting estimator and the true signal up to
        current boosting iteration.
    """

    def __init__(self, inputMatrix, outputVariable, trueSignal = None):
        self.inputMatrix    = inputMatrix
        self.outputVariable = outputVariable
        self.trueSignal     = trueSignal
 
        # Parameters of the model
        self.sampleSize = np.shape(inputMatrix)[0]
        self.paraSize   = np.shape(inputMatrix)[1]

        # Estimation quantities
        self.iter               = 0
        self.selectedComponents = np.array([])
        self.orthDirections     = []
        self.coefficients       = np.zeros(self.paraSize)
        self.boostEstimate      = np.zeros(self.sampleSize)

        # Residual quantities
        self.__residualVector = outputVariable
        self.residuals        = np.array([np.mean(self.__residualVector**2)])
   
        if self.trueSignal is not None:
            self.__errorVector      = self.outputVariable - self.trueSignal 
            self.__bias2Vector      = self.trueSignal
            self.__stochErrorVector = np.zeros(self.sampleSize)

            self.bias2      = np.array([np.mean(self.__bias2Vector**2)])
            self.stochError = np.array([0])
            self.mse        = np.array([np.mean(self.trueSignal**2)])

    def boost(self, m = 1):
        """Performs m iterations of the orthogonal boosting algorithm"""
        for index in range(m): self.__boostOneIteration()

    def boostToEarlyStop(self, crit, maxIter):
        """Early stopping for the boosting procedure

            Procedure is stopped when the residuals go below crit or iteration
            maxIter is reached.
        """
        while self.residuals[self.iter] > crit and self.iter <= maxIter:
            self.__boostOneIteration()

    def boostToBalancedOracle(self):
        """
            Performs iterations of the orthogonal boosting algorithm until the
            balanced oracle index at which the squared bias is smaller than the
            stochastic error is reached.
        """
        if self.trueSignal is None:
            return("This method is only available when the true signal is known")
        else: 
            while self.bias2[self.iter] > self.stochError[self.iter]:
                self.__boostOneIteration()

    # TODO Select iteration method. 

    def predict(self, inputVariable):
        """Predicts the output variable based on the current boosting estimate"""
        return np.dot(inputVariable, self.coefficients)
  
    def __boostOneIteration(self):
        """Performs one iteration of the orthogonal boosting algorithm"""
        # Compute weak learner index and check for repetition
        weakLearnerIndex            = self.__computeWeakLearnerIndex()
        componentSelectedRepeatedly = False
        for m in range(self.iter):
            if weakLearnerIndex == self.selectedComponents[m]:
                componentSelectedRepeatedly = True

        if componentSelectedRepeatedly: 
            print("Algorithm terminated")
        else:
            # Update selected variables
            self.selectedComponents = np.append(self.selectedComponents,
                                                weakLearnerIndex)
            self.__updateOrthDirections(self.inputMatrix[:, weakLearnerIndex])
            weakLearner = self.orthDirections[-1]

            # Update estimation quantities
            coefficient           = np.dot(self.outputVariable, weakLearner) / \
                                    self.sampleSize
            self.coefficients[weakLearnerIndex] = coefficient
            self.boostEstimate    = self.boostEstimate + coefficient * weakLearner
            self.__residualVector = self.outputVariable - self.boostEstimate
            newResiduals          = np.mean(self.__residualVector**2)
            self.residuals        = np.append(self.residuals, newResiduals)
            self.iter             = self.iter + 1

            # Update theoretical quantities
            if self.trueSignal is not None:
                self.__updateMse()
                self.__updateBias2(weakLearner)
                self.__updateStochasticError(weakLearner)

    def __computeWeakLearnerIndex(self):
        """ Computes the column index of the design matrix which reduces the
            resiudals the most
        """
        decreasedResiduals = np.zeros(self.paraSize)
        for j in range(self.paraSize):
            direction             = self.inputMatrix[:, j]
            directionNorm         = np.sqrt(np.mean(direction**2))
            direction             = direction / directionNorm
            stepSize              = np.dot(self.__residualVector, direction) / \
                                    self.sampleSize
            decreasedResiduals[j] = np.mean((self.__residualVector -
                                             stepSize * direction)**2)
        weakLearnerIndex = np.argmin(decreasedResiduals)
        return(weakLearnerIndex)

    def __updateOrthDirections(self, direction):
        """Updates the list of orthogonal directions"""
        if self.iter == 0:
            directionNorm = np.sqrt(np.mean(direction**2))
            direction     = direction / directionNorm
        else: 
            for orthDirection in self.orthDirections:
                dotProduct = np.dot(direction, orthDirection) / self.sampleSize
                direction  = direction -  dotProduct * orthDirection
            directionNorm = np.sqrt(np.mean(direction**2))
            direction     = direction / directionNorm
        self.orthDirections.append(direction)
  
    def __updateMse(self):
        newMse   = np.mean((self.trueSignal - self.boostEstimate)**2)
        self.mse = np.append(self.mse, newMse)

    def __updateBias2(self, weakLearner):
        coefficient        = np.dot(self.trueSignal, weakLearner) / \
                             self.sampleSize
        self.__bias2Vector = self.__bias2Vector - coefficient * weakLearner
        newBias2           = np.mean(self.__bias2Vector**2)
        self.bias2         = np.append(self.bias2, newBias2)

    def __updateStochasticError(self, weakLearner):
        coefficient             = np.dot(self.__errorVector, weakLearner) / \
                                 self.sampleSize
        self.__stochErrorVector = self.__stochErrorVector + \
                                  coefficient * weakLearner
        newStochError           = np.mean(self.__stochErrorVector**2) 
        self.stochError         = np.append(self.stochError, newStochError)
