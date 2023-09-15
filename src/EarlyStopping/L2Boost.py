# TODO: Right place to import numpy?
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

    """

    def __init__(self, inputMatrix, outputVariable, trueSignal = None):
        self.inputMatrix    = inputMatrix
        self.outputVariable = outputVariable
        self.trueSignal     = trueSignal
 
        # Parameters of the model
        self.sampleSize = np.shape(inputMatrix)[0]
        self.paraSize   = np.shape(inputMatrix)[1]
