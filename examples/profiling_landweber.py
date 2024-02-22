import numpy as np
import EarlyStopping as es
from scipy.sparse import dia_matrix
import cProfile

np.random.seed(42)
D = 10000
indices = np.arange(D) + 1
design_matrix = dia_matrix(np.diag(1 / (np.sqrt(indices))))
signal_supersmooth = 5 * np.exp(-0.1 * indices)

NOISE_LEVEL = 0.01
noise = np.random.normal(0, NOISE_LEVEL, D)
observation_supersmooth = noise + design_matrix @ signal_supersmooth

models_supersmooth = es.Landweber(
    design_matrix, observation_supersmooth, true_noise_level=NOISE_LEVEL, true_signal=signal_supersmooth
)

iterations = 100
cProfile.run('models_supersmooth.landweber_gather_all(iterations)')
#models_supersmooth.landweber_gather_all(iterations)

#models_supersmooth.landweber_gather_all(iterations)