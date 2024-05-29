for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]



import numpy as np
import importlib
import EarlyStopping as es
from scipy.sparse import dia_matrix

importlib.reload(es)


# Set parameters
sample_size = 1000
parameter_size = sample_size
max_iter = sample_size
noise_level = 0.01
critical_value = sample_size * (noise_level**2)

# Create diagonal design matrices
indices = np.arange(sample_size) + 1
design = dia_matrix(np.diag(1 / (np.sqrt(indices))))

# Create signals
signal_supersmooth = 5 * np.exp(-0.1 * indices)

# Create observations for the three different signals


simulation = es.SimulationWrapper(design=design, true_signal=signal_supersmooth,
                     true_noise_level=noise_level, max_iterations=max_iter)
results = simulation.run_simulation()



