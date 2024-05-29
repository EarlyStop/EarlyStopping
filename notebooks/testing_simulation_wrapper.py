for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]



import numpy as np
import importlib
import EarlyStopping as es
from scipy.sparse import dia_matrix

importlib.reload(es)

sample_size = 1000
indices = np.arange(1000) + 1

parameters = es.SimulationParameters(
    design=dia_matrix(np.diag(1 / np.sqrt(indices))),
    true_signal=5 * np.exp(-0.1 * indices),
    true_noise_level=0.01,
    max_iterations=1000)

simulation = es.SimulationWrapper(**parameters.__dict__)

results = simulation.run_simulation()



