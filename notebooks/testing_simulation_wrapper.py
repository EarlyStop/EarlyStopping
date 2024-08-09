for name in list(globals()):
    if not name.startswith("_"):
        del globals()[name]

import numpy as np
import importlib
import EarlyStopping as es
from scipy.sparse import dia_matrix

importlib.reload(es)


design, response_noiseless, true_signal = es.SimulationData.diagonal_data(sample_size=1000, type = 'supersmooth')
#design, response_noiseless, true_signal = es.SimulationData.gravity(sample_size=100)
#design, response_noiseless, true_signal = es.SimulationData.heat(sample_size=100)
#design, response_noiseless, true_signal = es.SimulationData.deriv2(sample_size=100)
#design, response_noiseless, true_signal = es.SimulationData.phillips(sample_size=100)

parameters = es.SimulationParameters(
    design=design,
    true_signal=true_signal,
    true_noise_level=0.01,
    max_iterations=1000,
    monte_carlo_runs=5,
    cores=5
)

simulation = es.SimulationWrapper(**parameters.__dict__)
results = simulation.run_simulation()
