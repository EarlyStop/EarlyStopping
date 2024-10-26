################################################################################
#             Reproduction example for truncated SVD estimation                #
################################################################################

# Imports
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import EarlyStopping as es
import pandas as pd

sample_size  = 10000
indices      = np.arange(sample_size) + 1
eigenvalues  = indices**(-0.5)
design       = np.diag(eigenvalues)

true_signal_supersmooth = 5    * np.exp(-0.1 * indices)
true_signal_smooth      = 5000 * np.abs(np.sin(0.01  * indices))  * indices**(-1.6)
true_signal_rough       = 250  * np.abs(np.sin(0.002 * indices))  * indices**(-0.8)


response_noiseless_rough = eigenvalues * true_signal_rough
response_noiseless_smooth = eigenvalues * true_signal_smooth
response_noiseless_supersmooth = eigenvalues * true_signal_supersmooth

# design_smooth, response_noiseless_smooth, true_signal_smooth = es.SimulationData.diagonal_data(sample_size=10000, type = 'smooth')
# design_supersmooth, response_noiseless_supersmooth, true_signal_supersmooth = es.SimulationData.diagonal_data(sample_size=10000, type = 'supersmooth')
# design_rough, response_noiseless_rough, true_signal_rough = es.SimulationData.diagonal_data(sample_size=10000, type = 'rough')

parameters_smooth = es.SimulationParameters(
    design=design,
    true_signal=true_signal_smooth,
    true_noise_level=0.01,
    max_iteration=1000,
    monte_carlo_runs=10,
    cores=12
)

parameters_supersmooth = es.SimulationParameters(
    design=design,
    true_signal=true_signal_supersmooth,
    true_noise_level=0.01,
    max_iteration=1000,
    monte_carlo_runs=10, #500
    cores=12
)

parameters_rough = es.SimulationParameters(
    design=design,
    true_signal=true_signal_rough,
    true_noise_level=0.01,
    max_iteration=3000,
    monte_carlo_runs=10,
    cores=12
)

simulation_smooth = es.SimulationWrapper(**parameters_smooth.__dict__)
simulation_supersmooth = es.SimulationWrapper(**parameters_supersmooth.__dict__)
simulation_rough = es.SimulationWrapper(**parameters_rough.__dict__)

results_smooth = simulation_smooth.run_simulation_truncated_svd(diagonal=True, data_set_name="truncated_svd_simulation_smooth")
results_supersmooth = simulation_supersmooth.run_simulation_truncated_svd(diagonal=True, data_set_name="truncated_svd_simulation_supersmooth")
results_rough = simulation_rough.run_simulation_truncated_svd(diagonal=True, data_set_name="truncated_svd_simulation_rough")



















