"""
Usage of the SimulationWrapper class
============================================

We illustrate how the SimulationWrapper class can be used to simplify
the process of creating a Montecarlo simmulation for analying the performance of the Landweber method.
"""

import numpy as np
import seaborn as sns
import EarlyStopping as es
import pandas as pd

np.random.seed(42)
sns.set_theme()

# %%
# Gererate synthetic data using the SimulationData class.
# Other options besides diagonal_data include 'gravity', 'heat', 'deriv2', 'phillips'
design_smooth, response_noiseless_smooth, true_signal_smooth = es.SimulationData.diagonal_data(
    sample_size=1000, type="smooth"
)

# %%
# Setup and verify the Simulation paramters for the Simulation
parameters_smooth = es.SimulationParameters(
    design=design_smooth,
    true_signal=true_signal_smooth,
    true_noise_level=0.01,
    monte_carlo_runs=3,
    cores=3,
)

# %%
# Create a SimulationWrapper object based on the SimulationParameters and run the simulation based on the desired estimation method.
# The data_set_name parameter is optional and is used to save the results of the simulation.
# Since the parameter is not specified, the results will not be saved
# and are simply returned as a pd.DataFrame.
simulation_smooth = es.SimulationWrapper(**parameters_smooth.__dict__)
simmulation_results = simulation_smooth.run_simulation_landweber(max_iteration=1000)

pd.set_option("display.max_rows", None, "display.max_columns", None)  # Display all rows and columns of the DataFrame
print(simmulation_results)
