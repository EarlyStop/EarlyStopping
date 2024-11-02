################################################################################
#             Reproduction study for truncated SVD estimation                  #
################################################################################

# Imports
# ------------------------------------------------------------------------------
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import seaborn           as sns   
sns.set_theme()
import EarlyStopping     as es


# Signals and design
# ------------------------------------------------------------------------------
# From G. Blanchard, M. Hoffmann, M. Reiß. "Early stopping for statistical inverse problems via
# truncated SVD estimation". In: Electronic Journal of Statistics 12(2): 3204-3231 (2018).
sample_size  = 10000
indices      = np.arange(sample_size) + 1
eigenvalues  = indices**(-0.5)
design       = np.diag(eigenvalues)

true_signal_supersmooth = 5    * np.exp(-0.1 * indices)
true_signal_smooth      = 5000 * np.abs(np.sin(0.01  * indices))  * indices**(-1.6)
true_signal_rough       = 250  * np.abs(np.sin(0.002 * indices))  * indices**(-0.8)


# Setting the simulation parameters
# ------------------------------------------------------------------------------
# TODO-BS-2024-11-02: Rework the truncated svd class to take sparse diagonal matrices as inputs and
# rework this to be used with the SimulationData class from the wrapper, e.g. like
# response_noiseless_supersmooth = eigenvalues * true_signal_supersmooth
# design_supersmooth, \
# response_noiseless_supersmooth, \
# true_signal_supersmooth         = es.SimulationData.diagonal_data(sample_size = 10000, type = 'supersmooth')
parameters_supersmooth = es.SimulationParameters(
    design=design,
    true_signal=true_signal_supersmooth,
    true_noise_level=0.01,
    max_iteration=1000,
    monte_carlo_runs=1000,
    cores=12
)
parameters_smooth = es.SimulationParameters(
    design=design,
    true_signal=true_signal_smooth,
    true_noise_level=0.01,
    max_iteration=1000,
    monte_carlo_runs=1000,
    cores=12
)
parameters_rough = es.SimulationParameters(
    design=design,
    true_signal=true_signal_rough,
    true_noise_level=0.01,
    max_iteration=3000,
    monte_carlo_runs=1000,
    cores=12
)

# Initialize simulation classes and run sims.
# -----------------------------------------------------------------------------
# Use **-notation for auto-extracting.
simulation_supersmooth = es.SimulationWrapper(**parameters_supersmooth.__dict__)
simulation_smooth      = es.SimulationWrapper(**parameters_smooth.__dict__)
simulation_rough       = es.SimulationWrapper(**parameters_rough.__dict__)

results_supersmooth = simulation_supersmooth.run_simulation_truncated_svd(
                        diagonal=True, data_set_name="truncated_svd_simulation_supersmooth"
                      )
results_smooth      = simulation_smooth.run_simulation_truncated_svd(
                        diagonal=True, data_set_name="truncated_svd_simulation_smooth"
                      )
results_rough       = simulation_rough.run_simulation_truncated_svd(
                       diagonal=True, data_set_name="truncated_svd_simulation_rough"
                      )


# Figures figures
# ------------------------------------------------------------------------------
# Strong relative efficiency
supersmooth_strong_relative_efficiency = results_supersmooth["strong_relative_efficiency"]
smooth_strong_relative_efficiency      = results_smooth["strong_relative_efficiency"]
rough_strong_relative_efficiency       = results_rough["strong_relative_efficiency"]

data = {'supersmooth': supersmooth_strong_relative_efficiency,
        'smooth':      smooth_strong_relative_efficiency,
        'rough':       rough_strong_relative_efficiency}
df   = pd.DataFrame(data)   

fig = plt.figure(figsize = (7,5))
fig.patch.set_facecolor('white')
ax  = sns.boxplot(data = df)
ax.set_ylim(0, 1.15)
ax.set_title("Relative efficiencies in strong risk")

(df["supersmooth"] > 1)
# TODO-BS-2024-11-02: We don't get values > 1 which contradicts the simulation from the paper.
# Relative efficiency differently defined / loss at stopping time.



















