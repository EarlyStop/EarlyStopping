"""
This is a comparison of Landweber and conjugate gradients
=========================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import EarlyStopping as es
from scipy.sparse import dia_matrix
import timeit

np.random.seed(42)
plt.rcParams.update({"font.size": 20})
print("The seed is 42.")


import time
import numpy as np
import pandas as pd
from scipy.sparse import dia_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es

sns.set_theme(style="ticks")
np.random.seed(42)

# %%
# Simulation Setting
# ------------------
# To make our results comparable, we use the same simulation setting as `Blanchard et al. (2018) <https://doi.org/10.1137/17M1154096>`_ and `Stankewitz (2020) <https://doi.org/10.1214/20-EJS1747>`_.

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
signal_smooth = 5000 * np.abs(np.sin(0.01 * indices)) * indices ** (-1.6)
signal_rough = 250 * np.abs(np.sin(0.002 * indices)) * indices ** (-0.8)

# %%
# We plot the SVD coefficients of the three signals.

plt.plot(indices, signal_supersmooth, label="supersmooth signal")
plt.plot(indices, signal_smooth, label="smooth signal")
plt.plot(indices, signal_rough, label="rough signal")
plt.ylabel("Signal")
plt.xlabel("Index")
plt.ylim([-0.05, 1.6])
plt.legend()
plt.show()

# %%
# Monte Carlo Study
# -----------------
# We simulate NUMBER_RUNS realisations of the Gaussian sequence space model.

# Specify number of Monte Carlo runs
NUMBER_RUNS = 10

# Set computation threshold
computation_threshold = 0

# Create observations for the three different signals
noise = np.random.normal(0, noise_level, (sample_size, NUMBER_RUNS))
observation_supersmooth = noise + (design @ signal_supersmooth)[:, None]
observation_smooth = noise + (design @ signal_smooth)[:, None]
observation_rough = noise + (design @ signal_rough)[:, None]

# %%
# We choose to interpolate linearly between the conjugate gradient estimates at integer iteration indices and create the models.

# Set interpolation boolean
interpolation_boolean = False

supersmooth_weak_empirical_error_cg = np.zeros(NUMBER_RUNS)
smooth_weak_empirical_error_cg = np.zeros(NUMBER_RUNS)
rough_weak_empirical_error_cg = np.zeros(NUMBER_RUNS)

supersmooth_weak_empirical_error_landweber = np.zeros(NUMBER_RUNS)
smooth_weak_empirical_error_landweber = np.zeros(NUMBER_RUNS)
rough_weak_empirical_error_landweber = np.zeros(NUMBER_RUNS)

for i in range(NUMBER_RUNS):
    # Create models
    models_supersmooth_cg = es.ConjugateGradients(
        design,
        observation_supersmooth[:, i],
        true_signal=signal_supersmooth,
        true_noise_level=noise_level,
        interpolation=interpolation_boolean,
        computation_threshold=computation_threshold,
    )

    models_smooth_cg = es.ConjugateGradients(
        design,
        observation_smooth[:, i],
        true_signal=signal_smooth,
        true_noise_level=noise_level,
        interpolation=interpolation_boolean,
        computation_threshold=computation_threshold,
    )

    models_rough_cg = es.ConjugateGradients(
        design,
        observation_rough[:, i],
        true_signal=signal_rough,
        true_noise_level=noise_level,
        interpolation=interpolation_boolean,
        computation_threshold=computation_threshold,
    )

    models_supersmooth_landweber = es.Landweber(
        design, observation_supersmooth[:, i], true_signal=signal_supersmooth, true_noise_level=noise_level
    )

    models_smooth_landweber = es.Landweber(
        design, observation_smooth[:, i], true_signal=signal_smooth, true_noise_level=noise_level
    )

    models_rough_landweber = es.Landweber(
        design, observation_rough[:, i], true_signal=signal_rough, true_noise_level=noise_level
    )

    models_supersmooth_cg.gather_all(max_iter)
    models_smooth_cg.gather_all(max_iter)
    models_rough_cg.gather_all(max_iter)

    models_supersmooth_landweber.iterate(max_iter)
    models_smooth_landweber.iterate(max_iter)
    models_rough_landweber.iterate(max_iter)

    supersmooth_weak_empirical_error_landweber[i] = np.mean(
        ((models_supersmooth_landweber.landweber_estimate_collect - models_supersmooth_landweber.true_signal)) ** 2
    )

    smooth_weak_empirical_error_landweber[i] = np.mean(
        ((models_smooth_landweber.landweber_estimate_collect - models_smooth_landweber.true_signal)) ** 2
    )

    rough_weak_empirical_error_landweber[i] = np.mean(
        ((models_rough_landweber.landweber_estimate_collect - models_rough_landweber.true_signal)) ** 2
    )

    supersmooth_weak_empirical_error_cg[i] = models_supersmooth_cg.weak_empirical_errors[
        models_supersmooth_cg.early_stopping_index
    ]

    smooth_weak_empirical_error_cg[i] = models_smooth_cg.weak_empirical_errors[models_smooth_cg.early_stopping_index]
    rough_weak_empirical_error_cg[i] = models_rough_cg.weak_empirical_errors[models_rough_cg.early_stopping_index]


# Plot of the weak empirical errors
weak_empirical_errors_Monte_Carlo = pd.DataFrame(
    {
        "algorithm": ["conjugate gradients"] * NUMBER_RUNS,
        "supersmooth_cg": supersmooth_weak_empirical_error_cg,
        "smooth_cg": smooth_weak_empirical_error_cg,
        "rough_cg": rough_weak_empirical_error_cg,
    }
)

weak_empirical_errors_Monte_Carlo_landweber = pd.DataFrame(
    {
        "algorithm": ["landweber"] * NUMBER_RUNS,
        "supersmooth_landweber": supersmooth_weak_empirical_error_landweber,
        "smooth_landweber": smooth_weak_empirical_error_landweber,
        "rough_landweber": rough_weak_empirical_error_landweber,
    }
)

weak_empirical_errors_Monte_Carlo = pd.concat(
    [weak_empirical_errors_Monte_Carlo, weak_empirical_errors_Monte_Carlo_landweber]
)


# weak_empirical_errors_Monte_Carlo = pd.DataFrame(
#     {
#         "algorithm": ["conjugate gradients"] * NUMBER_RUNS,
#         "supersmooth": supersmooth_weak_empirical_error_cg,
#         "smooth": smooth_weak_empirical_error_cg,
#         "rough": rough_weak_empirical_error_cg,
#     }
# )

weak_empirical_errors_Monte_Carlo = pd.melt(
    weak_empirical_errors_Monte_Carlo,
    id_vars="algorithm",
    value_vars=[
        "supersmooth_cg",
        "smooth_cg",
        "rough_cg",
        "supersmooth_landweber",
        "smooth_landweber",
        "rough_landweber",
    ],
)

plt.figure()
weak_empirical_errors_boxplot = sns.boxplot(x="variable", y="value", data=weak_empirical_errors_Monte_Carlo, width=0.4)
weak_empirical_errors_boxplot.set(xlabel="Signal", ylabel="Weak empirical error at $\\tau$")
plt.show()
