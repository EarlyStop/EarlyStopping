"""
Simulation study for conjugate gradients
========================================

We conduct in the following a simulation study illustrating the conjugate gradients algorithm.
"""

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
SAMPLE_SIZE = 10000
PARAMETER_SIZE = SAMPLE_SIZE
MAXIMAL_ITERATION = SAMPLE_SIZE
NOISE_LEVEL = 0.01
CRITICAL_VALUE = SAMPLE_SIZE * (NOISE_LEVEL**2)

# Create diagonal design matrices
indices = np.arange(SAMPLE_SIZE) + 1
design_matrix = dia_matrix(np.diag(1 / (np.sqrt(indices))))

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
NUMBER_RUNS = 2  # set to 1000 for final simulations

# Create observations for the three different signals
noise = np.random.normal(0, NOISE_LEVEL, (SAMPLE_SIZE, NUMBER_RUNS))
observation_supersmooth = noise + (design_matrix @ signal_supersmooth)[:, None]
observation_smooth = noise + (design_matrix @ signal_smooth)[:, None]
observation_rough = noise + (design_matrix @ signal_rough)[:, None]

# %%
# We choose to interpolate linearly between the conjugate gradient estimates at integer iteration indices and create the models.

# Set interpolation boolean
INTERPOLATION_BOOLEAN = True

# Create models
models_supersmooth = [
    es.ConjugateGradients(
        design_matrix,
        observation_supersmooth[:, i],
        true_signal=signal_supersmooth,
        true_noise_level=NOISE_LEVEL,
        interpolation=INTERPOLATION_BOOLEAN,
    )
    for i in range(NUMBER_RUNS)
]
models_smooth = [
    es.ConjugateGradients(
        design_matrix,
        observation_smooth[:, i],
        true_signal=signal_smooth,
        true_noise_level=NOISE_LEVEL,
        interpolation=INTERPOLATION_BOOLEAN,
    )
    for i in range(NUMBER_RUNS)
]
models_rough = [
    es.ConjugateGradients(
        design_matrix,
        observation_rough[:, i],
        true_signal=signal_rough,
        true_noise_level=NOISE_LEVEL,
        interpolation=INTERPOLATION_BOOLEAN,
    )
    for i in range(NUMBER_RUNS)
]

# %%
# We calculate the early stopping index, the conjugate gradient estimate at the early stopping index and the squared residual norms along the whole iteration path (until MAXIMAL_ITERATION) for the three signals.

for run in range(NUMBER_RUNS):
    start_time = time.time()
    models_supersmooth[run].conjugate_gradients_gather_all(MAXIMAL_ITERATION)
    models_smooth[run].conjugate_gradients_gather_all(MAXIMAL_ITERATION)
    models_rough[run].conjugate_gradients_gather_all(MAXIMAL_ITERATION)
    end_time = time.time()
    print(f"The {run+1}-th Monte Carlo step took {end_time - start_time} seconds.")
    print(f"Supersmooth early stopping index: {models_supersmooth[run].early_stopping_index}")
    print(f"Smooth early stopping index: {models_smooth[run].early_stopping_index}")
    print(f"Rough early stopping index: {models_rough[run].early_stopping_index}")

# %%
# Plot of squared residual norms and empirical error terms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We plot for the first Monte Carlo run the squared residual norms along the whole iteration path and the corresponding weak and strong empirical error terms,
# i.e. the empirical prediction and reconstruction errors, for the supersmooth signal. The critical value is denoted by :math:`\kappa` and the early stopping index by :math:`\tau`.
# If we choose to interpolate, i.e. INTERPOLSTION_BOOLEAN is set to
# true, we need to interpolate between the residual polynomials and therefore between the estimators.

# Set gridsize for the x-axis of the plots
GRIDSIZE = 0.01

# Calculate interpolated squared residual norms and interpolated strong and weak empirical errors for the first Monte Carlo run
if INTERPOLATION_BOOLEAN:
    grid = np.arange(0, MAXIMAL_ITERATION + GRIDSIZE, GRIDSIZE)
    residuals_supersmooth = models_supersmooth[0].calculate_interpolated_residual(index=grid)
    strong_empirical_errors_supersmooth = models_supersmooth[0].calculate_interpolated_strong_empirical_error(
        index=grid
    )
    weak_empirical_errors_supersmooth = models_supersmooth[0].calculate_interpolated_weak_empirical_error(index=grid)
else:
    grid = np.arange(0, MAXIMAL_ITERATION + 1)
    residuals_supersmooth = models_supersmooth[0].residuals
    strong_empirical_errors_supersmooth = models_supersmooth[0].strong_empirical_errors
    weak_empirical_errors_supersmooth = models_supersmooth[0].weak_empirical_errors

# Plot
plot_residuals_empirical_errors = plt.figure()
plt.plot(grid, residuals_supersmooth, label="squared residual norm", color="green")
plt.plot(grid, strong_empirical_errors_supersmooth, label="strong empirical error", color="blue")
plt.plot(grid, weak_empirical_errors_supersmooth, label="weak empirical error", color="orange")
plt.axvline(x=models_supersmooth[0].early_stopping_index, color="grey", linestyle="--")
plt.axhline(y=models_supersmooth[0].critical_value, color="grey", linestyle="--")
plt.xlim([0, 14])
plt.ylim([0, 2])
plt.xlabel("Iteration index")
plt.xticks(list(plt.xticks()[0]) + [models_supersmooth[0].early_stopping_index], list(plt.xticks()[1]) + ["$\\tau$"])
plt.yticks(
    list(plt.yticks()[0]) + [models_supersmooth[0].critical_value], list(plt.yticks()[1]) + ["$\\kappa = D \\delta^2$"]
)
plt.legend()
plt.show()

# %%
# Boxplots of the strong and weak empirical losses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We make boxplots comparing the performance of the conjugate gradient estimator at the early stopping index for the three different signals in terms of its strong and weak empirical error.

if INTERPOLATION_BOOLEAN:
    strong_empirical_errors_supersmooth_Monte_Carlo = [
        float(model.calculate_interpolated_strong_empirical_error(index=model.early_stopping_index).item())
        for model in models_supersmooth
    ]
    strong_empirical_errors_smooth_Monte_Carlo = [
        float(model.calculate_interpolated_strong_empirical_error(index=model.early_stopping_index).item())
        for model in models_smooth
    ]
    strong_empirical_errors_rough_Monte_Carlo = [
        float(model.calculate_interpolated_strong_empirical_error(index=model.early_stopping_index).item())
        for model in models_rough
    ]
    weak_empirical_errors_supersmooth_Monte_Carlo = [
        float(model.calculate_interpolated_weak_empirical_error(index=model.early_stopping_index).item())
        for model in models_supersmooth
    ]
    weak_empirical_errors_smooth_Monte_Carlo = [
        float(model.calculate_interpolated_weak_empirical_error(index=model.early_stopping_index).item())
        for model in models_smooth
    ]
    weak_empirical_errors_rough_Monte_Carlo = [
        float(model.calculate_interpolated_weak_empirical_error(index=model.early_stopping_index).item())
        for model in models_rough
    ]
else:
    strong_empirical_errors_supersmooth_Monte_Carlo = [
        models_supersmooth[i].strong_empirical_errors[models_supersmooth[i].early_stopping_index]
        for i in range(NUMBER_RUNS)
    ]
    strong_empirical_errors_smooth_Monte_Carlo = [
        models_smooth[i].strong_empirical_errors[models_smooth[i].early_stopping_index] for i in range(NUMBER_RUNS)
    ]
    strong_empirical_errors_rough_Monte_Carlo = [
        models_rough[i].strong_empirical_errors[models_rough[i].early_stopping_index] for i in range(NUMBER_RUNS)
    ]
    weak_empirical_errors_supersmooth_Monte_Carlo = [
        models_supersmooth[i].weak_empirical_errors[models_supersmooth[i].early_stopping_index]
        for i in range(NUMBER_RUNS)
    ]
    weak_empirical_errors_smooth_Monte_Carlo = [
        models_smooth[i].weak_empirical_errors[models_smooth[i].early_stopping_index] for i in range(NUMBER_RUNS)
    ]
    weak_empirical_errors_rough_Monte_Carlo = [
        models_rough[i].weak_empirical_errors[models_rough[i].early_stopping_index] for i in range(NUMBER_RUNS)
    ]

# Plot of the strong empirical errors
strong_empirical_errors_Monte_Carlo = pd.DataFrame(
    {
        "algorithm": ["conjugate gradients"] * NUMBER_RUNS,
        "supersmooth": strong_empirical_errors_supersmooth_Monte_Carlo,
        "smooth": strong_empirical_errors_smooth_Monte_Carlo,
        "rough": strong_empirical_errors_rough_Monte_Carlo,
    }
)
strong_empirical_errors_Monte_Carlo = pd.melt(
    strong_empirical_errors_Monte_Carlo, id_vars="algorithm", value_vars=["supersmooth", "smooth", "rough"]
)
plt.figure()
strong_empirical_errors_boxplot = sns.boxplot(
    x="variable", y="value", data=strong_empirical_errors_Monte_Carlo, width=0.4
)
strong_empirical_errors_boxplot.set(xlabel="Signal", ylabel="Strong empirical error at $\\tau$")
plt.show()

# Plot of the weak empirical errors
weak_empirical_errors_Monte_Carlo = pd.DataFrame(
    {
        "algorithm": ["conjugate gradients"] * NUMBER_RUNS,
        "supersmooth": weak_empirical_errors_supersmooth_Monte_Carlo,
        "smooth": weak_empirical_errors_smooth_Monte_Carlo,
        "rough": weak_empirical_errors_rough_Monte_Carlo,
    }
)
weak_empirical_errors_Monte_Carlo = pd.melt(
    weak_empirical_errors_Monte_Carlo, id_vars="algorithm", value_vars=["supersmooth", "smooth", "rough"]
)
plt.figure()
weak_empirical_errors_boxplot = sns.boxplot(x="variable", y="value", data=weak_empirical_errors_Monte_Carlo, width=0.4)
weak_empirical_errors_boxplot.set(xlabel="Signal", ylabel="Weak empirical error at $\\tau$")
plt.show()
