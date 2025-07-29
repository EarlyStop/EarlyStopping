################################################################################
#             Reproduction study for L2-boost                                  #
################################################################################

# Imports
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import EarlyStopping as es
import pandas as pd
import os

np.random.seed(42)


# Signals and design
# ------------------------------------------------------------------------------
# To simulate some data we consider the signals from `Stankewitz (2022) <https://arxiv.org/abs/2210.07850v1>`_.
sample_size = 1000
para_size = 1000

cov = np.identity(para_size)
sigma = np.sqrt(1)
design = np.random.multivariate_normal(np.zeros(para_size), cov, sample_size)

# Gamma-sparse signals
beta_3 = 1 / (1 + np.arange(para_size)) ** 3
beta_3 = 10 * beta_3 / np.sum(np.abs(beta_3))

beta_2 = 1 / (1 + np.arange(para_size)) ** 2
beta_2 = 10 * beta_2 / np.sum(np.abs(beta_2))

beta_1 = 1 / (1 + np.arange(para_size))
beta_1 = 10 * beta_1 / np.sum(np.abs(beta_1))

# S-sparse signals
beta_15 = np.zeros(para_size)
beta_15[0:15] = 1
beta_15 = 10 * beta_15 / np.sum(np.abs(beta_15))

beta_60 = np.zeros(para_size)
beta_60[0:20] = 1
beta_60[20:40] = 0.5
beta_60[40:60] = 0.25
beta_60 = 10 * beta_60 / np.sum(np.abs(beta_60))

beta_90 = np.zeros(para_size)
beta_90[0:30] = 1
beta_90[30:60] = 0.5
beta_90[60:90] = 0.25
beta_90 = 10 * beta_90 / np.sum(np.abs(beta_90))

# Setting the simulation parameters
# ------------------------------------------------------------------------------
parameters_beta_3 = es.SimulationParameters(
    design=design, true_signal=beta_3, true_noise_level=1, monte_carlo_runs=100, cores=12
)
parameters_beta_2 = es.SimulationParameters(
    design=design, true_signal=beta_2, true_noise_level=1, monte_carlo_runs=100, cores=12
)
parameters_beta_1 = es.SimulationParameters(
    design=design, true_signal=beta_1, true_noise_level=1, monte_carlo_runs=100, cores=12
)

parameters_beta_15 = es.SimulationParameters(
    design=design, true_signal=beta_15, true_noise_level=1, monte_carlo_runs=100, cores=12
)
parameters_beta_60 = es.SimulationParameters(
    design=design, true_signal=beta_60, true_noise_level=1, monte_carlo_runs=100, cores=12
)
parameters_beta_90 = es.SimulationParameters(
    design=design, true_signal=beta_90, true_noise_level=1, monte_carlo_runs=100, cores=12
)

# Initialize simulation classes and run sims.
# -----------------------------------------------------------------------------
# Use **-notation for auto-extracting.
simulation_beta_3 = es.SimulationWrapper(**parameters_beta_3.__dict__)
simulation_beta_2 = es.SimulationWrapper(**parameters_beta_2.__dict__)
simulation_beta_1 = es.SimulationWrapper(**parameters_beta_1.__dict__)

simulation_beta_15 = es.SimulationWrapper(**parameters_beta_15.__dict__)
simulation_beta_60 = es.SimulationWrapper(**parameters_beta_60.__dict__)
simulation_beta_90 = es.SimulationWrapper(**parameters_beta_90.__dict__)

results_beta_3 = simulation_beta_3.run_simulation_L2_boost(max_iteration=50, data_set_name="L2_boost_beta_15")
results_beta_2 = simulation_beta_2.run_simulation_L2_boost(max_iteration=100, data_set_name="L2_boost_beta_60")
results_beta_1 = simulation_beta_1.run_simulation_L2_boost(max_iteration=200, data_set_name="L2_boost_beta_90")

results_beta_15 = simulation_beta_15.run_simulation_L2_boost(max_iteration=50, data_set_name="L2_boost_beta_15")
results_beta_60 = simulation_beta_60.run_simulation_L2_boost(max_iteration=100, data_set_name="L2_boost_beta_60")
results_beta_90 = simulation_beta_90.run_simulation_L2_boost(max_iteration=200, data_set_name="L2_boost_beta_90")


# Figures
# ------------------------------------------------------------------------------


def create_custom_boxplot(data, labels, y_lim_lower, y_lim_upper, fig_dir, name):
    # Create a boxplot for the given data
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels=labels)

    # Define custom colors
    colors = ["blue", "purple", "#CCCC00", "green", "red", "orange"]

    # Set colors for each box
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)  # Set the border thickness

    # Making whiskers, caps, and medians thicker and grey
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.5)
    for cap in bp["caps"]:
        cap.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_linewidth(1.5)

    # Add a horizontal line at y=1
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1.5)

    # Enable gridlines
    plt.grid(True)

    # Set y-axis limits (can adjust based on your data)
    plt.ylim(y_lim_lower, y_lim_upper)

    # Customize tick labels and layout
    plt.tick_params(axis="both", which="major", labelsize=14)

    # Add sub-label "weak" beneath the first three x-axis labels
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)

    plt.savefig(os.path.join(fig_dir, f"boxplot_{name}.png"), bbox_inches="tight", dpi=300)

    plt.tight_layout()  # Adjust layout

    # Show the plot
    plt.show()


# Relative efficiency for discrepany stop
relative_efficiency_discrepancy_beta_3 = results_beta_3["relative_efficiency_discrepancy"]
relative_efficiency_discrepancy_beta_2 = results_beta_2["relative_efficiency_discrepancy"]
relative_efficiency_discrepancy_beta_1 = results_beta_1["relative_efficiency_discrepancy"]

relative_efficiency_discrepancy_beta_15 = results_beta_15["relative_efficiency_discrepancy"]
relative_efficiency_discrepancy_beta_60 = results_beta_60["relative_efficiency_discrepancy"]
relative_efficiency_discrepancy_beta_90 = results_beta_90["relative_efficiency_discrepancy"]

data = [
    relative_efficiency_discrepancy_beta_3,
    relative_efficiency_discrepancy_beta_2,
    relative_efficiency_discrepancy_beta_1,
    relative_efficiency_discrepancy_beta_15,
    relative_efficiency_discrepancy_beta_60,
    relative_efficiency_discrepancy_beta_90,
]

labels = ["beta_3", "beta_2", "beta_1", "beta_15", "beta_60", "beta_90"]
fig_dir = ""
create_custom_boxplot(
    data, labels, y_lim_lower=0.1, y_lim_upper=1.1, fig_dir=fig_dir, name="L2_boost_relative_efficiencies_discrepancy"
)

# Relative efficiency for residual ratio stop
relative_efficiency_residual_ratio_beta_3 = results_beta_3["relative_efficiency_residual_ratio"]
relative_efficiency_residual_ratio_beta_2 = results_beta_2["relative_efficiency_residual_ratio"]
relative_efficiency_residual_ratio_beta_1 = results_beta_1["relative_efficiency_residual_ratio"]

relative_efficiency_residual_ratio_beta_15 = results_beta_15["relative_efficiency_residual_ratio"]
relative_efficiency_residual_ratio_beta_60 = results_beta_60["relative_efficiency_residual_ratio"]
relative_efficiency_residual_ratio_beta_90 = results_beta_90["relative_efficiency_residual_ratio"]

data = [
    relative_efficiency_residual_ratio_beta_3,
    relative_efficiency_residual_ratio_beta_2,
    relative_efficiency_residual_ratio_beta_1,
    relative_efficiency_residual_ratio_beta_15,
    relative_efficiency_residual_ratio_beta_60,
    relative_efficiency_residual_ratio_beta_90,
]

labels = ["beta_3", "beta_2", "beta_1", "beta_15", "beta_60", "beta_90"]
fig_dir = ""
create_custom_boxplot(
    data,
    labels,
    y_lim_lower=0.1,
    y_lim_upper=1.1,
    fig_dir=fig_dir,
    name="L2_boost_relative_efficiencies_residual_ratio",
)

# Quantities for AIC
# Relative efficiency for residual ratio stop
relative_efficiency_aic_beta_3 = results_beta_3["relative_efficiency_aic"]
relative_efficiency_two_step_discrepancy_stop_beta_3 = results_beta_3["relative_efficiency_two_step_discrepancy_stop"]
relative_efficiency_two_step_residual_ratio_stop_beta_3 = results_beta_3[
    "relative_efficiency_two_step_residual_ratio_stop"
]


relative_efficiency_aic_beta_2 = results_beta_2["relative_efficiency_aic"]
relative_efficiency_two_step_discrepancy_stop_beta_2 = results_beta_2["relative_efficiency_two_step_discrepancy_stop"]
relative_efficiency_two_step_residual_ratio_stop_beta_2 = results_beta_2[
    "relative_efficiency_two_step_residual_ratio_stop"
]


relative_efficiency_aic_beta_1 = results_beta_1["relative_efficiency_aic"]
relative_efficiency_two_step_discrepancy_stop_beta_1 = results_beta_1["relative_efficiency_two_step_discrepancy_stop"]
relative_efficiency_two_step_residual_ratio_stop_beta_1 = results_beta_1[
    "relative_efficiency_two_step_residual_ratio_stop"
]


aic_beta_60 = results_beta_60["aic_time"]

dp_time_beta_60 = results_beta_60["discrepancy_time"]

relative_efficiency_aic_beta_15 = results_beta_15["relative_efficiency_aic"]
relative_efficiency_two_step_discrepancy_stop_beta_15 = results_beta_15[
    "relative_efficiency_two_step_discrepancy_stop"
]
relative_efficiency_two_step_residual_ratio_stop_beta_15 = results_beta_15[
    "relative_efficiency_two_step_residual_ratio_stop"
]


relative_efficiency_aic_beta_60 = results_beta_60["relative_efficiency_aic"]
relative_efficiency_two_step_discrepancy_stop_beta_60 = results_beta_60[
    "relative_efficiency_two_step_discrepancy_stop"
]
relative_efficiency_two_step_residual_ratio_stop_beta_60 = results_beta_60[
    "relative_efficiency_two_step_residual_ratio_stop"
]


relative_efficiency_aic_beta_90 = results_beta_90["relative_efficiency_aic"]
relative_efficiency_two_step_discrepancy_stop_beta_90 = results_beta_90[
    "relative_efficiency_two_step_discrepancy_stop"
]
relative_efficiency_two_step_residual_ratio_stop_beta_90 = results_beta_90[
    "relative_efficiency_two_step_residual_ratio_stop"
]


data_aic = [
    relative_efficiency_aic_beta_3,
    relative_efficiency_aic_beta_2,
    relative_efficiency_aic_beta_1,
    relative_efficiency_aic_beta_15,
    relative_efficiency_aic_beta_60,
    relative_efficiency_aic_beta_90,
]

data_two_step_discrepancy_stop = [
    relative_efficiency_two_step_discrepancy_stop_beta_3,
    relative_efficiency_two_step_discrepancy_stop_beta_2,
    relative_efficiency_two_step_discrepancy_stop_beta_1,
    relative_efficiency_two_step_discrepancy_stop_beta_15,
    relative_efficiency_two_step_discrepancy_stop_beta_60,
    relative_efficiency_two_step_discrepancy_stop_beta_90,
]

data_two_step_residual_ratio_stop = [
    relative_efficiency_two_step_residual_ratio_stop_beta_3,
    relative_efficiency_two_step_residual_ratio_stop_beta_2,
    relative_efficiency_two_step_residual_ratio_stop_beta_1,
    relative_efficiency_two_step_residual_ratio_stop_beta_15,
    relative_efficiency_two_step_residual_ratio_stop_beta_60,
    relative_efficiency_two_step_residual_ratio_stop_beta_90,
]


labels = ["beta_3", "beta_2", "beta_1", "beta_15", "beta_60", "beta_90"]
fig_dir = ""
# create_custom_boxplot(data_aic, labels, y_lim_lower = 0, y_lim_upper=1.1, fig_dir=fig_dir, name='L2_boost_relative_efficiencies_aic')
create_custom_boxplot(
    data_two_step_discrepancy_stop,
    labels,
    y_lim_lower=0,
    y_lim_upper=1.1,
    fig_dir=fig_dir,
    name="L2_boost_relative_efficiency_two_step_discrepancy_stop",
)
create_custom_boxplot(
    data_two_step_residual_ratio_stop,
    labels,
    y_lim_lower=0,
    y_lim_upper=1.1,
    fig_dir=fig_dir,
    name="L2_boost_relative_efficiency_two_step_residual_ratio_stop",
)
