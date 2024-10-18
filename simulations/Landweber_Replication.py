for name in list(globals()):
    if not name.startswith("_"):
        del globals()[name]

import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
import os
import pandas as pd

importlib.reload(es)

design_smooth, response_noiseless_smooth, true_signal_smooth = es.SimulationData.diagonal_data(sample_size=10000, type = 'smooth')
design_supersmooth, response_noiseless_supersmooth, true_signal_supersmooth = es.SimulationData.diagonal_data(sample_size=10000, type = 'supersmooth')
design_rough, response_noiseless_rough, true_signal_rough = es.SimulationData.diagonal_data(sample_size=10000, type = 'rough')

#design, response_noiseless, true_signal = es.SimulationData.gravity(sample_size=100)
#design, response_noiseless, true_signal = es.SimulationData.heat(sample_size=100)
#design, response_noiseless, true_signal = es.SimulationData.deriv2(sample_size=100)
#design, response_noiseless, true_signal = es.SimulationData.phillips(sample_size=100)

parameters_smooth = es.SimulationParameters(
    design=design_smooth,
    true_signal=true_signal_smooth,
    true_noise_level=0.01,
    max_iteration=1000,
    monte_carlo_runs=10,
    cores=12
)

parameters_supersmooth = es.SimulationParameters(
    design=design_supersmooth,
    true_signal=true_signal_supersmooth,
    true_noise_level=0.01,
    max_iteration=1000,
    monte_carlo_runs=10, #500
    cores=12
)

parameters_rough = es.SimulationParameters(
    design=design_rough,
    true_signal=true_signal_rough,
    true_noise_level=0.01,
    max_iteration=1500,
    monte_carlo_runs=10,
    cores=12
)

simulation_smooth = es.SimulationWrapper(**parameters_smooth.__dict__)
simulation_supersmooth = es.SimulationWrapper(**parameters_supersmooth.__dict__)
simulation_rough = es.SimulationWrapper(**parameters_rough.__dict__)

results_smooth = simulation_smooth.run_simulation_landweber(data_set_name="landweber_simulation_smooth") # use learning_rate = "auto" for the best learning rate
results_supersmooth = simulation_supersmooth.run_simulation_landweber(data_set_name="landweber_simulation_supersmooth") # use learning_rate = "auto" for the best learning rate
results_rough = simulation_rough.run_simulation_landweber(data_set_name="landweber_simulation_rough") # use learning_rate = "auto" for the best learning rate

weak_relative_efficiency_smooth = np.array(results_smooth["landweber_weak_relative_efficiency"])
strong_relative_efficiency_smooth = np.array(results_smooth["landweber_strong_relative_efficiency"])
stopping_iteration_landweber_smooth = np.array(results_smooth["stopping_index_landweber"])
weak_balanced_oracle_iteration_smooth = np.array(results_smooth["balanced_oracle_weak"])
strong_balanced_oracle_iteration_smooth = np.array(results_smooth["balanced_oracle_strong"])

weak_relative_efficiency_supersmooth = np.array(results_supersmooth["landweber_weak_relative_efficiency"])
strong_relative_efficiency_supersmooth = np.array(results_supersmooth["landweber_strong_relative_efficiency"])
stopping_iteration_landweber_supersmooth = np.array(results_supersmooth["stopping_index_landweber"])
weak_balanced_oracle_iteration_supersmooth = np.array(results_supersmooth["balanced_oracle_weak"])
strong_balanced_oracle_iteration_supersmooth = np.array(results_supersmooth["balanced_oracle_strong"])

weak_relative_efficiency_rough = np.array(results_rough["landweber_weak_relative_efficiency"])
strong_relative_efficiency_rough = np.array(results_rough["landweber_strong_relative_efficiency"])
stopping_iteration_landweber_rough = np.array(results_rough["stopping_index_landweber"])
weak_balanced_oracle_iteration_rough = np.array(results_rough["balanced_oracle_weak"])
strong_balanced_oracle_iteration_rough = np.array(results_rough["balanced_oracle_strong"])

# Relative iterations
weak_relative_iteration_smooth = stopping_iteration_landweber_smooth/weak_balanced_oracle_iteration_smooth
strong_relative_iteration_smooth = stopping_iteration_landweber_smooth/strong_balanced_oracle_iteration_smooth

weak_relative_iteration_supersmooth = stopping_iteration_landweber_supersmooth/weak_balanced_oracle_iteration_supersmooth
strong_relative_iteration_supersmooth = stopping_iteration_landweber_supersmooth/strong_balanced_oracle_iteration_supersmooth

weak_relative_iteration_rough = stopping_iteration_landweber_rough/weak_balanced_oracle_iteration_rough
strong_relative_iteration_rough = stopping_iteration_landweber_rough/strong_balanced_oracle_iteration_rough

def create_custom_boxplot(data, labels, y_lim_lower, y_lim_upper, fig_dir, name):
    # Create a boxplot for the given data
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels=labels)

    # Define custom colors
    colors = ['blue', 'purple', '#CCCC00', 'blue', 'purple', '#CCCC00']

    # Set colors for each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)  # Set the border thickness

    # Making whiskers, caps, and medians thicker and grey
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(1.5)

    # Add a horizontal line at y=1
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)

    # Enable gridlines
    plt.grid(True)

    # Set y-axis limits (can adjust based on your data)
    plt.ylim(y_lim_lower, y_lim_upper)

    # Customize tick labels and layout
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Add sub-label "weak" beneath the first three x-axis labels
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)

    # Add a single "weak" label beneath the first three x-tick labels
    plt.text(2, plt.ylim()[0] - 0.1, 'weak norm', ha='center', va='top', fontsize=14)
    plt.text(5, plt.ylim()[0] - 0.1, 'strong norm', ha='center', va='top', fontsize=14)


    plt.savefig(os.path.join(fig_dir, f'boxplot_{name}.png'), bbox_inches='tight', dpi=300)

    plt.tight_layout()  # Adjust layout

    # Show the plot
    plt.show()

efficiency_to_plot = [
    weak_relative_efficiency_supersmooth,
    weak_relative_efficiency_smooth,
    weak_relative_efficiency_rough,
    strong_relative_efficiency_supersmooth,
    strong_relative_efficiency_smooth,
    strong_relative_efficiency_rough
]

# Labels for the boxplot
labels = [
    "supersmooth", "smooth", "rough",
    "supersmooth", "smooth", "rough"
]

fig_dir = ""

create_custom_boxplot(efficiency_to_plot, labels, y_lim_lower = 0.3, y_lim_upper=1.3, fig_dir=fig_dir, name='efficiency')


relative_iteration_to_plot = [weak_relative_iteration_supersmooth,
                              weak_relative_iteration_smooth,
                              weak_relative_iteration_rough,
                              strong_relative_iteration_supersmooth,
                              strong_relative_iteration_smooth,
                              strong_relative_iteration_rough]

create_custom_boxplot(relative_iteration_to_plot, labels, y_lim_lower = 0.3, y_lim_upper=1.3, fig_dir=fig_dir, name='iteration')

