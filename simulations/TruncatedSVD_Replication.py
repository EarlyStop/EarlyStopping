################################################################################
#             Reproduction study for truncated SVD estimation                  #
################################################################################

# Imports
# ------------------------------------------------------------------------------
import numpy             as np
import matplotlib.pyplot as plt
import EarlyStopping     as es
import pandas as pd
import os

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

# Weak relative efficiency
supersmooth_weak_relative_efficiency = results_supersmooth["weak_relative_efficiency"]
smooth_weak_relative_efficiency      = results_smooth["weak_relative_efficiency"]
rough_weak_relative_efficiency       = results_rough["weak_relative_efficiency"]

data = [supersmooth_strong_relative_efficiency,
        smooth_strong_relative_efficiency,
        rough_strong_relative_efficiency,
        supersmooth_weak_relative_efficiency,
        smooth_weak_relative_efficiency,
        rough_weak_relative_efficiency
        ]

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

# Labels for the boxplot
labels = [
    "supersmooth", "smooth", "rough",
    "supersmooth", "smooth", "rough"
]

fig_dir = ""

create_custom_boxplot(data, labels, y_lim_lower = 0.3, y_lim_upper=1.3, fig_dir=fig_dir, name='efficiency_SVD')



# Signal plot:

plt.figure(figsize=(10, 6))
plt.plot(indices, true_signal_supersmooth, color='blue')
plt.plot(indices, true_signal_smooth, color='purple')
plt.plot(indices, true_signal_rough, color='#CCCC00')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim([0, 1.6])
# Add sub-label "weak" beneath the first three x-axis labels
#plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)

# **Add the main x-axis label**
plt.xlabel(" ", fontsize = 22)


# Add a single "weak" label beneath the first three x-tick labels
# plt.text(2, plt.ylim()[0] - 0.1, '', ha='center', va='top', fontsize=14)
# plt.text(5, plt.ylim()[0] - 0.1, '', ha='center', va='top', fontsize=14)

# plt.show()

plt.savefig(os.path.join(fig_dir, f'signals.png'), bbox_inches='tight', dpi=300)









