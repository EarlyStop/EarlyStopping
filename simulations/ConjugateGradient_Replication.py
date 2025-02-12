import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt
import os


importlib.reload(es)

# Generate data for different signals
design_smooth, response_noiseless_smooth, true_signal_smooth = es.SimulationData.diagonal_data(sample_size=10000, type='smooth')
design_supersmooth, response_noiseless_supersmooth, true_signal_supersmooth = es.SimulationData.diagonal_data(sample_size=10000, type='supersmooth')
design_rough, response_noiseless_rough, true_signal_rough = es.SimulationData.diagonal_data(sample_size=10000, type='rough')

# Define simulation parameters
parameters_smooth = es.SimulationParameters(
    design=design_smooth,
    true_signal=true_signal_smooth,
    true_noise_level=0.01,
    max_iteration=5000,
    monte_carlo_runs=1000,
    cores=12
)

parameters_supersmooth = es.SimulationParameters(
    design=design_supersmooth,
    true_signal=true_signal_supersmooth,
    true_noise_level=0.01,
    max_iteration=5000,
    monte_carlo_runs=1000,
    cores=12
)

parameters_rough = es.SimulationParameters(
    design=design_rough,
    true_signal=true_signal_rough,
    true_noise_level=0.01,
    max_iteration=5000,
    monte_carlo_runs=1000,
    cores=12
)

# Create SimulationWrapper instances
simulation_smooth = es.SimulationWrapper(**parameters_smooth.__dict__)
simulation_supersmooth = es.SimulationWrapper(**parameters_supersmooth.__dict__)
simulation_rough = es.SimulationWrapper(**parameters_rough.__dict__)

# Run Conjugate Gradients simulations
results_smooth = simulation_smooth.run_simulation_conjugate_gradients()
results_supersmooth = simulation_supersmooth.run_simulation_conjugate_gradients()
results_rough = simulation_rough.run_simulation_conjugate_gradients()

# Extract strong relative efficiencies
strong_relative_efficiency_smooth = np.array([res[8] for res in results_smooth])
strong_relative_efficiency_supersmooth = np.array([res[8] for res in results_supersmooth])
strong_relative_efficiency_rough = np.array([res[8] for res in results_rough])

# Extract weak relative efficiencies
weak_relative_efficiency_smooth = np.array([res[9] for res in results_smooth])
weak_relative_efficiency_supersmooth = np.array([res[9] for res in results_supersmooth])
weak_relative_efficiency_rough = np.array([res[9] for res in results_rough])

# Plot results
def create_custom_boxplot(data, labels, y_lim_lower, y_lim_upper, fig_dir, name):
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels=labels)

    colors = ['blue', 'purple', '#CCCC00', 'blue', 'purple', '#CCCC00']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(1.5)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
    plt.grid(True)
    plt.ylim(y_lim_lower, y_lim_upper)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.text(2, plt.ylim()[0] - 0.1, 'weak norm', ha='center', va='top', fontsize=14)
    plt.text(5, plt.ylim()[0] - 0.1, 'strong norm', ha='center', va='top', fontsize=14)

    plt.savefig(os.path.join(fig_dir, f'boxplot_{name}.png'), bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


efficiency_to_plot = [
    weak_relative_efficiency_supersmooth,
    weak_relative_efficiency_smooth,
    weak_relative_efficiency_rough,
    strong_relative_efficiency_supersmooth,
    strong_relative_efficiency_smooth,
    strong_relative_efficiency_rough
]

labels = ["supersmooth", "smooth", "rough", "supersmooth", "smooth", "rough"]
fig_dir = ""

create_custom_boxplot(efficiency_to_plot, labels, y_lim_lower=0, y_lim_upper=1, fig_dir=fig_dir, name='cg_efficiency')
