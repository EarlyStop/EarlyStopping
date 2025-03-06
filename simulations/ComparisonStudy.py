import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt
import os

importlib.reload(es)


sample_size = 100
name = "phillips"

design, response_noiseless, true_signal = None, None, None
if name == "phillips":
    design, response_noiseless, true_signal = es.SimulationData.phillips(sample_size=sample_size)
elif name == "gravity":
    # Generate data using the Gravity example
    design, response_noiseless, true_signal = es.SimulationData.gravity(sample_size=100, d=0.2)


# design, response_noiseless, true_signal = es.SimulationData.diagonal_data(sample_size=100, type="rough")

# Define simulation parameters
parameters = es.SimulationParameters(
    design=design, true_signal=true_signal, true_noise_level=0.1, monte_carlo_runs=100, cores=12  # 0.01,
)

# Create SimulationWrapper instance
simulation = es.SimulationWrapper(**parameters.__dict__)

# Run simulations for each method
results_landweber = simulation.run_simulation_landweber(
    learning_rate=1 / 100, max_iteration=10000, 
    data_set_name="landweber_simulation_gravity"
)

results_cg = simulation.run_simulation_conjugate_gradients(
    max_iteration=500,
    data_set_name="conjugate_gradients_simulation_gravity"
)

results_svd = simulation.run_simulation_truncated_svd(
    max_iteration=sample_size,
    diagonal=False,
    data_set_name="truncated_svd_simulation_gravity"
)

# Extract relative efficiencies for each method
# Landweber
landweber_weak_efficiency = np.array(results_landweber["landweber_weak_relative_efficiency"])
landweber_strong_efficiency = np.array(results_landweber["landweber_strong_relative_efficiency"])

# Conjugate Gradients
cg_weak_efficiency = np.array(results_cg["conjugate_gradients_weak_relative_efficiency"])
cg_strong_efficiency = np.array(results_cg["conjugate_gradients_strong_relative_efficiency"])

# Truncated SVD
svd_weak_efficiency = np.array(results_svd["weak_relative_efficiency"])
svd_strong_efficiency = np.array(results_svd["strong_relative_efficiency"])

# Prepare data for plotting
efficiency_to_plot = [
    landweber_weak_efficiency,
    cg_weak_efficiency,
    svd_weak_efficiency,
    landweber_strong_efficiency,
    cg_strong_efficiency,
    svd_strong_efficiency,
]


def create_custom_boxplot(data, labels, y_lim_lower, y_lim_upper, fig_dir, name):
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels=labels)

    # Define custom colors for each method
    colors = ["blue", "purple", "#CCCC00", "blue", "purple", "#CCCC00"]  # Blue, Green, Orange

    # Set colors for each box
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)

    # Making whiskers, caps, and medians thicker
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

    # Set y-axis limits
    plt.ylim(y_lim_lower, y_lim_upper)

    # Customize tick labels and layout
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)

    # Add norm type labels
    plt.text(2, plt.ylim()[0] - 0.1, "weak norm", ha="center", va="top", fontsize=14)
    plt.text(5, plt.ylim()[0] - 0.1, "strong norm", ha="center", va="top", fontsize=14)

    # Add title
    # plt.title("Method Comparison - Gravity Example", fontsize=16, pad=20)

    plt.savefig(os.path.join(fig_dir, f"boxplot_{name}.png"), bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()


# Labels for the boxplot
labels = ["Landweber", "CG", "SVD", "Landweber", "CG", "SVD"]

fig_dir = ""

# Create comparison boxplot
create_custom_boxplot(
    efficiency_to_plot, labels, y_lim_lower=0, y_lim_upper=1.3, fig_dir=fig_dir, name=f"method_comparison_{name}"
)
