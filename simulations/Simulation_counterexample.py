import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Set all font sizes to 14
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14
})

importlib.reload(es)

sample_size = 100
max_iteration = 100



# design, response_noiseless, true_signal = es.SimulationData.gravity(sample_size=sample_size)
design, response_noiseless, true_signal = es.SimulationData.phillips(sample_size=sample_size)
# design, response_noiseless, true_signal = es.SimulationData.diagonal_data(sample_size=sample_size, type="smooth")

print(true_signal)

# simulation = es.SimulationWrapper(**parameters.__dict__)
# results = simulation.run_simulation_landweber(data_set_name = "gravity_simulation")

true_noise_level = 1/10
noise = true_noise_level * np.random.normal(0, 1, sample_size)
response = response_noiseless + noise

# model_gravity = es.Landweber(
#     design, response, learning_rate=1 / 100, true_signal=true_signal, true_noise_level=true_noise_level
# )

model_gravity = es.TruncatedSVD(
    design, response, true_signal=true_signal, true_noise_level=true_noise_level
)


model_gravity.iterate(max_iteration)

# Stopping index
m_gravity = model_gravity.get_discrepancy_stop(sample_size * (true_noise_level**2), max_iteration)

print(m_gravity)
# Weak balanced oracle
weak_oracle_gravity = model_gravity.get_weak_balanced_oracle(max_iteration)

# Strong balanced oracle
strong_oracle_gravity = model_gravity.get_strong_balanced_oracle(max_iteration)

# Original figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

print(len(model_gravity.residuals))

axs[0].plot(range(0, max_iteration + 1), model_gravity.residuals)
# axs[0].axvline(x=m, color="red", linestyle="--")
axs[0].set_xlim([0, 50])
axs[0].set_ylim([0, 10000])
axs[0].set_xlabel("Iteration", fontsize=14)
axs[0].set_ylabel("Residuals", fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=14)


# axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_risk, color="orange", label="Error")
axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_mse, color="orange", label="Error")
axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_bias2, label="$Bias^2$", color="grey")
axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_variance, label="Variance", color="blue")
axs[1].axvline(x=m_gravity, color="red", linestyle="--")
axs[1].axvline(x=strong_oracle_gravity, color="green", linestyle="--")
# axs[1].set_xlim([0, 50])
axs[1].set_xlim([0, 50])
axs[1].set_ylim([0, 0.5])  # 0.2
axs[1].set_xlabel("Iteration", fontsize=14)
axs[1].set_ylabel("Strong Quantities", fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=14)

print(model_gravity.weak_variance)

#axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_risk, color="orange", label="Error")
axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_mse, color="orange", label="Error")
axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_bias2, label="$Bias^2$", color="grey")
axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_variance, label="Variance", color="blue")
axs[2].axvline(x=m_gravity, color="red", linestyle="--", label=r"$\tau$")
axs[2].axvline(x=weak_oracle_gravity, color="green", linestyle="--", label="$t$ (oracle)")
axs[2].set_xlim([0, 50])
axs[2].set_ylim([0, 0.5])  # 0.002
axs[2].set_xlabel("Iteration", fontsize=14)
axs[2].set_ylabel("Weak Quantities", fontsize=14)
axs[2].legend(loc="upper right", fontsize=14)
axs[2].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# Define a consistent colormap for both heatmaps
colormap = "viridis"

# Create and save design matrix heatmap as a separate image
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(design, cmap=colormap, cbar_kws={'label': 'Value'})
# Set colorbar label font size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(' ', fontsize=14)
# No title for design matrix
# Remove axis labels and ticks
plt.xlabel("", fontsize=14)
plt.ylabel("", fontsize=14)
# plt.xticks([])
#plt.yticks([])
plt.tight_layout()
plt.savefig('design_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create and save true signal as a line plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(true_signal)), true_signal, linewidth=1.5, color="blue")
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('true_signal_lineplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Show the original plots
plt.show()
