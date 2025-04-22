import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Ensure consistent style - using the style from error_decomposition_plots.py
plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=15)
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)

importlib.reload(es)

sample_size = 100
max_iteration = 10000

# design, response_noiseless, true_signal = es.SimulationData.gravity(sample_size=sample_size)
design, response_noiseless, true_signal = es.SimulationData.phillips(sample_size=sample_size)
# design, response_noiseless, true_signal = es.SimulationData.diagonal_data(sample_size=sample_size, type="smooth")


# Define a consistent colormap for both heatmaps
colormap = "viridis"

# Create and save design matrix heatmap as a separate image
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
heatmap = sns.heatmap(design, cmap=colormap, cbar_kws={"label": "Value"}, ax=ax)
# Set colorbar label font size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(" ", fontsize=14)
# No title for design matrix
# Remove axis labels and ticks
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("design_matrix_heatmap.png", dpi=300, bbox_inches="tight")

# Create and save true signal as a line plot
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.plot(range(len(true_signal)), true_signal, linewidth=1.5, color="blue")
ax.grid(True)
ax.set_xlabel("Iteration $m$")
ax.set_ylabel("")
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("true_signal_lineplot.png", dpi=300, bbox_inches="tight")

# Show the original plots
plt.show()
