import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure consistent style
plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=15)
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)

# Data preparation
index = np.arange(1, 125)
alpha = 0.001
variance = index * (2.0) * alpha
bias_1 = (1000 / (index) ** 0.5) * alpha
risk_1 = bias_1 + variance
bias_2 = 1000 * np.exp(-0.09 * index) * alpha
risk_2 = bias_2 + variance

oracle_1 = np.argmin(risk_1)
balanced_oracle_1 = np.where(variance > bias_1)[0][0]
oracle_2 = np.argmin(risk_2)
balanced_oracle_2 = np.where(variance > bias_2)[0][0]

print(f"The oracle is {oracle_1} and the balanced orcale is {balanced_oracle_1} for the first signal!")
print(f"The oracle is {oracle_2} and the balanced orcale is {balanced_oracle_2} for the second signal!")

# Figure setup
fig_1, ax_1 = plt.subplots(figsize=(10, 6))
fig_1.patch.set_facecolor("white")
fig_2, ax_2 = plt.subplots(figsize=(10, 6))
fig_2.patch.set_facecolor("white")

# Plot elements with matching colors and styles
ax_1.plot(index, bias_1, color="blue", linewidth=1.5, label=r"$a_m(f^*)$")
ax_1.plot(index, variance, linewidth=2, color="red", label=r"$s_m$")
ax_1.plot(index, risk_1, color="black", linewidth=1.5, label=r"$\mathcal{R}(f^*, m)$")
ax_1.axvline(x=oracle_1, ymin=0, ymax=0.6, color="black", linestyle="--", linewidth=1.5)
ax_1.text(oracle_1 - 2, 630 * alpha, r"$m^\mathfrak{o}(f^*)$", fontsize=14)
ax_1.axvline(x=balanced_oracle_1 + 1, ymin=0, ymax=0.6, color="black", linestyle="--", linewidth=1.5)
ax_1.text(balanced_oracle_1 - 4, 630 * alpha, r"$m^\mathfrak{b}(f^*)$", fontsize=14)

ax_2.plot(index, bias_2, color="blue", linewidth=1.5, label=r"$a_m(g^*)$")
ax_2.plot(index, variance, linewidth=2, color="red", label=r"$s_m$")
ax_2.plot(index, risk_2, color="black", linewidth=1.5, label=r"$\mathcal{R}(g^*, m)$")
ax_2.axvline(x=oracle_2, ymin=0, ymax=0.6, color="black", linestyle="--", linewidth=1.5)
ax_2.text(oracle_2 - 2, 630 * alpha, r"$m^\mathfrak{o}(g^*)$", fontsize=14)
ax_2.axvline(x=balanced_oracle_2 + 1, ymin=0, ymax=0.6, color="black", linestyle="--", linewidth=1.5)
ax_2.text(balanced_oracle_2 - 4, 630 * alpha, r"$m^\mathfrak{b}(g^*)$", fontsize=14)

# Labels and legend
ax_1.legend(fontsize=14)
ax_1.set_ylim(0, 1)
ax_2.legend(fontsize=14)
ax_2.set_ylim(0, 1)

# Enable grid for better readability
ax_1.grid(True)
ax_1.set_xlabel("Iteration $m$")  # Set x-axis label
ax_1.set_ylabel("")  # Remove y-axis label
ax_2.grid(True)
ax_2.set_xlabel("Iteration $m$")  # Set x-axis label
ax_2.set_ylabel("")  # Remove y-axis label

# Save figure
fig_dir = "."  # Change to desired directory
fig_1.savefig(os.path.join(fig_dir, "GeneralBiasVarianceDecomposition_1.png"), bbox_inches="tight", dpi=300)
fig_2.savefig(os.path.join(fig_dir, "GeneralBiasVarianceDecomposition_2.png"), bbox_inches="tight", dpi=300)

# Show plot
plt.show()
