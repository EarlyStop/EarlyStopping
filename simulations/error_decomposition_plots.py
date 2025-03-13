import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure consistent style
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# Data preparation
index = np.arange(1, 125)
alpha = 0.001
variance = index * (2.0) * alpha
# bias_1 = (1000 / (index) ** 0.5) * alpha
bias_1 = 1000 * np.exp(-0.09 * index) * alpha
risk_1 = bias_1 + variance
# risk_2 = bias_2 + variance

oracle = np.argmin(risk_1)
balanced_oracle = np.where(variance > bias_1)[0][0]


print(f"The oracle is {oracle} and the balanced orcale is {balanced_oracle}!")

# Figure setup
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

#Plot elements with matching colors and styles
ax.plot(index, bias_1, color="blue", linewidth=1.5, label=r"$a_m(g^*)$")
ax.plot(index, variance, linewidth=2, color="red", label=r"$s_m$")
ax.plot(index, risk_1, color="black", linewidth=1.5, label=r"$\mathcal{R}(g^*, m)$")
ax.axvline(x=oracle, ymin=0, ymax=0.6, color="black", linestyle="--", linewidth=1.5)
ax.text(oracle-2, 630 * alpha, r"$m^\mathfrak{o}(g^*)$", fontsize=14)
ax.axvline(x=balanced_oracle, ymin=0, ymax=0.6, color="black", linestyle="--", linewidth=1.5)
ax.text(balanced_oracle - 4, 630 * alpha, r"$m^\mathfrak{b}(g^*)$", fontsize=14)

# ax.plot(index, bias_2, color="blue", linestyle="--", linewidth=1.5, label=r"$a_m(f^*)$")
# ax.plot(index, risk_2, color="black", linestyle="--", linewidth=1.5, label=r"$\mathcal{R}(f^*, m)$")
# ax.axvline(x=5, ymin=0, ymax=0.6, color="black", linestyle="--", linewidth=1.5)
# ax.text(5, 650, r"$m^o(f^*)$", fontsize=14, ha="center")

# ax.plot(index, variance, linewidth=2, color="red", label=r"$s_m$")

# Horizontal reference line
# ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5)

# Labels and legend
# ax.set_xlabel("Iterations $m$")
#ax.set_title("General risk decomposition")
ax.legend(fontsize=14)
ax.set_ylim(0,1)

# Enable grid for better readability
ax.grid(True)

ax.set_xlabel("Iteration $m$")  # Remove x-axis label
ax.set_ylabel("")  # Remove y-axis label

# ax.set_xticks([])
ax.tick_params(axis='y', length=0)
# ax.set_yticklabels([])

# Save figure
fig_dir = "."  # Change to desired directory
fig.savefig(os.path.join(fig_dir, "GeneralBiasVarianceDecomposition_2.png"), bbox_inches="tight", dpi=300)

# Show plot
plt.show()