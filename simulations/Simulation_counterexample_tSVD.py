import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt

plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=15)
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)

importlib.reload(es)

sample_size = 100
max_iteration = 100

design, response_noiseless, true_signal = es.SimulationData.phillips(sample_size=sample_size)

true_noise_level = 1 / 10
noise = true_noise_level * np.random.normal(0, 1, sample_size)
response = response_noiseless + noise


model = es.TruncatedSVD(design, response, true_signal=true_signal, true_noise_level=true_noise_level)


model.iterate(max_iteration)

# Stopping index
m_gravity = model.get_discrepancy_stop(sample_size * (true_noise_level**2), max_iteration)

# Weak balanced oracle
weak_oracle_gravity = model.get_weak_balanced_oracle(max_iteration)

# Strong balanced oracle
strong_oracle_gravity = model.get_strong_balanced_oracle(max_iteration)

# Create separate figure for Strong Quantities
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")

# Plot elements with matching colors and styles
ax.plot(range(0, max_iteration + 1), model.strong_bias2, color="blue", linewidth=1.5, label=r"$a_m(g^*)$")
ax.plot(range(0, max_iteration + 1), model.strong_variance, color="red", linewidth=2, label=r"$s_m$")
ax.plot(range(0, max_iteration + 1), model.strong_risk, color="black", linewidth=1.5, label=r"$\mathcal{R}(g^*, m)$")
ax.axvline(x=m_gravity, ymin=0, ymax=0.6, color="green", linestyle="--", linewidth=1.5, label=r"$\tau$")
ax.axvline(
    x=strong_oracle_gravity, ymin=0, ymax=0.6, color="orange", linestyle="--", linewidth=1.5, label=r"$t$ (oracle)"
)
ax.set_xlim([0, 24])
ax.set_ylim([0, 0.5])
ax.grid(True)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("strong_quantities_plot.png", dpi=300, bbox_inches="tight")

# Create separate figure for Weak Quantities
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")

# Plot elements with matching colors and styles
ax.plot(range(0, max_iteration + 1), model.weak_bias2, color="blue", linewidth=1.5, label=r"$a_m(g^*)$")
ax.plot(range(0, max_iteration + 1), model.weak_variance, color="red", linewidth=2, label=r"$s_m$")
ax.plot(range(0, max_iteration + 1), model.weak_risk, color="black", linewidth=1.5, label=r"$\mathcal{R}(g^*, m)$")
ax.axvline(x=m_gravity, ymin=0, ymax=0.6, color="green", linestyle="--", linewidth=1.5, label=r"$\tau$")
ax.axvline(
    x=weak_oracle_gravity, ymin=0, ymax=0.6, color="orange", linestyle="--", linewidth=1.5, label=r"$t$ (oracle)"
)
ax.set_xlim([0, 24])
ax.set_ylim([0, 0.5])
ax.grid(True)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("weak_quantities_plot.png", dpi=300, bbox_inches="tight")

plt.show()
