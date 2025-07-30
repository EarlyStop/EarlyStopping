import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt

np.random.seed(21)

# Ensure consistent style - using the style from error_decomposition_plots.py
plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=15)
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)
plt.tick_params(axis="both", which="major", labelsize=14)

importlib.reload(es)
plt.close()
sample_size = 1000
max_iteration = 200

design, response_noiseless, true_signal = es.SimulationData.diagonal_data(sample_size=sample_size, type="supersmooth")

true_noise_level = 1 / 10
noise = true_noise_level * np.random.normal(0, 1, sample_size)
response = response_noiseless + noise

model_landweber = es.Landweber(
    design, response, learning_rate=1 / 100, true_signal=true_signal, true_noise_level=true_noise_level
)
model_landweber.iterate(max_iteration)

model_svd = es.TruncatedSVD(
    design, response, true_signal=true_signal, true_noise_level=true_noise_level, diagonal=True
)
model_svd.iterate(max_iteration)


# Stopping index
m_gravity = model_svd.get_discrepancy_stop(sample_size * (true_noise_level**2), max_iteration)

# Weak balanced oracle
weak_oracle = model_svd.get_weak_balanced_oracle(max_iteration)

# Strong balanced oracle
strong_oracle = model_svd.get_strong_balanced_oracle(max_iteration)

# Create separate figure for Strong Quantities
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")

# Plot elements with matching colors and styles
ax.plot(range(0, max_iteration + 1), model_svd.strong_bias2, color="blue", linewidth=1.5, label=r"$a_m(g^*)$")
ax.plot(range(0, max_iteration + 1), model_svd.strong_variance, color="red", linewidth=2, label=r"$s_m$")
ax.plot(
    range(0, max_iteration + 1), model_svd.strong_risk, color="black", linewidth=1.5, label=r"$\mathcal{R}(g^*, m)$"
)

ax.axvline(x=m_gravity, ymin=0, ymax=0.6, color="green", linestyle="--", linewidth=1.5, label=r"$\tau$")
ax.axvline(x=strong_oracle, ymin=0, ymax=0.6, color="orange", linestyle="--", linewidth=1.5, label=r"$t$ (oracle)")
ax.set_xlim([0, 49])
ax.set_ylim([0, 20])
ax.grid(True)

plt.tight_layout()
plt.tick_params(axis="both", which="major", labelsize=14)
plt.savefig("demonstration_strong_quantities_plot.png", dpi=300, bbox_inches="tight")

plt.show()

# Create separate figure for Strong Quantities
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")

# Plot elements with matching colors and styles
ax.plot(range(0, max_iteration + 1), model_svd.weak_bias2, color="blue", linewidth=1.5, label=r"$a_m(g^*)$")
ax.plot(range(0, max_iteration + 1), model_svd.weak_variance, color="red", linewidth=2, label=r"$s_m$")
ax.plot(range(0, max_iteration + 1), model_svd.weak_risk, color="black", linewidth=1.5, label=r"$\mathcal{R}(g^*, m)$")

print(weak_oracle)
ax.axvline(x=m_gravity, ymin=0, ymax=0.6, color="green", linestyle="--", linewidth=1.5, label=r"$\tau$")
ax.axvline(x=weak_oracle, ymin=0, ymax=0.6, color="orange", linestyle="--", linewidth=1.5, label=r"$t$ (oracle)")
ax.set_xlim([0, 49])
ax.set_ylim([0, 1])
ax.grid(True)
plt.tight_layout()
plt.tick_params(axis="both", which="major", labelsize=14)
plt.savefig("demonstration_weak_quantities_plot.png", dpi=300, bbox_inches="tight")
plt.show()
