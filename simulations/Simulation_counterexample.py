import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt
import os

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

fig, axs = plt.subplots(3, 1, figsize=(14, 12))

print(len(model_gravity.residuals))

axs[0].plot(range(0, max_iteration + 1), model_gravity.residuals)
# axs[0].axvline(x=m, color="red", linestyle="--")
axs[0].set_xlim([0, 50])
axs[0].set_ylim([0, 10000])
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Residuals")


# axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_risk, color="orange", label="Error")
axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_mse, color="orange", label="Error")
axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_bias2, label="$Bias^2$", color="grey")
axs[1].plot(range(0, max_iteration + 1), model_gravity.strong_variance, label="Variance", color="blue")
axs[1].axvline(x=m_gravity, color="red", linestyle="--")
axs[1].axvline(x=strong_oracle_gravity, color="green", linestyle="--")
# axs[1].set_xlim([0, 50])
axs[1].set_xlim([0, 50])
axs[1].set_ylim([0, 0.5])  # 0.2
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Strong Quantities")

print(model_gravity.weak_variance)

#axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_risk, color="orange", label="Error")
axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_mse, color="orange", label="Error")
axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_bias2, label="$Bias^2$", color="grey")
axs[2].plot(range(0, max_iteration + 1), model_gravity.weak_variance, label="Variance", color="blue")
axs[2].axvline(x=m_gravity, color="red", linestyle="--", label=r"$\tau$")
axs[2].axvline(x=weak_oracle_gravity, color="green", linestyle="--", label="$t$ (oracle)")
axs[2].set_xlim([0, 50])
axs[2].set_ylim([0, 0.5])  # 0.002
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Weak Quantities")
axs[2].legend(loc="upper right")

plt.tight_layout()

plt.show()
