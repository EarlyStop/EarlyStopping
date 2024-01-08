import matplotlib.pyplot as plt
import numpy as np
import EarlyStopping as es
import random, time
random.seed(42)

# Create diagonal design matrices
sample_size = 1000
indices = np.arange(sample_size)+1
print(len(indices))
input_matrix = np.diag(1/(np.sqrt(indices)))

# Create signals from Stankewitz (2020)
signal_supersmooth = 5*np.exp(-0.1*indices)
signal_smooth = 5000*np.abs(np.sin(0.01*indices))*indices**(-1.6)
signal_rough = 250*np.abs(np.sin(0.002*indices))*indices**(-0.8)

plt.plot(indices, signal_supersmooth, label="supersmooth signal")
plt.plot(indices, signal_smooth, label="smooth signal")
plt.plot(indices, signal_rough, label="rough signal")
plt.ylabel("Signal")
plt.xlabel("Index")
plt.xlim([0,1000])
plt.ylim([0,1.6])
plt.legend()
plt.show()

# Specify number of Monte-Carlo runs
NUMBER_RUNS = 1

# Create observations
NOISE_LEVEL = 0.01
noise = np.random.normal(0, NOISE_LEVEL, (sample_size, NUMBER_RUNS))
observation_supersmooth = noise + np.matmul(input_matrix, signal_supersmooth)[:, None]
observation_smooth = noise + np.matmul(input_matrix, signal_smooth)[:, None]
observation_rough = noise + np.matmul(input_matrix, signal_rough)[:, None]

# Create models
models_supersmooth = [es.ConjugateGradients(input_matrix, observation_supersmooth[:, i], true_signal=signal_supersmooth, true_noise_level=NOISE_LEVEL) for i in range(NUMBER_RUNS)]
models_smooth = [es.ConjugateGradients(input_matrix, observation_smooth[:, i], true_signal=signal_smooth, true_noise_level=NOISE_LEVEL) for i in range(NUMBER_RUNS)]
models_rough = [es.ConjugateGradients(input_matrix, observation_rough[:, i], true_signal=signal_rough, true_noise_level=NOISE_LEVEL) for i in range(NUMBER_RUNS)]

# Calculate conjugate gradients estimates after 1000 iterations
NUMBER_ITERATIONS = 1000
for i in range(NUMBER_RUNS):
    start_time = time.time()
    models_supersmooth[i].conjugate_gradients_gather_all(NUMBER_ITERATIONS)
    models_smooth[i].conjugate_gradients_gather_all(NUMBER_ITERATIONS)
    models_rough[i].conjugate_gradients_gather_all(NUMBER_ITERATIONS)
    end_time = time.time()
    print(f"The {i}-th Montecarlo step took {end_time - start_time} seconds!")

for i in range(NUMBER_RUNS):
    print(f"Supersmooth stopping index: {models_supersmooth[i].early_stopping_index}")
    print(f"Smooth stopping index: {models_smooth[i].early_stopping_index}")
    print(f"Rough stopping index: {models_rough[i].early_stopping_index}")

montecarlo_residuals_supersmooth = [models_supersmooth[i].residuals for i in range(NUMBER_RUNS)]
montecarlo_residuals_smooth = [models_smooth[i].residuals for i in range(NUMBER_RUNS)]
montecarlo_residuals_rough = [models_rough[i].residuals for i in range(NUMBER_RUNS)]

means_montecarlo_residuals_supersmooth = np.mean(montecarlo_residuals_supersmooth, 0)
means_montecarlo_residuals_smooth = np.mean(montecarlo_residuals_smooth, 0)
means_montecarlo_residuals_rough = np.mean(montecarlo_residuals_rough, 0)

montecarlo_weak_empirical_error_supersmooth = [models_supersmooth[i].weak_empirical_error for i in range(NUMBER_RUNS)]
montecarlo_weak_empirical_error_smooth = [models_smooth[i].weak_empirical_error for i in range(NUMBER_RUNS)]
montecarlo_weak_empirical_error_rough = [models_rough[i].weak_empirical_error for i in range(NUMBER_RUNS)]

means_montecarlo_weak_empirical_error_supersmooth = np.mean(montecarlo_weak_empirical_error_supersmooth, 0)
means_montecarlo_weak_empirical_error_smooth = np.mean(montecarlo_weak_empirical_error_smooth, 0)
means_montecarlo_weak_empirical_error_rough = np.mean(montecarlo_weak_empirical_error_rough, 0)

montecarlo_strong_empirical_error_supersmooth = [models_supersmooth[i].strong_empirical_error for i in range(NUMBER_RUNS)]
montecarlo_strong_empirical_error_smooth = [models_smooth[i].strong_empirical_error for i in range(NUMBER_RUNS)]
montecarlo_strong_empirical_error_rough = [models_rough[i].strong_empirical_error for i in range(NUMBER_RUNS)]

means_montecarlo_strong_empirical_error_supersmooth = np.mean(montecarlo_strong_empirical_error_supersmooth, 0)
means_montecarlo_strong_empirical_error_smooth = np.mean(montecarlo_strong_empirical_error_smooth, 0)
means_montecarlo_strong_empirical_error_rough = np.mean(montecarlo_strong_empirical_error_rough, 0)

fig, axs = plt.subplots(3, 1, figsize=(14, 8))

axs[0].set_xlim([0, 20])  
axs[0].set_ylim([0, 100])  
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Residuals')

axs[0].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_residuals_supersmooth, label="supersmooth")
axs[0].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_residuals_smooth, label="smooth")
axs[0].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_residuals_rough, label="rough")
axs[0].legend()

axs[1].set_xlim([0, 20])  
axs[1].set_ylim([0, 0.2])  
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Weak empirical error')

axs[1].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_weak_empirical_error_supersmooth, label="supersmooth")
axs[1].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_weak_empirical_error_smooth, label="smooth")
axs[1].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_weak_empirical_error_rough, label="rough")
axs[1].legend()

axs[2].set_xlim([0, 40])  
axs[2].set_ylim([0, 100])  
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Strong empirical error')

axs[2].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_strong_empirical_error_supersmooth, label="supersmooth")
axs[2].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_strong_empirical_error_smooth, label="smooth")
axs[2].plot(range(NUMBER_ITERATIONS+1), means_montecarlo_strong_empirical_error_rough, label="rough")
axs[2].legend()
plt.show()