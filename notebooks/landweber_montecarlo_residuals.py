import matplotlib.pyplot as plt
import numpy as np
import EarlyStopping as es
import random, time
random.seed(42)

# Create diagonal design matrices
D = 1000
indices = np.arange(D)+1
design_matrix = np.diag(1/(np.sqrt(indices)))
print(f"The design matrix is given by {design_matrix}")

# Create signals from Stankewitz (2020)
signal_supersmooth = 5*np.exp(-0.1*indices)
signal_smooth = 5000*np.abs(np.sin(0.01*indices))*indices**(-1.6)
signal_rough = 250*np.abs(np.sin(0.002*indices))*indices**(-0.8)

plt.plot(indices, signal_supersmooth, label="supersmooth signal")
plt.plot(indices, signal_smooth, label="smooth signal")
plt.plot(indices, signal_rough, label="rough signal")
plt.ylabel("Signal")
plt.xlabel("Index")
plt.xlim([0,2000])
plt.ylim([0,1.6])
plt.legend()
plt.show()

# Specify number of Monte-Carlo runs
NUMBER_RUNS = 100

# Create observations
NOISE_LEVEL = 0.01
noise = np.random.normal(0, NOISE_LEVEL, (D, NUMBER_RUNS))
observation_supersmooth = noise + np.matmul(design_matrix, signal_supersmooth)[:, None]
observation_smooth = noise + np.matmul(design_matrix, signal_smooth)[:, None]
observation_rough = noise + np.matmul(design_matrix, signal_rough)[:, None]

# Create models
models_supersmooth = [es.Landweber(design_matrix, observation_supersmooth[:, i], true_noise_level=NOISE_LEVEL) for i in range(NUMBER_RUNS)]
models_smooth = [es.Landweber(design_matrix, observation_smooth[:, i], true_noise_level=NOISE_LEVEL) for i in range(NUMBER_RUNS)]
models_rough = [es.Landweber(design_matrix, observation_rough[:, i], true_noise_level=NOISE_LEVEL) for i in range(NUMBER_RUNS)]

# Calculate Landweber estimates after 1000 iterations
NUMBER_ITERATIONS = 1000
for i in range(NUMBER_RUNS):
    start_time = time.time()
    models_supersmooth[i].landweber_gather_all(NUMBER_ITERATIONS)
    models_smooth[i].landweber_gather_all(NUMBER_ITERATIONS)
    models_rough[i].landweber_gather_all(NUMBER_ITERATIONS)
    end_time = time.time()
    print(f"The {i}-th Montecarlo step took {end_time - start_time} seconds!")

for i in range(NUMBER_RUNS):
    print(models_supersmooth[i].early_stopping_index)
    print(models_smooth[i].early_stopping_index)
    print(models_rough[i].early_stopping_index)

montecarlo_residuals_supersmooth = [models_supersmooth[i].residuals for i in range(NUMBER_RUNS)]
montecarlo_residuals_smooth = [models_smooth[i].residuals for i in range(NUMBER_RUNS)]
montecarlo_residuals_rough = [models_rough[i].residuals for i in range(NUMBER_RUNS)]

means_montecarlo_residuals_supersmooth = np.mean(montecarlo_residuals_supersmooth, 0)
means_montecarlo_residuals_smooth = np.mean(montecarlo_residuals_smooth, 0)
means_montecarlo_residuals_rough = np.mean(montecarlo_residuals_rough, 0)

plt.plot(range(NUMBER_ITERATIONS+1), means_montecarlo_residuals_supersmooth, label="supersmooth")
plt.plot(range(NUMBER_ITERATIONS+1), means_montecarlo_residuals_smooth, label="smooth")
plt.plot(range(NUMBER_ITERATIONS+1), means_montecarlo_residuals_rough, label="rough")
plt.legend()
plt.xlim([0,40])
plt.ylim([0,100])
plt.ylabel("Squared residual")
plt.xlabel("Iteration index")
plt.show()