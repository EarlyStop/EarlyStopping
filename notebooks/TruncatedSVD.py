################################################################################
#             Reproduction example for truncated SVD estimation                #
################################################################################

# Imports
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import EarlyStopping as es


# Plot different signals
# ------------------------------------------------------------------------------

D = 10000
indices = np.arange(D) + 1

signal_supersmooth = 5    * np.exp(-0.1 * indices)
signal_smooth      = 5000 * np.abs(np.sin(0.01  * indices))  * indices**(-1.6)
signal_rough       = 250  * np.abs(np.sin(0.002 * indices))  * indices**(-0.8)

plt.figure(figsize=(14, 4))
plt.plot(indices, signal_supersmooth, label="supersmooth signal")
plt.plot(indices, signal_smooth, label="smooth signal")
plt.plot(indices, signal_rough, label="rough signal")
plt.ylabel("Signal")
plt.xlabel("Index")
plt.ylim([0, 1.6])
plt.legend(loc="upper right")
plt.show()

# Display class functionality on an individual example
# ------------------------------------------------------------------------------

# Choose true model quantities
true_signal      = signal_rough
eigenvalues      = indices**(-0.5)
design           = np.diag(eigenvalues)
true_noise_level = 0.01

# Simulate data
response = eigenvalues * true_signal + \
           true_noise_level * np.random.normal(0, 1, D)

# Initialize TruncatedSVD class and iterate
alg = es.TruncatedSVD(design, response, true_signal, true_noise_level,
                      diagonal = True)

# Bias-variance decomposition and oracle quantities
alg.iterate(3000)

plt.figure()
plt.plot(indices[0: alg.iteration + 1], alg.weak_variance, label="Variance")
plt.plot(indices[0: alg.iteration + 1], alg.weak_bias2, label="Bias")
alg.get_weak_balanced_oracle(3000)

plt.figure()
plt.plot(indices[0: alg.iteration + 1], alg.strong_variance, label="Variance")
plt.plot(indices[0: alg.iteration + 1], alg.strong_bias2, label="Bias")
alg.get_strong_balanced_oracle(3000)

# Early stopping w/ discrepancy principle
critical_value   = D * true_noise_level**2
discrepancy_time = alg.get_discrepancy_stop(critical_value, 3000)
estimated_signal = alg.get_estimate(discrepancy_time)

plt.figure(figsize=(14, 4))
plt.plot(indices, estimated_signal)
plt.plot(indices, true_signal)
plt.ylim([0, 2])

---------------------------------------------------------------------------------

# Two step procedure
aic_index = alg.get_aic_iteration(4000)

aic = alg.preliminary_aic + 2 * true_noise_level**2 * np.sum(alg.diagonal_design



















