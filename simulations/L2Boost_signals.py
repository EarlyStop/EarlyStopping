import numpy as np
import matplotlib.pyplot as plt
import EarlyStopping as es
import pandas as pd
import os

sample_size = 1000
para_size = 1000

cov    = np.identity(para_size)
sigma  = np.sqrt(1)
design = np.random.multivariate_normal(np.zeros(para_size), cov, sample_size)

# Gamma-sparse signals
beta_3 = 1 / (1 + np.arange(para_size)) ** 3
beta_3 = 10 * beta_3 / np.sum(np.abs(beta_3))

beta_2 = 1 / (1 + np.arange(para_size)) ** 2
beta_2 = 10 * beta_2 / np.sum(np.abs(beta_2))

beta_1 = 1 / (1 + np.arange(para_size))
beta_1 = 10 * beta_1 / np.sum(np.abs(beta_1))

# S-sparse signals
beta_15 = np.zeros(para_size)
beta_15[0:15] = 1
beta_15 = 10 * beta_15 / np.sum(np.abs(beta_15))

beta_60 = np.zeros(para_size)
beta_60[0:20] = 1
beta_60[20:40] = 0.5
beta_60[40:60] = 0.25
beta_60 = 10 * beta_60 / np.sum(np.abs(beta_60))

beta_90 = np.zeros(para_size)
beta_90[0:30] = 1
beta_90[30:60] = 0.5
beta_90[60:90] = 0.25
beta_90 = 10 * beta_90 / np.sum(np.abs(beta_90))

plt.figure(figsize=(10, 6))

# Plot only the first 1000 components for better visibility
plot_range = 100
x_indices = np.arange(1, plot_range + 1)

# Plot Landweber estimate
plt.plot(x_indices, beta_3[:plot_range], color="purple", label='Landweber', linewidth=1.5)
# Plot tSVD estimate
plt.plot(x_indices, beta_2[:plot_range], color="#CCCC00", label='tSVD', linewidth=1.5)
# Plot Conjugate Gradient estimate
plt.plot(x_indices, beta_1[:plot_range], color="blue", label='Conjugate Gradient', linewidth=1.5)
# Plot true signal
#plt.plot(x_indices, true_signal[:plot_range], color="black", label='True Signal')
plt.tick_params(axis='both', which='major', labelsize=14)

# Add labels and title
plt.xlabel('', fontsize=22)
plt.ylabel('', fontsize=22)
plt.ylim([0, 1])
plt.xlim([0, plot_range])
plt.grid(True)

plt.tick_params(axis="both", which="major", labelsize=14)
# Save the figure
plt.tight_layout()
plt.savefig(f'boosting_signals.png', dpi=300, bbox_inches='tight')
plt.show()