"""
Usage of the RegressionTree class
===============================================================

We illustrate the usage and available methods of the regression tree class via a
small example.
"""

for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]

import importlib
import numpy as np
import EarlyStopping as es
importlib.reload(es)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()



# %%
# Generating synthetic data
# -------------------------
# To simulate some data we consider the processes from Miftachov and Reiß (2024).
sample_size = 1000
para_size = 5
true_noise_level = 1
design = np.random.uniform(0, 1, size=(sample_size, para_size))

def generate_rectangular(design, noise_level, add_noise=True):
    sample_size = design.shape[0]
    if add_noise:
        noise = np.random.normal(0, noise_level, sample_size)
    else:
        noise = np.zeros(sample_size)

    response_temp = ((1 / 3 <= design[:, 0]) * (design[:, 0] <= 2 * 1 / 3) * (1 / 3 <= design[:, 1]) * (design[:, 1] <= 2 * 1 / 3))
    response = response_temp.astype(int) + noise

    return response, noise

def generate_smooth_signal(design, noise_level, add_noise=True):
    sample_size = design.shape[0]
    if add_noise:
        noise = np.random.normal(0, noise_level, sample_size)
    else:
        noise = np.zeros(sample_size)

    y = 20 * np.exp(-5 * ((design[:, 0] - 1 / 2) ** 2 + (design[:, 1] - 1 / 2) ** 2 - 0.9 * (design[:, 0] - 1 / 2) * (design[:, 1] - 1 / 2))) + noise

    return y, noise

def generate_sine_cosine(design, noise_level, add_noise=True):

    sample_size = design.shape[0]
    if add_noise:
        noise = np.random.normal(0, noise_level, sample_size)
    else:
        noise = np.zeros(sample_size)
    y = np.sin(3 * np.pi * design[:, 0]) + np.cos(5 * np.pi * design[:, 1]) + noise
    return y, noise

response, noise = generate_sine_cosine(design, true_noise_level, add_noise=True)
f, _ = generate_sine_cosine(design, true_noise_level, add_noise=False)


# %%
# Bias-variance decomposition
# ------------------------------
# Decompose the risk into squared bias and variance.
alg = es.RegressionTree(design=design, response=response, min_samples_split=1, true_signal=f,
                        true_noise_vector=noise)
alg.iterate(max_depth=30)

plt.figure(figsize=(8, 6))
plt.plot(alg.risk, label='Risk')
plt.plot(alg.bias2, label='Bias')
plt.plot(alg.variance, label='Variance')
plt.xlabel('Generation')
plt.grid(True)
x_ticks = np.arange(0, len(alg.risk) + 1, step=3)  # Adjust 'step' for tick frequency
plt.xticks(x_ticks)
plt.show()



# %%
# Early stopping via the discrepancy principle
# --------------------------------------------------
# Stop the breadth-first search growth of the tree when the residuals become smaller than the critical value.
stopping_iteration = alg.get_discrepancy_stop(critical_value=1, max_depth=10)
balanced_oracle_iteration = alg.get_balanced_oracle(max_depth=20)
print("The discrepancy based early stopping generation is given by", stopping_iteration)
print("The balanced oracle generation is given by", balanced_oracle_iteration)


# %%
# Prediction
# ------------
# The class has a method to predict the response for new design points.
prediction = alg.predict(design, depth=20)

