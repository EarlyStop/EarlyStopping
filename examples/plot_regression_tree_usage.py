"""
Usage example for global early stopping for the regression tree
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


# %%
# Generating synthetic data
# -------------------------
# To simulate some data we consider the processes from Miftachov and Reiß (2024).
sample_size = 1000
para_size = 5
true_noise_level = 1
X_train = np.random.uniform(0, 1, size=(sample_size, para_size))
X_test = np.random.uniform(0, 1, size=(sample_size, para_size))

def generate_rectangular(X, noise_level, add_noise=True):
    n = X.shape[0]
    if add_noise:
        noise = np.random.normal(0, noise_level, n)
    else:
        noise = np.zeros(n)

    # For X uniform:
    y_temp = ((1 / 3 <= X[:, 0]) * (X[:, 0] <= 2 * 1 / 3) * (1 / 3 <= X[:, 1]) * (X[:, 1] <= 2 * 1 / 3))

    y = y_temp.astype(int) + noise
    return y, noise



response_train, noise_train = generate_rectangular(X_train, true_noise_level, add_noise=True)
response_test, noise_test = generate_rectangular(X_test, true_noise_level, add_noise=True)
f, _ = generate_rectangular(X_train, true_noise_level, add_noise=False)


# %%
# Initialize RegressionTree class
alg = es.RegressionTree(design=X_train, response=response_train, min_samples_split=1, true_signal=f,
                        true_noise_vector=noise_train)
alg.iterate(max_depth=20)

# Bias-variance decomposition and oracle quantities
alg.bias2
alg.variance
alg.risk
alg.residuals

# Early stopping w/ discrepancy principle
tau = alg.get_discrepancy_stop(critical_value=1)
balanced_oracle_iteration = alg.get_balanced_oracle()

# Prediction
prediction = alg.predict(X_train, depth=10)


