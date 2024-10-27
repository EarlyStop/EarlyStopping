"""
Usage example for global early stopping for the regression tree
======================================

We illustrate the usage and available methods of the regression tree class via a
small example.
"""

for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]

import importlib
import numpy as np
import EarlyStopping as es
import examples.data_generation_regression_tree as data_generation
importlib.reload(es)
importlib.reload(data_generation)


# %%
# Generating synthetic data
# -------------------------
# To simulate some data we consider the processes from Miftachov and Reiß (2024).
sample_size = 1000
para_size = 5
true_noise_level = 1
X_train = np.random.uniform(0, 1, size=(sample_size, para_size))
X_test = np.random.uniform(0, 1, size=(sample_size, para_size))

response_train, noise_train = data_generation.generate_data_from_X(X_train, true_noise_level, dgp_name='rectangular',
                                                                   add_noise=True)
response_test, noise_test = data_generation.generate_data_from_X(X_test, true_noise_level, dgp_name='rectangular',
                                                                 add_noise=True)
f, _ = data_generation.generate_data_from_X(X_train, noise_level = true_noise_level, dgp_name='rectangular',
                                            add_noise=False)


# %%
# Initialize RegressionTree class
alg = es.RegressionTree(design=X_train, response=response_train, min_samples_split=1, true_signal=f,
                        true_noise_vector=noise_train)
alg.iterate(max_depth=20)

# Bias-variance decomposition and oracle quantities
alg.bias2
alg.variance
# TODO: Why is risk and residuals the same ?
alg.risk
alg.residuals

# Early stopping w/ discrepancy principle
tau = alg.get_discrepancy_stop(critical_value=1)
balanced_oracle_iteration = alg.get_balanced_oracle()

# Prediction
prediction = alg.predict(X_train, depth=10)


