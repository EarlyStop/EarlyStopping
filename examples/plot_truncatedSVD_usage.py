"""
Usage and methods of the TruncatedSVD class 
===========================================


We illustrate the usage and available methods of the TruncatedSVD class via a
small example.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es

np.random.seed(42)
sns.set_theme()

# %%
# Generating synthetic data
# -------------------------
# To simulate some data we consider the signals from `Blanchard, Hoffmann and Reiß (2018) <https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-12/issue-2/Early-stopping-for-statistical-inverse-problems-via-truncated-SVD-estimation/10.1214/18-EJS1482.full>`_.
sample_size = 10000
indices     = np.arange(sample_size) + 1

signal_supersmooth = 5    * np.exp(-0.1 * indices)
signal_smooth      = 5000 * np.abs(np.sin(0.01  * indices))  * indices**(-1.6)
signal_rough       = 250  * np.abs(np.sin(0.002 * indices))  * indices**(-0.8)

plt.figure(figsize=(10, 4))
plt.xlabel("Index")
plt.xlim([0, 5000])
plt.ylabel("Signal")
plt.ylim([0, 1.6])
plt.plot(indices, signal_supersmooth, label="supersmooth signal")
plt.plot(indices, signal_smooth, label="smooth signal")
plt.plot(indices, signal_rough, label="rough signal")
plt.legend(loc="upper right")
plt.show()

# %%
# We simulate data from a prototypical inverse problem based on one of the signals
true_signal      = signal_rough
eigenvalues      = indices**(-0.5)
design           = np.diag(eigenvalues)
true_noise_level = 0.01
response         = eigenvalues * true_signal + \
                   true_noise_level * np.random.normal(0, 1, sample_size)

# %%
# Theoretical bias-variance decomposition
# ---------------------------------------
# By giving the true function f to the class, we can track the theoretical bias-variance decomposition and the balanced oracle.
alg = es.TruncatedSVD(design, response, true_signal, true_noise_level,
                      diagonal = True)
alg.iterate(3000)

plt.figure()
plt.plot(indices[0: alg.iteration + 1], alg.weak_variance, label="Weak variance")
plt.plot(indices[0: alg.iteration + 1], alg.weak_bias2, label="Weak squared bias")
plt.legend(loc="upper right")
plt.show()
weak_balanced_oracle = alg.get_weak_balanced_oracle(3000)
weak_balanced_oracle_mse = alg.weak_bias2[weak_balanced_oracle] + alg.weak_variance[weak_balanced_oracle]
print(f"The weakly balanced oracle is given by {weak_balanced_oracle} with mse = {weak_balanced_oracle_mse}.")

# plt.figure()
# plt.plot(indices[0: alg.iteration + 1], alg.strong_variance, label="Strong variance")
# plt.plot(indices[0: alg.iteration + 1], alg.strong_bias2, label="Strong squared bias")
# plt.legend(loc="upper right")
# strong_balanced_oracle = alg.get_strong_balanced_oracle(3000)
# strong_balanced_oracle_mse = alg.strong_bias2[strong_balanced_oracle] + alg.strong_variance[strong_balanced_oracle]
# print(f"The strongly balanced oracle is given by {strong_balanced_oracle} with mse = {strong_balanced_oracle_mse}.")
# 
# # %%
# # Early stopping via the discrepancy principle
# # --------------------------------------------
# # The TruncatedSVD class provides a data driven method to choose an iteration making the right tradeoff between bias and variance. 
# # It is based on the discrepancy principle, which stops when the residuals become smaller than a critical value.
# # Theoretically this critical value should be chosen depending on the noise level of the model, which, in the inverse problem setting, can be assumed to be known.
# critical_value   = sample_size * true_noise_level**2
# stopping_time    = alg.get_discrepancy_stop(critical_value, 3000)
# estimated_signal = alg.get_estimate(stopping_time)
# early_stopping_mse = alg.strong_bias2[stopping_time] + alg.strong_variance[stopping_time]
# plt.figure(figsize=(10, 5))
# plt.plot(indices, estimated_signal)
# plt.plot(indices, true_signal)
# plt.ylim([0, 2])
# plt.xlim([0, 5000])
# print(f"The discrepancy based early stopping time is given by {stopping_time} with mse = {early_stopping_mse}.")
# 
# # %%
# # A two-step procedure
# # ----------------------
# # Via a classical Akaike criterion, the alforithm can also be run with a two-step procedure, i.e. an additional model selection step up to the stopping time.
# aic = alg.get_aic(stopping_time)
# aic_time = aic[0]
# aic_mse = alg.strong_bias2[aic_time] + alg.strong_variance[aic_time]
# print(f"The AIC based stopping time is given by {aic_time} with mse = {aic_mse}.")
# # This confirms the validity of the discrepancy stopping time for this particular signal.
