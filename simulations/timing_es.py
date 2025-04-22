import timeit
import numpy as np
import importlib
import EarlyStopping as es
import matplotlib.pyplot as plt
import os


# Ensure consistent style - using the style from error_decomposition_plots.py
plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=15)
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)

importlib.reload(es)

sample_size = 1000


# design, response_noiseless, true_signal = es.SimulationData.gravity(sample_size=sample_size)
# design, response_noiseless, true_signal = es.SimulationData.phillips(sample_size=sample_size)
design, response_noiseless, true_signal = es.SimulationData.diagonal_data(sample_size=sample_size, type="rough")

true_noise_level = 1 / 10
noise = true_noise_level * np.random.normal(0, 1, sample_size)
response = response_noiseless + noise

model_svd = es.TruncatedSVD(
    design, response, diagonal=True
)  # , true_signal=true_signal, true_noise_level=true_noise_level
model_landweber = es.Landweber(design, response, learning_rate=1 / 100)
model_cg = es.ConjugateGradients(design, response)


def time_svd():
    model_svd.iterate(1000)


def time_landweber():
    model_landweber.iterate(1000)


def time_cg():
    model_cg.iterate(1000)


execution_time_cg = timeit.timeit(time_cg, number=1) / 10
execution_time_landweber = timeit.timeit(time_landweber, number=1) / 10
execution_time_svd = timeit.timeit(time_svd, number=1) / 10


print(f"Execution time cg: {execution_time_cg:.6f} seconds")
print(f"Execution time landweber: {execution_time_landweber:.6f} seconds")
print(f"Execution time svd: {execution_time_svd:.6f} seconds")
