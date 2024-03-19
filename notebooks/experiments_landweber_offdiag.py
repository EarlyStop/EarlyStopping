

import matplotlib.pyplot as plt
import numpy as np
import EarlyStopping as es
from scipy.sparse import dia_matrix
import cProfile
import pstats



#indices = np.arange(D) + 1
#design_matrix = dia_matrix(np.diag(1 / (np.sqrt(indices))))


D = 100
normal_matrix = np.random.normal(0, 0.1, size=(D, D))
indices = np.arange(D) + 1
signal_supersmooth = 5 * np.exp(-0.1 * indices)

indices = np.arange(1, D + 1)
diagonal_values = 1 / np.sqrt(indices)

np.fill_diagonal(normal_matrix, diagonal_values)
design_matrix = normal_matrix


NOISE_LEVEL = 0.01
noise = np.random.normal(0, NOISE_LEVEL, D)

observation_supersmooth = noise + design_matrix @ signal_supersmooth


models_supersmooth = es.Landweber(
    design_matrix, observation_supersmooth, true_noise_level=NOISE_LEVEL, true_signal=signal_supersmooth
)


iter = 100
models_supersmooth.landweber_gather_all(iter)
models_supersmooth.early_stopping_index
models_supersmooth.design


supersmooth_m = models_supersmooth.early_stopping_index
supersmooth_weak_oracle = models_supersmooth.weak_balanced_oracle
supersmooth_strong_oracle = models_supersmooth.strong_balanced_oracle

print(supersmooth_m)
print(models_supersmooth.landweber_estimate)

#Profiling:
#cProfile.run('models_supersmooth.landweber_gather_all(iter)', filename='profile_results.prof')

# Analyze the profiling results using pstats
#stats = pstats.Stats('profile_results.prof')
#stats.strip_dirs()  # Strip unnecessary path information
#stats.sort_stats('cumulative')  # Sort the profiling results by cumulative time
#stats.print_stats()  # Print the profiling statistics

