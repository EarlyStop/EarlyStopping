for name in list(globals()):
    if not name.startswith("_"):
        del globals()[name]


import numpy as np
import importlib
import EarlyStopping as es
from scipy.sparse import dia_matrix

importlib.reload(es)

# sample_size = 1000
# indices = np.arange(sample_size) + 1

# parameters_supersmooth = es.SimulationParameters(
#     design=dia_matrix(np.diag(1 / np.sqrt(indices))),
#     true_signal=5 * np.exp(-0.1 * indices),
#     true_noise_level=0.01,
#     max_iterations=1000,
# )

sample_size_gravity = 100  # 2**9
a = 0
b = 1
d = 0.25  # Parameter controlling the ill-posedness: the larger, the more ill-posed, default in regtools: d = 0.25

t = (np.arange(1, sample_size_gravity + 1) - 0.5) / sample_size_gravity
s = ((np.arange(1, sample_size_gravity + 1) - 0.5) * (b - a)) / sample_size_gravity
T, S = np.meshgrid(t, s)


parameters_gravity = es.SimulationParameters(
    design=(1 / sample_size_gravity)
    * d
    * (d**2 * np.ones((sample_size_gravity, sample_size_gravity)) + (S - T) ** 2) ** (-(3 / 2)),
    true_signal=np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t),
    true_noise_level=0.01,
    max_iterations=1000,
)

simulation = es.SimulationWrapper(**parameters_gravity.__dict__)

results = simulation.run_simulation()
