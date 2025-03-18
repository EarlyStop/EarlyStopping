import numpy as np
import matplotlib.pyplot as plt
import EarlyStopping as es
import pandas as pd
import os

# Signals and design
# ------------------------------------------------------------------------------
# From G. Blanchard, M. Hoffmann, M. Reiß. "Early stopping for statistical inverse problems via
# truncated SVD estimation". In: Electronic Journal of Statistics 12(2): 3204-3231 (2018).
sample_size  = 10000
indices      = np.arange(sample_size) + 1
# eigenvalues  = indices**(-0.5)
# design       = np.diag(eigenvalues)

# true_signal_supersmooth = 5    * np.exp(-0.1 * indices)
# true_signal_smooth      = 5000 * np.abs(np.sin(0.01  * indices))  * indices**(-1.6)
# true_signal_rough       = 250  * np.abs(np.sin(0.002 * indices))  * indices**(-0.8)


design, response_noiseless_smooth, true_signal_smooth = es.SimulationData.diagonal_data(
    sample_size=10000, type="smooth"
)
design, response_noiseless_supersmooth, true_signal_supersmooth = es.SimulationData.diagonal_data(
    sample_size=10000, type="supersmooth"
)
design, response_noiseless_rough, true_signal_rough = es.SimulationData.diagonal_data(sample_size=10000, type="rough")

# Signal plot:
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(indices, true_signal_supersmooth, color="blue")
ax.plot(indices, true_signal_smooth, color="purple")
ax.plot(indices, true_signal_rough, color="#CCCC00")
ax.grid(True)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.set_ylim([0, 1.6])
ax.set_xlim([0, 10000])

# Add a single "weak" label beneath the first three x-tick labels
plt.text(2, plt.ylim()[0] - 0.1, " ", ha="center", va="top", fontsize=14)
plt.text(5, plt.ylim()[0] - 0.1, " ", ha="center", va="top", fontsize=14)

fig_dir = ""
plt.savefig(os.path.join(fig_dir, f"signals_new.png"), bbox_inches="tight", dpi=300)
print("saved the figure")