"""
A minimal example for the ADNI application
==========================================
"""
import os
import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es

password = os.environ.get("PASSWORD")

result = subprocess.run(
    [
        "gpg",
        "--batch",
        "--yes",
        "--pinentry-mode", "loopback",           # <-- important in CI
        "--passphrase", 
        password,
        "--decrypt", 
        "data/ADNI_data.gpg",
    ],
    capture_output=True,
    check=True,
    text=True,  # so result.stdout is already str
)

merged_data = pd.read_csv(StringIO(result.stdout))

# Boosting example for PHASE = ADNI1
min_example_data = merged_data[merged_data["PHASE"] == "ADNI1"]
min_example_data = min_example_data.drop_duplicates(subset=["RID"], keep = "first")
min_example_data.info()
min_example_data.to_csv("min_example_data.csv", index=False)

response                 = min_example_data["MMSCORE"].to_numpy()
np.isnan(response).any()

first_covariate_location = min_example_data.columns.get_loc("ST101SV")
last_covariate_location  = min_example_data.columns.get_loc("ST155SV")
design                   = min_example_data.iloc[:, first_covariate_location:last_covariate_location].to_numpy()
design                   = np.nan_to_num(design, nan = 0)
np.isnan(response).any()

alg = es.L2_boost(design, response)
alg.iterate(300)

# Discrepancy stop
noise_estimate = alg.get_noise_estimate(K = 1)
stopping_time  = alg.get_discrepancy_stop(critical_value = noise_estimate, max_iteration=300)
stopping_time

# Early stopping via residual ratios
stopping_time = alg.get_residual_ratio_stop(max_iteration=200, K=1.2)
stopping_time

stopping_time = alg.get_residual_ratio_stop(max_iteration=200, K=0.2)
stopping_time

stopping_time = alg.get_residual_ratio_stop(max_iteration=200, K=0.1)
stopping_time

# Classical model selection via AIC
aic_minimizer = alg.get_aic_iteration(K=2)
aic_minimizer
