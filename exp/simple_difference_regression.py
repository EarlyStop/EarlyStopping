import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es

data_06 = pd.read_csv("data/ADNI_I_m06_data.csv")
data_36 = pd.read_csv("data/ADNI_I_m36_data.csv")

# keep only common RIDs and align both frames by RID
common_rids = data_06["RID"].isin(data_36["RID"])
data_06_reduced = data_06.loc[common_rids].set_index("RID")

common_rids = data_36["RID"].isin(data_06["RID"])
data_36_reduced = data_36.loc[common_rids].set_index("RID")

# align to the intersection, in identical order
data_06_reduced, data_36_reduced = data_06_reduced.align(data_36_reduced, join="inner", axis=0)

response_06 = data_06_reduced["MMSCORE"].to_numpy()
response_36 = data_36_reduced["MMSCORE"].to_numpy()
response_diff = response_36 - response_06

# Extract features
first_covariate_location = data_06_reduced.columns.get_loc("ST101SV")
last_covariate_location = data_06_reduced.columns.get_loc("ST155SV")

design_06 = data_06_reduced.iloc[:, first_covariate_location:last_covariate_location]
design_06 = design_06.fillna(design_06.mean(numeric_only=True)) # Replace missing values with the mean of the respective column
design_06 = design_06.to_numpy()

first_covariate_location = data_36_reduced.columns.get_loc("ST101SV")
last_covariate_location = data_36_reduced.columns.get_loc("ST155SV")

design_36 = data_36_reduced.iloc[:, first_covariate_location:last_covariate_location]
design_36 = design_36.fillna(design_36.mean(numeric_only=True)) # Replace missing values with the mean of the respective column
design_36 = design_36.to_numpy()

design_diff = design_36 - design_06

alg = es.L2_boost(design_diff, response_diff)
alg.iterate(300)

# Discrepancy stop
noise_estimate = alg.get_noise_estimate(K = 100)
print(f"Noise estimate: {noise_estimate}")

stopping_time  = alg.get_discrepancy_stop(critical_value = noise_estimate, max_iteration=300)
print(f"Stopping time (discrepancy): {stopping_time}")

# Early stopping via residual ratios
stopping_time = alg.get_residual_ratio_stop(max_iteration=200, K=1.2)
print(f"Stopping time (residual ratio): {stopping_time}")

stopping_time = alg.get_residual_ratio_stop(max_iteration=200, K=0.2)
print(f"Stopping time (residual ratio): {stopping_time}")

stopping_time = alg.get_residual_ratio_stop(max_iteration=200, K=0.1)
print(f"Stopping time (residual ratio): {stopping_time}")

# Classical model selection via AIC
aic_minimizer = alg.get_aic_iteration(K=2)
print(f"Best model (AIC): {aic_minimizer}")
