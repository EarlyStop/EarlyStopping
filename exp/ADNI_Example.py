# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import EarlyStopping as es

#|%%--%%| <RTFTaisXbt|CBv7JXAUOg>
# Importing data and merging
mmse_data    = pd.read_csv("/home/be5tan/Projects/EarlyStopping/exp/data/MMSE_04Oct2025.csv")
mri_roi_data = pd.read_csv("/home/be5tan/Projects/EarlyStopping/exp/data/UCSFFSX7_04Oct2025.csv")

# Merge MMSCORE from df2 into df1 only for rows where RID, VISCODE, and VISCODE2 match
merged_data = pd.merge(
    mri_roi_data, 
    mmse_data[['RID', 'VISCODE', 'VISCODE2', 'MMSCORE']],  # only keep relevant columns from df2
    on=['RID', 'VISCODE', 'VISCODE2'],                     # merge on these columns
    how='inner'                                            # keep only rows that match
)

RID2_test_data = merged_data[merged_data["RID"] == 2]
print(RID2_test_data[['RID', 'VISCODE', 'VISCODE2', 'MMSCORE']])  # Correct values are: 2, sc, sc, 28

# merged_data.to_csv("merged_data.csv", index=False)
#|%%--%%| <CBv7JXAUOg|7zHp5GZPFs>
# Minimal boosting example
min_example_data = merged_data[merged_data["PHASE"] == "ADNI1"]
min_example_data = min_example_data.drop_duplicates(subset=["RID"], keep = "first")
min_example_data.info()
min_example_data.to_csv("min_example_data.csv", index=False)

response                 = min_example_data["MMSCORE"].to_numpy()
first_covariate_location = min_example_data.columns.get_loc("ST101SV")
last_covariate_location  = min_example_data.columns.get_loc("ST155SV")
design                   = min_example_data.iloc[:, first_covariate_location:last_covariate_location].to_numpy()
design                   = np.nan_to_num(design, nan = 0)

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
