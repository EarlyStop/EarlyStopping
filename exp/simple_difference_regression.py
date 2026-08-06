"""
In this file, we apply the EarlyStopping package to a simple difference regression problem, based
on the ADNI study. More specifically, we used the preprocessed data obtained from the DataProcessing.py script gathered from the
original source files:

- MMSE_15Jul2026.csv
- UCSFFSX7_15OJul2026.csv

downloaded from the ADNI Database via IDA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es
import warnings

warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data_06 = pd.read_csv("data/ADNI_I_m06_data.csv")
data_36 = pd.read_csv("data/ADNI_I_m36_data.csv")

# Keep only common RIDs at both timepoints m06 and m36 and align both frames by RID.
# RID stands for Roster ID. Each participant has one RID, reused across their visits and across ADNI tables.
common_rids = data_06["RID"].isin(data_36["RID"])
data_06_reduced = data_06.loc[common_rids].set_index("RID")
common_rids = data_36["RID"].isin(data_06["RID"])
data_36_reduced = data_36.loc[common_rids].set_index("RID")

# Align both datasets with respect to the intersection in identical order.
# This ensures that the patients occur in both datasets in the same sequence.
data_06_reduced, data_36_reduced = data_06_reduced.align(data_36_reduced, join="inner", axis=0)

# Extract the MMSCORE as a numpy array and construct the response variable
response_06 = data_06_reduced["MMSCORE"].to_numpy()
response_36 = data_36_reduced["MMSCORE"].to_numpy()
response_diff = response_36 - response_06

# Extract features for m06 in order to gather all the features from the datasets into design matrices
first_covariate_location = data_06_reduced.columns.get_loc("ST101SV")  # Get first design column number
last_covariate_location = data_06_reduced.columns.get_loc("ST155SV")  # Get last design column number
design_06 = data_06_reduced.iloc[:, first_covariate_location:last_covariate_location]
design_06 = design_06.fillna(
    design_06.mean(numeric_only=True)
)  # Replace missing values with the mean of the respective column
design_06 = design_06.to_numpy()

# Extract features for m36 in order to gather all the features from the datasets into design matrices
first_covariate_location = data_36_reduced.columns.get_loc("ST101SV")
last_covariate_location = data_36_reduced.columns.get_loc("ST155SV")
design_36 = data_36_reduced.iloc[:, first_covariate_location:last_covariate_location]
design_36 = design_36.fillna(
    design_36.mean(numeric_only=True)
)  # Replace missing values with the mean of the respective column
design_36 = design_36.to_numpy()
design_diff = design_36 - design_06


def evaluate_train_test_split(input_seed):
    """
    Step 1: Based on the given input_seed create a train_test_split based on design_diff and response_diff.
    Step 2: Train LassoCV and different early stopping methods on the training data
    Step 3: Determine which features are selected by the current method and determine what is contained within the intersection
    """
    # Split the data
    design_diff_train, design_diff_test, response_diff_train, response_diff_test = train_test_split(
        design_diff, response_diff, test_size=0.2, random_state=input_seed
    )

    # Compute the crossvalidated lasso on the training data
    lassoCV = linear_model.LassoCV(fit_intercept=False)
    lassoCV.fit(design_diff_train, response_diff_train)
    selected_mask = lassoCV.coef_ != 0

    # Identifying the selected covariates by label
    data_36_reduced_model_components = data_36_reduced.iloc[:, first_covariate_location:last_covariate_location]
    lasso_selected_covariates = data_36_reduced_model_components.columns[selected_mask]

    alg = es.L2_boost(design_diff_train, response_diff_train)
    alg.iterate(50)

    # Compute the different stopping times
    # Discrepancy stop
    noise_estimate = alg.get_noise_estimate(K=1)
    discrepancy_stopping_time = alg.get_discrepancy_stop(critical_value=noise_estimate, max_iteration=300)
    residual_ratio_stopping_time = alg.get_residual_ratio_stop(max_iteration=300, K=0.25)
    two_step_stopping_time = alg.get_aic_iteration(K=2)

    # Selected components from the different stopping rules
    # discrepancy stop
    data_36_reduced_selected_components_discrepancy_stopping_time = data_36_reduced_model_components.iloc[
        :, alg.selected_components[0:discrepancy_stopping_time]
    ]
    boost_selected_covariates_discrepancy_stopping_time = (
        data_36_reduced_selected_components_discrepancy_stopping_time.columns
    )

    # residual ratio
    data_36_reduced_selected_components_residual_ratio_stopping_time = data_36_reduced_model_components.iloc[
        :, alg.selected_components[0:residual_ratio_stopping_time]
    ]
    boost_selected_covariates_residual_ratio_stopping_time = (
        data_36_reduced_selected_components_residual_ratio_stopping_time.columns
    )

    # two step
    data_36_reduced_selected_components_two_step_stopping_time = data_36_reduced_model_components.iloc[
        :, alg.selected_components[0:two_step_stopping_time]
    ]
    boost_selected_covariates_two_step_stopping_time = (
        data_36_reduced_selected_components_two_step_stopping_time.columns
    )

    # Intersections with the selected covariates from LassoCV
    intersection_discrepancy_stopping_time = list(
        set(boost_selected_covariates_discrepancy_stopping_time) & set(lasso_selected_covariates)
    )
    intersection_residual_ratio_stopping_time = list(
        set(boost_selected_covariates_residual_ratio_stopping_time) & set(lasso_selected_covariates)
    )
    intersection_two_step_stopping_time = list(
        set(boost_selected_covariates_two_step_stopping_time) & set(lasso_selected_covariates)
    )

    # Stopping times
    print(f"Number of non zero features of the lasso: {len(lasso_selected_covariates)}")
    print(f"Stopping time (discrepancy): {discrepancy_stopping_time}")
    print(f"Stopping time (residual ratio): {residual_ratio_stopping_time}")
    print(f"Stopping time (two step): {two_step_stopping_time}\n")

    # MSE
    print(f"LassoCV predict MSE: {np.mean((response_diff_test - lassoCV.predict(design_diff_test)) ** 2)}")
    print(
        f"L2Boost discrepancy stop MSE: {np.mean((response_diff_test - alg.predict(design_diff_test, discrepancy_stopping_time))**2)}"
    )
    print(
        f"L2Boost residual ratio stop MSE: {np.mean((response_diff_test - alg.predict(design_diff_test, residual_ratio_stopping_time))**2)}"
    )
    print(
        f"L2Boost two step procedure MSE: {np.mean((response_diff_test - alg.predict(design_diff_test, two_step_stopping_time))**2)}\n"
    )

    # Selected features
    print(f"Lasso selected covariates: {lasso_selected_covariates}\n")
    print(f"Boost (Discrepancy stop) selected covariates: {boost_selected_covariates_discrepancy_stopping_time}")
    print(f"Boost (Residual ratio stop) selected covariates: {boost_selected_covariates_residual_ratio_stopping_time}")
    print(f"Boost (Two step) selected covariates: {boost_selected_covariates_two_step_stopping_time}")

    # Intersections
    print(f"Intersection discrepancy stopping time: {intersection_discrepancy_stopping_time}")
    print(f"Intersection residual ratio stopping time: {intersection_residual_ratio_stopping_time}")
    print(f"Intersection two step stopping time: {intersection_two_step_stopping_time}\n")


# Execute the procedure
for seed in range(2):
    print(f"Seed: {seed}\n")
    evaluate_train_test_split(seed)
