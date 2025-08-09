# Importing libraries -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import EarlyStopping as es


# Processing data ---------------------------------------------------------------------------------

data = pd.read_csv("/home/be5tan/Projects/EarlyStopping/data/riboflavin_dataset.csv")
data.insert(1, "constant", 1)
training_data, validation_data = train_test_split(data, test_size=11, random_state=1)
# random_state = 100 for very U-shaped out of sample mse

training_response = training_data.iloc[:, 0].to_numpy()
training_design   = training_data.iloc[:, 1:].to_numpy()

validation_response = validation_data.iloc[:, 0].to_numpy()
validation_design   = validation_data.iloc[:, 1:].to_numpy()


# Estimation procedures ---------------------------------------------------------------------------  

alg = es.L2_boost(training_design, training_response)
alg.iterate(50)

noise_estimate             = alg.get_noise_estimate(K = 0.5)
noise_estimate
discrepancy_stopping_time  = alg.get_discrepancy_stop(critical_value = noise_estimate, max_iteration=100)
discrepancy_stopping_time

residual_ratio_stopping_time = alg.get_residual_ratio_stop(max_iteration=200, K=0.2)
residual_ratio_stopping_time

aic_minimizer = alg.get_aic_iteration(K=2)
aic_minimizer


# Out of sample prediction mse ------------------------------------------------------------------

# plot the out of sample mse along the iterations
out_of_sample_mse = np.zeros(50)
for iteration in range(50):
    out_of_sample_mse[iteration] = np.mean((validation_response - alg.predict(validation_design, iteration))**2)

fig, ax = plt.subplots()
ax.plot(np.arange(50), out_of_sample_mse)
ax.set_ylim(0, 2)
plt.show()


# Comparison w/ the cross validated Lasso ----------------------------------------------------------

lasso_cv = linear_model.LassoCV(fit_intercept = False)
lasso_cv.fit(training_design, training_response)
lasso_cv_mse = np.mean((validation_response - lasso_cv.predict(validation_design))**2)
lasso_cv_mse
