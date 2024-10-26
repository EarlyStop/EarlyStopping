for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]

import importlib
import numpy as np
# Early Stopping and tree growing (breadth-first search):
from src.EarlyStopping import regression_tree as regression_tree

# Generate the data:
from examples import data_generation_regression_tree
# I Reload the function, in case I make some changes:
importlib.reload(data_generation_regression_tree)
importlib.reload(regression_tree)


# Setting:
n_train = 1000 # sample size
d = 5 # dimension
noise_level = 1
X_train = np.random.uniform(0, 1, size=(n_train, d))
X_test = np.random.uniform(0, 1, size=(n_train, d))

y_train, noise_train = data_generation_regression_tree.generate_data_from_X(X_train, noise_level, dgp_name='rectangular', add_noise=True)
y_test, noise_test = data_generation_regression_tree.generate_data_from_X(X_test, noise_level, dgp_name='rectangular', add_noise=True)
f, nuisance = data_generation_regression_tree.generate_data_from_X(X_train, noise_level=noise_level, dgp_name='rectangular', add_noise=False)



es_regression_tree = regression_tree.DecisionTreeRegressor(design=X_train, response=y_train,
                                                           min_samples_split=1,
                                                           true_signal=f,
                                                           true_noise_vector=noise_train)

# Iterate until max_depth! theoretical quantities are saved automatically.
es_regression_tree.iterate(X_train, y_train, max_depth=20)


#theoretical quantities:
es_regression_tree.bias2
es_regression_tree.variance
es_regression_tree.risk
es_regression_tree.residuals

#Get the stopping iterations:
tau = es_regression_tree.get_discrepancy_stop(critical_value=0.8)
balanced_oracle_iteration = es_regression_tree.get_balanced_oracle()

prediction = es_regression_tree.predict(X_train, depth=tau)


