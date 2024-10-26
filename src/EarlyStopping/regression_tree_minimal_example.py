for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]

import importlib
import numpy as np
# Early Stopping and tree growing (breadth-first search):
from src.EarlyStopping import regression_tree as RMG_runner

# Generate the data:
from src.EarlyStopping import data_generation
# I Reload the function, in case I make some changes:
importlib.reload(data_generation)
importlib.reload(RMG_runner)


# Setting:
n_train = 1000
d = 5
noise_level = 1
X_train = np.random.uniform(0, 1, size=(n_train, d))
X_test = np.random.uniform(0, 1, size=(n_train, d))

y_train, noise_train = data_generation.generate_data_from_X(X_train, noise_level, dgp_name='rectangular', n_points=1,
                                                            add_noise=True)
y_test, noise_test = data_generation.generate_data_from_X(X_test, noise_level, dgp_name='rectangular', n_points=1,
                                                          add_noise=True)
f, nuisance = data_generation.generate_data_from_X(X_train, noise_level=noise_level, dgp_name='rectangular', n_points=1,
                                                   add_noise=False)



tree_es = RMG_runner.DecisionTreeRegressor(loss='mse', global_es=True,
                                               min_samples_split=1,
                                               noise_vector=noise_train,
                                               signal=f,
                                               design=X_train,
                                               response=y_train)

tree_es.iterate(X_train, y_train, max_depth=3000)

#theoretical quantities:
# TODO: Would be nicer if this is numpy array and not some weird dictionary
tree_es.bias2_collect
tree_es.variance_collect

# TODO: think about it some more. Are these two generations refer to the same logic? Decide for a logic that should be used (generation iteration vs. splitting iteration).
tree_es.get_discrepancy_stop(critical_value=0.8)
tree_es.get_balanced_oracle_iteration

prediction = tree_es.predict(X_test, depth=20)
