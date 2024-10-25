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



# Fix a threshold:
kappa = 1
# Estimate the regression tree with and apply early stopping:
# The f and the noise_train are given as input to calculate theoretical quantities:
tree_es = RMG_runner.DecisionTreeRegressor(loss='mse', global_es=True,
                                               min_samples_split=1,
                                               kappa=kappa,
                                               noise_vector=noise_train,
                                               signal=f,
                                               design=X_train,
                                               response=y_train)

# Gives ES iteration; max_depth und kappa hier rein (dafür oben kappa weglassen und X_train, y_train oben rein in der object Erzeugung in welcher der contructor aufgerufen wird):
tree_es.iterate(X_train, y_train, max_depth=3000) # TODO: analog zu landweber anpassen; evtl später 'übernamung' implementieren mit .train
# TODO: .get_discrepancy_stop ist das analog in landweber
# TODO: Do it as close to landweber as possible.
# TODO: minimal_example.py zu notebooks tun (NICHT zu Source!)

tree_es.bias2_collect
tree_es.variance_collect
# For theretical quantities; baut auf dem selben object auf :
# something like "tree_es.run_to_iteration(iteration =XX)"


stop = tree_es.stopping_index # 6
balanced_oracle = tree_es.get_balanced_oracle_iteration # 4


print('the early stopping iteration is:', stop)
print('the balanced oracle iteration is:', balanced_oracle)

# Let us do prediction on the test set at the ES iteration:

prediction = tree_es.predict(X_test, depth=stop+10)
