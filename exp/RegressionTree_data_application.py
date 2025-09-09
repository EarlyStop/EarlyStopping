import numpy as np
np.random.seed(21)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import sys
import os
from sklearn.tree import DecisionTreeRegressor


# Add the src directory to path to import EarlyStopping
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import EarlyStopping as es

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Fetch the concrete dataset from OpenML
concrete = fetch_openml(data_id=4353, as_frame=True, parser='auto')

# The target variable is included in the data DataFrame as the last column
# Extract features (all columns except the last one)
design = concrete.data.iloc[:, :-1]  # All columns except the last
# Extract target (the last column - concrete compressive strength)
response = concrete.data.iloc[:, -1]  # Last column

# Convert to numpy arrays
design_array = design.to_numpy()
response_array = response.to_numpy()

# Split into train and test sets 
design_train, design_test, response_train, response_test = train_test_split(
    design_array, response_array, test_size=0.2, random_state=21
)

def estimate_1NN(design, response):
        """
        Estimate using the 1NN method described by Devroye et al. (2018).

        Returns:
        float: The 1NN estimator value.
        """
        nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
        nn.fit(design)
        distances, indices = nn.kneighbors(design)
        NN = indices[:, 1]
        m_1 = response[NN]
        n = len(response)
        S = np.dot(response, m_1) / n
        EY = np.dot(response, response) / n
        L = EY - S
        return L

# Estimate noise level using 1NN method
kappa = estimate_1NN(design_train, response_train)

regression_tree = es.RegressionTree(
        design=design_train, response=response_train, min_samples_split=1
    )
regression_tree.iterate(max_depth=12)
early_stopping_iteration = regression_tree.get_discrepancy_stop(critical_value=kappa)

# Global ES prediction
prediction_global_k1 = regression_tree.predict(design_test, depth=7)
mse_global = np.mean((prediction_global_k1 - response_test) ** 2)

# Deep tree prediction (for comparison)
deep_tree = DecisionTreeRegressor(random_state=21)
deep_tree.fit(design_train, response_train)
prediction_deep_tree = deep_tree.predict(design_test)
mse_deep_tree = np.mean((prediction_deep_tree - response_test) ** 2)

print(f"MSE of global ES: {mse_global:.4f}")
print(f"MSE of deep tree: {mse_deep_tree:.4f}")