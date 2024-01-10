"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import EarlyStopping as es

sample_size = 10
para_size   = 10
cov = np.identity(para_size)
X = np.random.multivariate_normal(np.zeros(para_size), cov, sample_size)
beta = np.ones(sample_size)
f = X @ beta
Y = f
alg = es.L2_boost(X, Y, f)

fig = plt.figure(figsize = (10,7))
plt.plot(beta)
plt.show()