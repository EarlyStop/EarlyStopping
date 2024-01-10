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

# %%
# Section 1 
# ------------------------
# This is the first section!
# The `#%%` signifies to Sphinx-Gallery that this text should be rendered as
# reST and if using one of the above IDE/plugin's, also signifies the start of a
# 'code block'.

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

# %%
# Section 2
# ------------------------
# This is the first section!
# The `#%%` signifies to Sphinx-Gallery that this text should be rendered as
# reST and if using one of the above IDE/plugin's, also signifies the start of a
# 'code block'.

sample_size = 10
para_size   = 10
cov = np.identity(para_size)
X = np.random.multivariate_normal(np.zeros(para_size), cov, sample_size)
beta = 2*np.ones(sample_size)
f = X @ beta
Y = f
alg = es.L2_boost(X, Y, f)

fig = plt.figure(figsize = (10,7))
plt.plot(beta)
plt.show()
