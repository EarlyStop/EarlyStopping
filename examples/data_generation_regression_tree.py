"""
This module generates data for the regression tree example.

It provides functions to create synthetic datasets that can be used
to test and visualize the performance of regression trees in an
early stopping context.
"""

import numpy as np

def generate_data_from_X(X, noise_level, dgp_name, n_points=None, add_noise=True):

    n = X.shape[0]
    if add_noise:
        noise = np.random.normal(0, noise_level, n)

    else:

        noise = np.zeros(n)

    if dgp_name == 'sine_cosine':
        return generate_sine_cosine(X, noise_level, n_points, noise)
    elif dgp_name == 'rectangular':
        return generate_rectangular(X, noise_level, n_points, noise)
    elif dgp_name == 'circular':
        return generate_circular(X, noise_level, n_points, noise)
    elif dgp_name == 'smooth_signal':
        return generate_smooth_signal(X, noise_level, n_points, noise)
    elif dgp_name == 'breiman84':
        return generate_breiman84(X, noise_level, noise)
    elif dgp_name == 'friedman1':
        return generate_friedman1(X, noise_level, noise)

    elif dgp_name == 'additive_smooth':
        return additive_smooth(X, noise_level, noise)
    elif dgp_name == 'additive_step':
        return additive_step(X, noise_level, noise)
    elif dgp_name == 'additive_linear':
        return additive_linear(X, noise_level, noise)
    elif dgp_name == 'additive_hills':
        return additive_hills(X, noise_level, noise)







def func_hills(x, split=0, vals=(1, 1, 10), rev=False):
    ans = np.full(len(x), np.nan)  # Initialize with NaNs
    if not rev:
        ans[x < split] = vals[0] + np.sin(vals[1] * x[x < split])
        eps = (vals[1] / vals[2]) * np.cos(vals[1] * split) / np.cos(vals[2] * split)
        delta = vals[0] + np.sin(vals[1] * split) - eps * np.sin(vals[2] * split)
        ans[x >= split] = delta + eps * np.sin(vals[2] * x[x >= split])
    else:
        ans[x > split] = vals[0] + np.sin(vals[1] * x[x > split])
        eps = (vals[1] / vals[2]) * np.cos(vals[1] * split) / np.cos(vals[2] * split)
        delta = vals[0] + np.sin(vals[1] * split) - eps * np.sin(vals[2] * split)
        ans[x <= split] = delta + eps * np.sin(vals[2] * x[x <= split])
    return ans

def additive_hills(X, noise_level, noise):
    f1 = func_hills(X[:,0], 0, (1, 1, 12))
    f2 = func_hills(X[:,1], 1, (1, 2, 8))
    f3 = func_hills(X[:,2], -1, (0, 3, 15), rev=True)
    f4 = func_hills(X[:,3], 1, (0, 2.5, 10), rev=True)

    y = f1 + f2 + f3 + f4 + noise
    return y, noise


def func_step(X, knots, vals):
    """Apply piecewise constant values based on knots."""
    # Start with the last value for all x (assuming x > last knot)
    y = np.full_like(X, vals[-1], dtype=float)

    # Assign values for intervals defined by knots
    for i in range(len(knots)):
        if i == 0:
            y[X <= knots[i]] = vals[i]
        else:
            y[(X > knots[i - 1]) & (X <= knots[i])] = vals[i]

    # For values beyond the last knot, the value is already set as vals[-1]
    return y


def f1(X):
    knots = [-2.3, -1.8, -0.5, 1.1]
    vals = [-3, -2.5, -1, 1, 1.8]
    return func_step(X, knots, vals)


def f2(X):
    knots = [-2, -1, 1, 2]
    vals = [3, 1.4, 0, -1.7, -1.8]
    return func_step(X, knots, vals)


def f3(X):
    knots = [-1.5, 0.5]
    vals = [-3.3, 2.5, -1]
    return func_step(X, knots, vals)


def f4(X):
    knots = [-1.7, -0.4, 1.5, 1.9]
    vals = [-2.8, 0.3, -1.4, 0.4, 1.8]
    return func_step(X, knots, vals)

def additive_step(X, noise_level, noise):

    # Apply functions
    y1 = f1(X[:,0])
    y2 = f2(X[:,1])
    y3 = f3(X[:,2])
    y4 = f4(X[:,3])

    y = y1 + y2 + y3 + y4 + noise
    return y, noise



def additive_smooth(X, noise_level, noise):

    # Linear, quadratic, sine and exponential
    y = -2 * np.sin(2 * X[:,0]) + (0.8 * X[:,1]**2 - 2.5) + (X[:,2] - 1/2) + (np.exp(-0.65 * X[:,3]) - 2.5) + noise

    return y, noise




def linear_interp(x, knots, values):
    return np.interp(x, knots, values)

# Define the functions with updated linear interpolation
def f1_lin(x):
    knots = [-2.5, -2.3, 1, 2.5]  # Extended to ensure range covers the plot
    values = [0.5, -2.5, 1.8, 2.3]
    return linear_interp(x, knots, values)

def f2_lin(x):
    knots = [-2.5, -2, -1, 1, 2, 2.5]  # Extended to ensure range covers the plot
    values = [-0.5, 2.5, 1, -0.5, -2.2, -2.3]
    return linear_interp(x, knots, values)

def f3_lin(x):
    knots = [-2.5, -1.5, 0.5, 2.5]  # Extended to ensure range covers the plot
    values = [0, -3, 2.5, -1]  # Adjusted to have the same number of values as knots
    return linear_interp(x, knots, values)

def f4_lin(x):
    knots = [-2.5, -1.8, -0.5, 1.5, 1.8, 2.5]  # Extended to ensure range covers the plot
    values = [-1, -3.8, -1, -2.3, -0.5, 0.8]
    return linear_interp(x, knots, values)


def additive_linear(X, noise_level, noise):

    # Apply functions
    y1 = f1_lin(X[:,0])
    y2 = f2_lin(X[:,1])
    y3 = f3_lin(X[:,2])
    y4 = f4_lin(X[:,3])

    y = y1 + y2 + y3 + y4 + noise

    return y, noise







# This is Example of Breiman, 1984, 8.6., page 238
def generate_breiman84(X, noise_level, noise):
    # Initialize Y
    m = np.zeros(noise.shape[0])

    # Compute Y based on the condition
    for i in range(noise.shape[0]):
        if X[i, 0] == 1:
            m[i] = 3 + 3*X[i, 1] + 2*X[i, 2] + X[i, 3]
        else:  # X[i, 0] == -1
            m[i] = -3 + 3*X[i, 4] + 2*X[i, 5] + X[i, 6]

    y = m + noise
    return y, noise

# Friedman #1
def generate_friedman1(X, noise_level, noise):

    y = 10*np.sin(np.pi * X[:, 0]*X[:, 1]) + 20*(X[:, 2] - 0.5)**2 + 10*X[:, 3] + 5*X[:, 4] + noise

    return y, noise


def generate_sine_cosine(X, noise_level, n_points, noise):

    x1 = X[:, 0] / n_points
    x2 = X[:, 1] / n_points
    # For X uniform:
    #x1 = X[:, 0]
    #x2 = X[:, 1]
    y = np.sin(3 * np.pi * x1) + np.cos(5 * np.pi * x2) + noise
    return y, noise

def generate_rectangular(X, noise_level, n_points, noise):

    # X equidistant, rectangular in middle:
    #y_temp = ((1 * n_points / 3 <= X[:, 0]) * (X[:, 0] <= 2 * n_points / 3) * (1 * n_points / 3 <= X[:, 1]) * (X[:, 1] <= 2 * n_points / 3))

    # For X uniform:
    y_temp = ((1 / 3 <= X[:, 0]) * (X[:, 0] <= 2 * 1 / 3) * (1 / 3 <= X[:, 1]) * (X[:, 1] <= 2 * 1 / 3))

    y = y_temp.astype(int) + noise
    return y, noise

def generate_circular(X, noise_level, n_points, noise):

    #y_temp = np.sqrt((X[:, 0] - n_points / 2) ** 2 + (X[:, 1] - n_points / 2) ** 2) <= n_points / 4
    # For X uniform:
    y_temp = np.sqrt((X[:, 0] - 1 / 2) ** 2 + (X[:, 1] - 1 / 2) ** 2) <= 1 / 4


    y = y_temp.astype(int) + noise
    return y, noise

def generate_smooth_signal(X, noise_level, n_points, noise):

    x1 = X[:, 0] / n_points
    x2 = X[:, 1] / n_points
    #x1 = X[:, 0] / 1
    #x2 = X[:, 1] / 1

    y = 20 * np.exp(-5 * ((x1 - 1 / 2) ** 2 + (x2 - 1 / 2) ** 2 - 0.9 * (x1 - 1 / 2) * (x2 - 1 / 2))) + noise
    return y, noise
