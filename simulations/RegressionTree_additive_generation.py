import numpy as np

def generate_data_from_X(X, noise_level, dgp_name, add_noise=True):

    n = X.shape[0]
    if add_noise:
        noise = np.random.normal(0, noise_level, n)
    else:
        noise = np.zeros(n)

    if dgp_name == 'additive_smooth':
        return additive_smooth(X, noise)
    elif dgp_name == 'additive_step':
        return additive_step(X, noise)
    elif dgp_name == 'additive_linear':
        return additive_linear(X, noise)
    elif dgp_name == 'additive_hills':
        return additive_hills(X, noise)


def func_hills(x, split=0, vals=(1, 1, 10), rev=False):
    ans = np.full(len(x), np.nan)
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

def additive_hills(X, noise):
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

def additive_step(X, noise):

    # Apply functions
    y1 = f1(X[:,0])
    y2 = f2(X[:,1])
    y3 = f3(X[:,2])
    y4 = f4(X[:,3])

    y = y1 + y2 + y3 + y4 + noise
    return y, noise


def additive_smooth(X, noise):

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


def additive_linear(X, noise):

    # Apply functions
    y1 = f1_lin(X[:,0])
    y2 = f2_lin(X[:,1])
    y3 = f3_lin(X[:,2])
    y4 = f4_lin(X[:,3])

    y = y1 + y2 + y3 + y4 + noise

    return y, noise


