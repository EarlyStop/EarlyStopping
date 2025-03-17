import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure consistent style
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)


def func_step(x, knots, vals):
    """Apply piecewise constant values based on knots."""
    # Start with the last value for all x (assuming x > last knot)
    y = np.full_like(x, vals[-1], dtype=float)

    # Assign values for intervals defined by knots
    for i in range(len(knots)):
        if i == 0:
            y[x <= knots[i]] = vals[i]
        else:
            y[(x > knots[i - 1]) & (x <= knots[i])] = vals[i]

    # For values beyond the last knot, the value is already set as vals[-1]
    return y


def f1(x):
    knots = [-2.3, -1.8, -0.5, 1.1]
    vals = [-3, -2.5, -1, 1, 1.8]
    return func_step(x, knots, vals)


def f2(x):
    knots = [-2, -1, 1, 2]
    vals = [3, 1.4, 0, -1.7, -1.8]
    return func_step(x, knots, vals)


def f3(x):
    knots = [-1.5, 0.5]
    vals = [-3.3, 2.5, -1]
    return func_step(x, knots, vals)


def f4(x):
    knots = [-1.7, -0.4, 1.5, 1.9]
    vals = [-2.8, 0.3, -1.4, 0.4, 1.8]
    return func_step(x, knots, vals)


# Generate uniformly distributed data
x = np.linspace(-2.5, 2.5, 400)

# Apply functions
y1 = f1(x)
y2 = f2(x)
y3 = f3(x)
y4 = f4(x)


# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

ax.plot(x, y1, color="blue", linewidth=1.5, label='Function 1')
ax.plot(x, y2, color="purple", linewidth=1.5, label='Function 2')
ax.plot(x, y3, color="#CCCC00", linewidth=1.5, label='Function 3')
ax.plot(x, y4, color="black", linewidth=1.5, label='Function 4')


ax.grid(True)

# Save figure
fig_dir = "."
fig.savefig(os.path.join(fig_dir, 'piecewise_constant_functions_esfiep.png'), bbox_inches="tight", dpi=300)
plt.show()


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

# Generate uniformly distributed data
x = np.linspace(-2.5, 2.5, 400)

# Apply functions
y1_lin = f1_lin(x)
y2_lin = f2_lin(x)
y3_lin = f3_lin(x)
y4_lin = f4_lin(x)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

ax.plot(x, y1_lin, color="blue", linewidth=1.5, label='Function 1')
ax.plot(x, y2_lin, color="purple", linewidth=1.5, label='Function 2')
ax.plot(x, y3_lin, color="#CCCC00", linewidth=1.5, label='Function 3')
ax.plot(x, y4_lin, color="black", linewidth=1.5, label='Function 4')

ax.grid(True)

# Save figure
fig_dir = "."
fig.savefig(os.path.join(fig_dir, 'piecewise_linear_functions_esfiep.png'), bbox_inches="tight", dpi=300)
plt.show()


def func_hills(x, split=0, vals=(1, 1, 10), rev=False):
    ans = np.full(len(x), np.nan)  # Initialize with NaNs
    if not rev:
        ans[x < split] = vals[0] + np.sin(vals[1] * x[x < split])
        eps = (vals[1] / vals[2]) * np.cos(vals[1] * split) / np.cos(vals[2] * split)  # Corrected indices
        delta = vals[0] + np.sin(vals[1] * split) - eps * np.sin(vals[2] * split)  # Corrected indices
        ans[x >= split] = delta + eps * np.sin(vals[2] * x[x >= split])  # Corrected indices
    else:
        ans[x > split] = vals[0] + np.sin(vals[1] * x[x > split])
        eps = (vals[1] / vals[2]) * np.cos(vals[1] * split) / np.cos(vals[2] * split)  # Corrected indices
        delta = vals[0] + np.sin(vals[1] * split) - eps * np.sin(vals[2] * split)  # Corrected indices
        ans[x <= split] = delta + eps * np.sin(vals[2] * x[x <= split])  # Corrected indices
    return ans

def scen5(x):
    x1 = func_hills(x, 0, (1, 1, 12))
    x2 = func_hills(x, 1, (1, 2, 8))
    x3 = func_hills(x, -1, (0, 3, 15), rev=True)
    x4 = func_hills(x, 1, (0, 2.5, 10), rev=True)
    return np.column_stack((x1, x2, x3, x4))

# Generate uniformly distributed data in the interval [-2.5, 2.5]
x = np.linspace(-2.5, 2.5, 400)
results = scen5(x)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

ax.plot(x, results[:, 0], color="blue", linewidth=1.5, label='Function 1')
ax.plot(x, results[:, 1], color="purple", linewidth=1.5, label='Function 2')
ax.plot(x, results[:, 2], color="#CCCC00", linewidth=1.5, label='Function 3')
ax.plot(x, results[:, 3], color="black", linewidth=1.5, label='Function 4')

ax.grid(True)

# Save figure
fig_dir = "."
fig.savefig(os.path.join(fig_dir, 'hills_functions_esfiep.png'), bbox_inches="tight", dpi=300)
plt.show()

def smooth(X):
    x1 = -2 * np.sin(2 * X)
    x2 = (0.8 * X ** 2 - 2.5)
    x3 = (X - 1 / 2)
    x4 = (np.exp(-0.65 * X) - 2.5)
    return np.column_stack((x1, x2, x3, x4))


x = np.linspace(-2.5, 2.5, 400)
results_smooth = smooth(x)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

ax.plot(x, results_smooth[:, 0], color="blue", linewidth=1.5, label='Function 1')
ax.plot(x, results_smooth[:, 1], color="purple", linewidth=1.5, label='Function 2')
ax.plot(x, results_smooth[:, 2], color="#CCCC00", linewidth=1.5, label='Function 3')
ax.plot(x, results_smooth[:, 3], color="black", linewidth=1.5, label='Function 4')

ax.grid(True)

# Save figure
fig_dir = "."
fig.savefig(os.path.join(fig_dir, 'smooth_functions_esfiep.png'), bbox_inches="tight", dpi=300)
plt.show()