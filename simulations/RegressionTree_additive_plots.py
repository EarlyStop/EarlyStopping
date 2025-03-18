import numpy as np
import matplotlib.pyplot as plt
import os
import RegressionTree_additive_generation as data_gen
import EarlyStopping as es

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


# Generate uniformly distributed data in the interval [-2.5, 2.5]
x = np.linspace(-2.5, 2.5, 400)

# Create a 2D array where each column is the same x values
# This is needed because the functions in data_gen expect a 2D array
X = np.column_stack([x, x, x, x])

# Plot 1: Piecewise constant functions
y1 = data_gen.f1(x)
y2 = data_gen.f2(x)
y3 = data_gen.f3(x)
y4 = data_gen.f4(x)



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

# Plot 2: Piecewise linear functions
y1_lin = data_gen.f1_lin(x)
y2_lin = data_gen.f2_lin(x)
y3_lin = data_gen.f3_lin(x)
y4_lin = data_gen.f4_lin(x)

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

# Plot 3: Hills functions
# For hills functions, we need to apply each function to the corresponding column
f1 = data_gen.func_hills(x, 0, (1, 1, 12))
f2 = data_gen.func_hills(x, 1, (1, 2, 8))
f3 = data_gen.func_hills(x, -1, (0, 3, 15), rev=True)
f4 = data_gen.func_hills(x, 1, (0, 2.5, 10), rev=True)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

ax.plot(x, f1, color="blue", linewidth=1.5, label='Function 1')
ax.plot(x, f2, color="purple", linewidth=1.5, label='Function 2')
ax.plot(x, f3, color="#CCCC00", linewidth=1.5, label='Function 3')
ax.plot(x, f4, color="black", linewidth=1.5, label='Function 4')

ax.grid(True)

# Save figure
fig_dir = "."
fig.savefig(os.path.join(fig_dir, 'hills_functions_esfiep.png'), bbox_inches="tight", dpi=300)
plt.show()

# Plot 4: Smooth functions
# Extract the individual components from additive_smooth
x_single = np.zeros((400, 4))
x_single[:, 0] = x  # Set first column to x values

# Function 1: -2 * np.sin(2 * X[:,0])
f1_smooth = -2 * np.sin(2 * x)

# Set second column to x values for function 2
x_single[:, 1] = x
# Function 2: (0.8 * X[:,1]**2 - 2.5)
f2_smooth = 0.8 * x**2 - 2.5

# Set third column to x values for function 3
x_single[:, 2] = x
# Function 3: (X[:,2] - 1/2)
f3_smooth = x - 1/2

# Set fourth column to x values for function 4
x_single[:, 3] = x
# Function 4: (np.exp(-0.65 * X[:,3]) - 2.5)
f4_smooth = np.exp(-0.65 * x) - 2.5

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

ax.plot(x, f1_smooth, color="blue", linewidth=1.5, label='Function 1')
ax.plot(x, f2_smooth, color="purple", linewidth=1.5, label='Function 2')
ax.plot(x, f3_smooth, color="#CCCC00", linewidth=1.5, label='Function 3')
ax.plot(x, f4_smooth, color="black", linewidth=1.5, label='Function 4')

ax.grid(True)

# Save figure
fig_dir = "."
fig.savefig(os.path.join(fig_dir, 'smooth_functions_esfiep.png'), bbox_inches="tight", dpi=300)
plt.show()