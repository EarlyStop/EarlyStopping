for name in list(globals()):
    if not name.startswith("_"):
        del globals()[name]


import numpy as np
import importlib
import EarlyStopping as es
from scipy.sparse import dia_matrix

importlib.reload(es)

# sample_size = 1000
# indices = np.arange(sample_size) + 1

# parameters_supersmooth = es.SimulationParameters(
#     design=dia_matrix(np.diag(1 / np.sqrt(indices))),
#     true_signal=5 * np.exp(-0.1 * indices),
#     true_noise_level=0.01,
#     max_iterations=1000,
# )
#
# sample_size_gravity = 100  # 2**9
# a = 0
# b = 1
# d = 0.25  # Parameter controlling the ill-posedness: the larger, the more ill-posed, default in regtools: d = 0.25
#
# t = (np.arange(1, sample_size_gravity + 1) - 0.5) / sample_size_gravity
# s = ((np.arange(1, sample_size_gravity + 1) - 0.5) * (b - a)) / sample_size_gravity
# T, S = np.meshgrid(t, s)
#
#
# parameters_gravity = es.SimulationParameters(
#     design=(1 / sample_size_gravity)
#     * d
#     * (d**2 * np.ones((sample_size_gravity, sample_size_gravity)) + (S - T) ** 2) ** (-(3 / 2)),
#     true_signal=np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t),
#     true_noise_level=0.01,
#     max_iterations=1000,
# )

import numpy as np
from scipy.linalg import toeplitz
#
#
# def phillips(n):
#     # Check if n is a multiple of 4
#     if n % 4 != 0:
#         raise ValueError('The order n must be a multiple of 4')
#
#     # Compute the matrix A using the toeplitz function
#     h = 12 / n
#     n4 = n // 4
#     c = np.cos(np.arange(-1, n4+1) * 4 * np.pi / n)
#     r1 = np.zeros(n)
#     # Debug: Print array shapes
#     print('c[1:n4+1].shape:', c[1:n4 + 1].shape, c[1:n4 + 1])
#     print('c[:n4].shape:', c[:n4].shape, c[:n4])
#     print('c[2:n4+2].shape:', c[2:n4 + 2].shape, c[2:n4 + 2])
#     print(n4)
#     print(c)
#
#     r1[:n4] = h + 9 / (h * np.pi ** 2) * (2 * c[1:n4 + 1] - c[:n4] - c[2:n4 + 2])
#     r1[n4] = h / 2 + 9 / (h * np.pi ** 2) * (np.cos(4 * np.pi / n) - 1)
#     A = toeplitz(r1)
#
#     # Compute the right-hand side b
#     b = np.zeros(n)
#     c = np.pi / 3
#     for i in range(n // 2, n):
#         t1 = -6 + (i + 1) * h
#         t2 = t1 - h
#         b[i] = (t1 * (6 - abs(t1) / 2) +
#                 ((3 - abs(t1) / 2) * np.sin(c * t1) - 2 / c * (np.cos(c * t1) - 1)) / c -
#                 t2 * (6 - abs(t2) / 2) -
#                 ((3 - abs(t2) / 2) * np.sin(c * t2) - 2 / c * (np.cos(c * t2) - 1)) / c)
#         b[n - i - 1] = b[i]
#     b /= np.sqrt(h)
#
#     # Compute the solution x
#     x = np.zeros(n)
#     s = np.arange(0, h * 5 + 10 * np.finfo(float).eps, h)
#     x[2 * n4:3 * n4] = (h + np.diff(np.sin(s * c)) / c) / np.sqrt(h)
#     x[n4:2 * n4] = x[3 * n4 - 1:2 * n4 - 1:-1]
#
#     return A, b, x
#
#
# # Example usage:
# n = 20  # n needs to be a multiple of 4
# design, response, signal = phillips(n)

#
# def deriv2(n, example=1):
#     # Initialize variables and compute coefficients
#     h = 1 / n
#     sqh = np.sqrt(h)
#     h32 = h * sqh
#     h2 = h ** 2
#     sqhi = 1 / sqh
#     t = 2 / 3
#     A = np.zeros((n, n))
#
#     # Compute the matrix A
#     for i in range(1, n+1):
#         A[i-1, i-1] = h2 * ((i**2 - i + 0.25) * h - (i - t))
#         for j in range(1, i):
#             A[i-1, j-1] = h2 * (j - 0.5) * ((i - 0.5) * h - 1)
#     A = A + np.tril(A, -1).T
#
#     # Compute the right-hand side vector b
#     b = np.zeros(n)
#     if example == 1:
#         for i in range(1, n+1):
#             b[i-1] = h32 * (i - 0.5) * ((i**2 + (i - 1)**2) * h2 / 2 - 1) / 6
#     elif example == 2:
#         ee = 1 - np.exp(1)
#         for i in range(1, n+1):
#             b[i-1] = sqhi * (np.exp(i * h) - np.exp((i-1) * h) + ee * (i - 0.5) * h2 - h)
#     elif example == 3:
#         if n % 2 != 0:
#             raise ValueError('Order n must be even')
#         else:
#             for i in range(1, n // 2 + 1):
#                 s1 = i * h
#                 s12 = s1 ** 2
#                 s2 = (i-1) * h
#                 s22 = s2 ** 2
#                 b[i-1] = sqhi * (s12 + s22 - 1.5) * (s12 - s22) / 24
#             for i in range(n // 2+1, n+1):
#                 s1 = i * h
#                 s12 = s1 ** 2
#                 s2 = (i-1) * h
#                 s22 = s2 ** 2
#                 b[i-1] = sqhi * (-(s12 + s22) * (s12 - s22) + 4 * (s1 ** 3 - s2 ** 3) -
#                                4.5 * (s12 - s22) + h) / 24
#     else:
#         raise ValueError('Illegal value of example')
#
#     # Compute the solution vector x
#     x = np.zeros(n)
#     if example == 1:
#         for i in range(1, n+1):
#             x[i-1] = h32 * (i - 0.5)
#     elif example == 2:
#         for i in range(1, n+1):
#             x[i-1] = sqhi * (np.exp(i * h) - np.exp((i-1) * h))
#     elif example == 3:
#         for i in range(1, n // 2 + 1):
#             x[i-1] = sqhi * ((i  * h) ** 2 - ((i-1) * h) ** 2) / 2
#         for i in range(n // 2 + 1, n+1):
#             x[i-1] = sqhi * (h - ((i * h) ** 2 - ((i-1) * h) ** 2) / 2)
#
#     return A, b, x
#
# # Example usage
# n = 20
# design, response, signal = deriv2(n, example=3)
#


import numpy as np
from scipy.linalg import toeplitz

def heat(n, kappa=1):
    """
    Test problem: inverse heat equation.

    Parameters:
    n (int): Number of discretization points.
    kappa (float): Controls the ill-conditioning of the matrix.

    Returns:
    A (numpy.ndarray): The matrix representing the integral operator.
    b (numpy.ndarray): The right-hand side vector.
    x (numpy.ndarray): The exact solution vector.
    """
    # Initialization
    h = 1 / n
    t = np.linspace(h / 2, 1, n)  # midpoints
    c = h / (2 * kappa * np.sqrt(np.pi))
    d = 1 / (4 * kappa ** 2)

    # Compute the matrix A
    k = c * t ** (-1.5) * np.exp(-d / t)
    r = np.zeros(len(t))
    r[0] = k[0]
    A = toeplitz(k, r)

    # Compute the vectors x and b
    x = np.zeros(n)
    for i in range(1, n // 2 + 1):
        ti = i * 20 / n
        if ti < 2:
            x[i-1] = 0.75 * (ti ** 2) / 4
        elif ti < 3:
            x[i-1] = 0.75 + (ti - 2) * (3 - ti)
        else:
            x[i-1] = 0.75 * np.exp(-(ti - 3) * 2)

    x[n // 2 + 1:] = 0
    b = np.dot(A, x)

    return A, b, x

# Example usage
n = 100  # Number of discretization points
kappa = 1  # Control parameter for ill-conditioning
design, response, signal = heat(n, kappa)








simulation = es.SimulationWrapper(**parameters_gravity.__dict__)

results = simulation.run_simulation()
