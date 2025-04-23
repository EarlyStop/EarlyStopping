################################################################################
#             Comparison of Landweber and tSVD Signal Estimation                  #
################################################################################

# Imports
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import EarlyStopping as es
from scipy.fft import fft


# Generate signals and design matrix
# ------------------------------------------------------------------------------
# Create the data for supersmooth, smooth, and rough signals
design, response_noiseless_supersmooth, true_signal_supersmooth = es.SimulationData.diagonal_data(sample_size=10000, type='supersmooth')
design, response_noiseless_smooth, true_signal_smooth = es.SimulationData.diagonal_data(sample_size=10000, type='smooth')
design, response_noiseless_rough, true_signal_rough = es.SimulationData.diagonal_data(sample_size=10000, type='rough')

# We'll focus on the smooth signal for our demonstration
signal_type = 'smooth'  # Options: 'supersmooth', 'smooth', 'rough'

# Select the appropriate signal based on the chosen type
if signal_type == 'supersmooth':
    true_signal = true_signal_supersmooth
    response_noiseless = response_noiseless_supersmooth
elif signal_type == 'smooth':
    true_signal = true_signal_smooth
    response_noiseless = response_noiseless_smooth
else:  # rough
    true_signal = true_signal_rough
    response_noiseless = response_noiseless_rough

# Set noise level and create noisy response
noise_level = 0.005
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, noise_level, design.shape[0])
response = response_noiseless + noise

# Landweber Estimation
# ------------------------------------------------------------------------------
# Initialize Landweber estimator
landweber = es.Landweber(
    design=design,
    response=response,
    learning_rate=1,
    true_signal=true_signal,
    true_noise_level=noise_level
)

# Run Landweber iterations
max_iterations_landweber = 1000
landweber.iterate(max_iterations_landweber)

# Get the stopping index using discrepancy principle
critical_value = noise_level**2 * design.shape[0] 
landweber_stopping_index = landweber.get_discrepancy_stop(critical_value, max_iterations_landweber)
print(f"Landweber stopping index: {landweber_stopping_index}")

# Get the Landweber estimate at the stopping index
landweber_estimate = landweber.get_estimate(landweber_stopping_index)

# Truncated SVD Estimation
# ------------------------------------------------------------------------------
# Initialize Truncated SVD estimator
tsvd = es.TruncatedSVD(
    design=design,
    response=response,
    true_signal=true_signal,
    true_noise_level=noise_level,
    diagonal=True  # Since we're using diagonal design matrix
)

# Run tSVD iterations
max_iterations_tsvd = 1000
tsvd.iterate(max_iterations_tsvd)

# Get the stopping index using discrepancy principle
tsvd_stopping_index = tsvd.get_discrepancy_stop(critical_value, max_iterations_tsvd)
print(f"tSVD stopping index: {tsvd_stopping_index}")

# Get the tSVD estimate at the stopping index
tsvd_estimate = tsvd.get_estimate(tsvd_stopping_index)

# Conjugate Gradient Estimation
# ------------------------------------------------------------------------------
# Initialize Conjugate Gradient estimator
conjugate_gradient = es.ConjugateGradients(
    design=design,
    response=response,
    true_signal=true_signal,
    true_noise_level=noise_level
)

# Run Conjugate Gradient iterations
max_iterations_cg = 1000
conjugate_gradient.iterate(max_iterations_cg)

# Get the stopping index using discrepancy principle
cg_stopping_index = conjugate_gradient.get_discrepancy_stop(critical_value, max_iterations_cg)
print(f"Conjugate Gradient stopping index: {cg_stopping_index}")

# Get the Conjugate Gradient estimate at the stopping index
cg_estimate = conjugate_gradient.get_estimate(cg_stopping_index)

# Compute risks and find the index with the smallest risk
# ------------------------------------------------------------------------------
landweber_risk = landweber.strong_risk
landweber_min_risk_index = np.argmin(landweber_risk)
landweber_min_risk_estimate = landweber.get_estimate(landweber_min_risk_index)

cg_risk = conjugate_gradient.strong_empirical_risk
cg_min_risk_index = np.argmin(cg_risk)
cg_min_risk_estimate = conjugate_gradient.get_estimate(cg_min_risk_index)

tsvd_risk = tsvd.strong_risk
tsvd_min_risk_index = np.argmin(tsvd_risk)
tsvd_min_risk_estimate = tsvd.get_estimate(tsvd_min_risk_index)

# Visualization of Stopping Index Estimates
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot only the first 1000 components for better visibility
plot_range = 10000
x_indices = np.arange(1, plot_range + 1)

# Plot Landweber estimate
plt.plot(x_indices, fft(landweber_estimate[:plot_range]), color="red", label='Landweber')
# Plot tSVD estimate
plt.plot(x_indices, fft(tsvd_estimate[:plot_range]), color="purple", label='tSVD')
# Plot Conjugate Gradient estimate
plt.plot(x_indices, fft(cg_estimate[:plot_range]), color="blue", label='Conjugate Gradient')
# Plot true signal
plt.plot(x_indices, fft(true_signal[:plot_range]), color="black", label='True Signal')
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.yscale("log")

# Add labels and title
plt.xlabel('', fontsize=22)
plt.ylabel('', fontsize=22)
plt.ylim([40, 100])
plt.xlim([250, 1000])
plt.grid(True)

# Save the figure
plt.tight_layout()
plt.savefig(f'signal_estimation_comparison_fft_{signal_type}.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization of Minimum Risk Estimates
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot only the first 1000 components for better visibility
plot_range = 10000
x_indices = np.arange(1, plot_range + 1)

# Plot Landweber minimum risk estimate
plt.plot(x_indices, landweber_estimate[:plot_range], color="red", label='Landweber')
# Plot tSVD estimate
plt.plot(x_indices, tsvd_estimate[:plot_range], color="purple", label='tSVD')
# Plot Conjugate Gradient estimate
plt.plot(x_indices, cg_estimate[:plot_range], color="blue", label='Conjugate Gradient')
# Plot true signal
plt.plot(x_indices, true_signal[:plot_range], color="black", label='True Signal')
plt.tick_params(axis='both', which='major', labelsize=14)

# Add labels and title
plt.xlabel('', fontsize=22)
plt.ylabel('', fontsize=22)
plt.ylim([0, 1.6])
plt.xlim([0, 1000])
plt.grid(True)

# Save the figure
plt.tight_layout()
plt.savefig(f'signal_estimation_comparison_{signal_type}.png', dpi=300, bbox_inches='tight')
plt.show()
