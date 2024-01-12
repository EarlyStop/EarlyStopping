import matplotlib.pyplot as plt
import numpy as np
import EarlyStopping as es
import random, time
random.seed(42)

# Create diagonal design matrices
D = 1000
indices = np.arange(D)+1
design_matrix = np.diag(1/(np.sqrt(indices)))

# Create signals from Stankewitz (2020)
signal_supersmooth = 5*np.exp(-0.1*indices)
signal_smooth = 5000*np.abs(np.sin(0.01*indices))*indices**(-1.6)
signal_rough = 250*np.abs(np.sin(0.002*indices))*indices**(-0.8)

plt.figure(figsize=(10, 6))
plt.plot(indices, signal_supersmooth, label="supersmooth signal")
plt.plot(indices, signal_smooth, label="smooth signal")
plt.plot(indices, signal_rough, label="rough signal")
plt.ylabel("Signal")
plt.xlabel("Index")
plt.xlim([0,1000])
plt.ylim([0,1.6])
plt.legend()
plt.show()


NOISE_LEVEL = 0.01
noise = np.random.normal(0, NOISE_LEVEL, D)

observation_supersmooth = noise + np.matmul(design_matrix, signal_supersmooth)
observation_smooth = noise + np.matmul(design_matrix, signal_smooth)
observation_rough = noise + np.matmul(design_matrix, signal_rough)


models_supersmooth = es.Landweber(design_matrix, observation_supersmooth, true_noise_level=NOISE_LEVEL, true_signal=signal_supersmooth)
models_smooth = es.Landweber(design_matrix, observation_smooth, true_noise_level=NOISE_LEVEL, true_signal=signal_smooth)
models_rough = es.Landweber(design_matrix, observation_rough, true_noise_level=NOISE_LEVEL, true_signal=signal_rough)

iter = 1500
models_supersmooth.landweber_gather_all(iter)
models_smooth.landweber_gather_all(iter)
models_rough.landweber_gather_all(iter)

# Stopping index
supersmooth_m = models_supersmooth.early_stopping_index
smooth_m = models_smooth.early_stopping_index
rough_m = models_rough.early_stopping_index

# Weak balanced oracle
supersmooth_weak_oracle = models_supersmooth.weak_balanced_oracle
smooth_weak_oracle = models_smooth.weak_balanced_oracle
rough_weak_oracle = models_rough.weak_balanced_oracle

# Strong balanced oracle
supersmooth_strong_oracle = models_supersmooth.strong_balanced_oracle
smooth_strong_oracle = models_smooth.strong_balanced_oracle
rough_strong_oracle = models_rough.strong_balanced_oracle

fig, axs = plt.subplots(3, 1, figsize=(14, 8))

axs[0].plot(range(0, iter+1), models_supersmooth.residuals)
axs[0].axvline(x=supersmooth_m, color='red', linestyle='--')
axs[0].set_xlim([0, 400])
axs[0].set_ylim([0, 100])
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Residuals')


axs[1].plot(range(0, iter+1), models_supersmooth.strong_error, color='orange', label='Error')
axs[1].plot(range(0, iter+1), models_supersmooth.strong_bias2, label='$Bias^2$', color='grey')
axs[1].plot(range(0, iter+1), models_supersmooth.strong_variance, label='Variance', color='blue')
axs[1].axvline(x=supersmooth_m, color='red', linestyle='--')
axs[1].axvline(x=supersmooth_strong_oracle, color='blue', linestyle=':')
axs[1].set_xlim([0, 400])
axs[1].set_ylim([0, 100])
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Strong Quantities')

axs[2].plot(range(0, iter+1), models_supersmooth.weak_error, color='orange', label='Error')
axs[2].plot(range(0, iter+1), models_supersmooth.weak_bias2, label='$Bias^2$', color='grey')
axs[2].plot(range(0, iter+1), models_supersmooth.weak_variance, label='Variance', color='blue')
axs[2].axvline(x=supersmooth_m, color='red', linestyle='--')
axs[2].axvline(x=supersmooth_weak_oracle, color='blue', linestyle=':')
axs[2].set_xlim([0, 400])
axs[2].set_ylim([0, 0.02])
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Weak Quantities')
axs[2].legend()

plt.tight_layout()

plt.show()


fig, axs = plt.subplots(3, 1, figsize=(14, 8))

axs[0].plot(range(0, iter+1), models_smooth.residuals)
axs[0].axvline(x=smooth_m, color='red', linestyle='--')
axs[0].set_xlim([0, 500])
axs[0].set_ylim([0, 100])
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Residuals')

axs[1].plot(range(0, iter+1), models_smooth.strong_error, color='orange', label='Error')
axs[1].plot(range(0, iter+1), models_smooth.strong_bias2, label='$Bias^2$', color='grey')
axs[1].plot(range(0, iter+1), models_smooth.strong_variance, label='Variance', color='blue')
axs[1].axvline(x=smooth_m, color='red', linestyle='--')
axs[1].axvline(x=smooth_strong_oracle, color='blue', linestyle=':')
axs[1].set_xlim([0, 500])
axs[1].set_ylim([0, 100])
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Strong Quantities')

axs[2].plot(range(0, iter+1), models_smooth.weak_error, color='orange', label='Error')
axs[2].plot(range(0, iter+1), models_smooth.weak_bias2, label='$Bias^2$', color='grey')
axs[2].plot(range(0, iter+1), models_smooth.weak_variance, label='Variance', color='blue')
axs[2].axvline(x=smooth_m, color='red', linestyle='--')
axs[2].axvline(x=smooth_weak_oracle, color='blue', linestyle=':')
axs[2].set_xlim([0, 500])
axs[2].set_ylim([0, 0.5])
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Weak Quantities')
axs[2].legend()

plt.tight_layout()

fig.suptitle('Smooth Signal', fontsize=16)

plt.show()


fig, axs = plt.subplots(3, 1, figsize=(14, 8))

axs[0].plot(range(0, iter+1), models_rough.residuals)
axs[0].axvline(x=rough_m, color='red', linestyle='--')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Residuals')

axs[1].plot(range(0, iter+1), models_rough.strong_error, color='orange', label='Error')
axs[1].plot(range(0, iter+1), models_rough.strong_bias2, label='$Bias^2$', color='grey')
axs[1].plot(range(0, iter+1), models_rough.strong_variance, label='Variance', color='blue')
axs[1].axvline(x=rough_m, color='red', linestyle='--')
axs[1].axvline(x=rough_strong_oracle, color='blue', linestyle=':')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Strong Quantities')

axs[2].plot(range(0, iter+1), models_rough.weak_error, color='orange', label='Error')
axs[2].plot(range(0, iter+1), models_rough.weak_bias2, label='$Bias^2$', color='grey')
axs[2].plot(range(0, iter+1), models_rough.weak_variance, label='Variance', color='blue')
axs[2].axvline(x=rough_m, color='red', linestyle='--')
axs[2].axvline(x=rough_weak_oracle, color='blue', linestyle=':')
axs[2].set_xlim([0, 1200+1])
axs[2].set_ylim([0, 1])
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Weak Quantities')
axs[2].legend()

plt.tight_layout()

fig.suptitle('Rough Signal', fontsize=16)

plt.show()