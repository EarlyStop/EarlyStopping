import EarlyStopping as es
import numpy as np

para_size = 30
indices = np.arange(para_size)+1
design_matrix = np.diag(1/(np.sqrt(indices)))

signal = 5*np.exp(-0.1*indices)
response_variable = design_matrix @ signal

model = es.Landweber(design_matrix, response_variable, true_signal = signal, true_noise_level = 1)

iteration_number = 500
model.landweber(iteration_number)

print(f"The weak error is given by {model.weak_error}")
