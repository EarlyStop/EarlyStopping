import unittest
import numpy as np
from scipy.sparse import dia_matrix
import EarlyStopping as es


class TestConjugateGradients(unittest.TestCase):
    """Tests for the conjugate gradients algorithm"""

    def setUp(self):
        # setUp is a class from unittest
        # Simulate data

        # Number of Monte-Carlo simulations
        self.NUMBER_RUNS = 20

        # Create diagonal design matrices
        self.sample_size = 10000
        indices = np.arange(self.sample_size) + 1
        self.design_matrix = dia_matrix(np.diag(1 / (np.sqrt(indices))))

        # Create signals from Stankewitz (2020)
        self.signal_supersmooth = 5 * np.exp(-0.1 * indices)
        self.signal_smooth = 5000 * np.abs(np.sin(0.01 * indices)) * indices ** (-1.6)
        self.signal_rough = 250 * np.abs(np.sin(0.002 * indices)) * indices ** (-0.8)

        # Create observations
        self.NOISE_LEVEL = 0.01
        noise = np.random.normal(0, self.NOISE_LEVEL, (self.sample_size, self.NUMBER_RUNS))
        self.observation_supersmooth = noise + (self.design_matrix @ self.signal_supersmooth)[:, None]
        self.observation_smooth = noise + (self.design_matrix @ self.signal_smooth)[:, None]
        self.observation_rough = noise + (self.design_matrix @ self.signal_rough)[:, None]

    def calculate_residual(self, response_variable, design_matrix, conjugate_gradient_estimate):
        return np.sum((response_variable - design_matrix @ conjugate_gradient_estimate) ** 2)

    def test_residuals(self):
        models_supersmooth = [
            es.ConjugateGradients(
                self.design_matrix,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.NOISE_LEVEL,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design_matrix,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.NOISE_LEVEL,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design_matrix,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.NOISE_LEVEL,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        for run in range(self.NUMBER_RUNS):
            models_supersmooth[run].conjugate_gradients_to_early_stop(self.NUMBER_RUNS)
            models_smooth[run].conjugate_gradients_to_early_stop(self.NUMBER_RUNS)
            models_rough[run].conjugate_gradients_to_early_stop(self.NUMBER_RUNS)
            residual_supersmooth = self.calculate_residual(
                models_supersmooth[run].response_variable,
                models_supersmooth[run].design_matrix,
                models_supersmooth[run].conjugate_gradient_estimate,
            )
            residual_smooth = self.calculate_residual(
                models_smooth[run].response_variable,
                models_smooth[run].design_matrix,
                models_smooth[run].conjugate_gradient_estimate,
            )
            residual_rough = self.calculate_residual(
                models_rough[run].response_variable,
                models_rough[run].design_matrix,
                models_rough[run].conjugate_gradient_estimate,
            )
            self.assertAlmostEqual(
                residual_supersmooth,
                models_supersmooth[run].residuals[int(models_supersmooth[run].early_stopping_index)],
                places=5,
            )
            self.assertAlmostEqual(
                residual_smooth, models_smooth[run].residuals[int(models_smooth[run].early_stopping_index)], places=5
            )
            self.assertAlmostEqual(
                residual_rough, models_rough[run].residuals[int(models_rough[run].early_stopping_index)], places=5
            )

    def calculate_interpolated_residual(self, residuals, early_stopping_index):
        early_stopping_index_ceil = int(np.ceil(early_stopping_index))
        early_stopping_index_floor = int(np.floor(early_stopping_index))
        alpha = early_stopping_index - early_stopping_index_floor
        interpolated_residual = (1 - alpha) ** 2 * residuals[early_stopping_index_floor] + (
            1 - (1 - alpha) ** 2
        ) * residuals[early_stopping_index_ceil]
        return interpolated_residual

    def test_interpolation(self):
        models_supersmooth = [
            es.ConjugateGradients(
                self.design_matrix,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.NOISE_LEVEL,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design_matrix,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.NOISE_LEVEL,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design_matrix,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.NOISE_LEVEL,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        for run in range(self.NUMBER_RUNS):
            models_supersmooth[run].conjugate_gradients_to_early_stop(self.NUMBER_RUNS)
            models_smooth[run].conjugate_gradients_to_early_stop(self.NUMBER_RUNS)
            models_rough[run].conjugate_gradients_to_early_stop(self.NUMBER_RUNS)
            interpolated_residual_supersmooth = self.calculate_interpolated_residual(
                models_supersmooth[run].residuals, models_supersmooth[run].early_stopping_index
            )
            interpolated_residual_smooth = self.calculate_interpolated_residual(
                models_supersmooth[run].residuals, models_supersmooth[run].early_stopping_index
            )
            interpolated_residual_rough = self.calculate_interpolated_residual(
                models_supersmooth[run].residuals, models_supersmooth[run].early_stopping_index
            )
            self.assertAlmostEqual(interpolated_residual_supersmooth, models_supersmooth[run].critical_value, places=5)
            self.assertAlmostEqual(interpolated_residual_smooth, models_smooth[run].critical_value, places=5)
            self.assertAlmostEqual(interpolated_residual_rough, models_rough[run].critical_value, places=5)
            interpolated_residual_supersmooth_via_estimator = self.calculate_residual(
                models_supersmooth[run].response_variable,
                models_supersmooth[run].design_matrix,
                models_supersmooth[run].conjugate_gradient_estimate,
            )
            interpolated_residual_smooth_via_estimator = self.calculate_residual(
                models_smooth[run].response_variable,
                models_smooth[run].design_matrix,
                models_smooth[run].conjugate_gradient_estimate,
            )
            interpolated_residual_rough_via_estimator = self.calculate_residual(
                models_rough[run].response_variable,
                models_rough[run].design_matrix,
                models_rough[run].conjugate_gradient_estimate,
            )
            self.assertAlmostEqual(
                interpolated_residual_supersmooth_via_estimator, interpolated_residual_supersmooth, places=5
            )
            self.assertAlmostEqual(interpolated_residual_smooth_via_estimator, interpolated_residual_smooth, places=5)
            self.assertAlmostEqual(interpolated_residual_rough_via_estimator, interpolated_residual_rough, places=5)
