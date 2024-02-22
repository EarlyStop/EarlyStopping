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
        self.sample_size = 1000
        indices = np.arange(self.sample_size) + 1
        self.design = dia_matrix(np.diag(1 / (np.sqrt(indices))))

        # Create signals from Stankewitz (2020)
        self.signal_supersmooth = 5 * np.exp(-0.1 * indices)
        self.signal_smooth = 5000 * np.abs(np.sin(0.01 * indices)) * indices ** (-1.6)
        self.signal_rough = 250 * np.abs(np.sin(0.002 * indices)) * indices ** (-0.8)

        # Create observations
        self.noise_level = 0.01
        noise = np.random.normal(0, self.noise_level, (self.sample_size, self.NUMBER_RUNS))
        self.observation_supersmooth = noise + (self.design @ self.signal_supersmooth)[:, None]
        self.observation_smooth = noise + (self.design @ self.signal_smooth)[:, None]
        self.observation_rough = noise + (self.design @ self.signal_rough)[:, None]

        # Create noise free model
        self.sample_size_noise_free = self.sample_size
        self.design_noise_free = np.random.rand(self.sample_size_noise_free, self.sample_size_noise_free)
        design_noise_free_diagonal = np.sum(np.abs(self.design_noise_free), axis=1)
        np.fill_diagonal(self.design_noise_free, design_noise_free_diagonal)

    def test_noise_free_model(self):
        model_supersmooth = es.ConjugateGradients(
            self.design_noise_free,
            self.design_noise_free @ self.signal_supersmooth,
            true_signal=self.signal_supersmooth,
            true_noise_level=0,
            interpolation=False,
        )
        model_smooth = es.ConjugateGradients(
            self.design_noise_free,
            self.design_noise_free @ self.signal_smooth,
            true_signal=self.signal_smooth,
            true_noise_level=0,
            interpolation=False,
        )
        model_rough = es.ConjugateGradients(
            self.design_noise_free,
            self.design_noise_free @ self.signal_rough,
            true_signal=self.signal_rough,
            true_noise_level=0,
            interpolation=False,
        )
        model_supersmooth.iterate(2 * self.sample_size_noise_free)
        model_smooth.iterate(2 * self.sample_size_noise_free)
        model_rough.iterate(2 * self.sample_size_noise_free)
        self.assertAlmostEqual(
            sum((model_supersmooth.conjugate_gradient_estimate - self.signal_supersmooth) ** 2), 0, places=5
        )
        self.assertAlmostEqual(sum((model_smooth.conjugate_gradient_estimate - self.signal_smooth) ** 2), 0, places=5)
        self.assertAlmostEqual(sum((model_rough.conjugate_gradient_estimate - self.signal_rough) ** 2), 0, places=5)

    def calculate_residual(self, response, design, conjugate_gradient_estimate):
        return np.sum((response - design @ conjugate_gradient_estimate) ** 2)

    def test_residuals(self):
        models_supersmooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.noise_level,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.noise_level,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.noise_level,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        for run in range(self.NUMBER_RUNS):
            models_supersmooth[run].discrepancy_stop(self.NUMBER_RUNS)
            models_smooth[run].discrepancy_stop(self.NUMBER_RUNS)
            models_rough[run].discrepancy_stop(self.NUMBER_RUNS)
            residual_supersmooth = self.calculate_residual(
                models_supersmooth[run].response,
                models_supersmooth[run].design,
                models_supersmooth[run].conjugate_gradient_estimate,
            )
            residual_smooth = self.calculate_residual(
                models_smooth[run].response,
                models_smooth[run].design,
                models_smooth[run].conjugate_gradient_estimate,
            )
            residual_rough = self.calculate_residual(
                models_rough[run].response,
                models_rough[run].design,
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

    def test_interpolation(self):
        models_supersmooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.noise_level,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.noise_level,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.noise_level,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        for run in range(self.NUMBER_RUNS):
            models_supersmooth[run].discrepancy_stop(self.NUMBER_RUNS)
            models_smooth[run].discrepancy_stop(self.NUMBER_RUNS)
            models_rough[run].discrepancy_stop(self.NUMBER_RUNS)
            interpolated_residual_supersmooth = models_supersmooth[run].calculate_interpolated_residual(
                models_supersmooth[run].early_stopping_index
            )
            interpolated_residual_smooth = models_supersmooth[run].calculate_interpolated_residual(
                models_supersmooth[run].early_stopping_index
            )
            interpolated_residual_rough = models_supersmooth[run].calculate_interpolated_residual(
                models_supersmooth[run].early_stopping_index
            )
            self.assertAlmostEqual(interpolated_residual_supersmooth, models_supersmooth[run].critical_value, places=5)
            self.assertAlmostEqual(interpolated_residual_smooth, models_smooth[run].critical_value, places=5)
            self.assertAlmostEqual(interpolated_residual_rough, models_rough[run].critical_value, places=5)
            interpolated_residual_supersmooth_via_estimator = self.calculate_residual(
                models_supersmooth[run].response,
                models_supersmooth[run].design,
                models_supersmooth[run].conjugate_gradient_estimate,
            )
            interpolated_residual_smooth_via_estimator = self.calculate_residual(
                models_smooth[run].response,
                models_smooth[run].design,
                models_smooth[run].conjugate_gradient_estimate,
            )
            interpolated_residual_rough_via_estimator = self.calculate_residual(
                models_rough[run].response,
                models_rough[run].design,
                models_rough[run].conjugate_gradient_estimate,
            )
            self.assertAlmostEqual(
                interpolated_residual_supersmooth_via_estimator, interpolated_residual_supersmooth, places=5
            )
            self.assertAlmostEqual(interpolated_residual_smooth_via_estimator, interpolated_residual_smooth, places=5)
            self.assertAlmostEqual(interpolated_residual_rough_via_estimator, interpolated_residual_rough, places=5)
            interpolated_strong_empirical_error_supersmooth = models_supersmooth[
                run
            ].calculate_interpolated_strong_empirical_error(models_supersmooth[run].early_stopping_index)
            interpolated_strong_empirical_error_smooth = models_smooth[
                run
            ].calculate_interpolated_strong_empirical_error(models_smooth[run].early_stopping_index)
            interpolated_strong_empirical_error_rough = models_rough[
                run
            ].calculate_interpolated_strong_empirical_error(models_rough[run].early_stopping_index)
            interpolated_strong_empirical_error_supersmooth_via_estimator = np.sum(
                (models_supersmooth[run].conjugate_gradient_estimate - models_supersmooth[run].true_signal) ** 2
            )
            interpolated_strong_empirical_error_smooth_via_estimator = np.sum(
                (models_smooth[run].conjugate_gradient_estimate - models_smooth[run].true_signal) ** 2
            )
            interpolated_strong_empirical_error_rough_via_estimator = np.sum(
                (models_rough[run].conjugate_gradient_estimate - models_rough[run].true_signal) ** 2
            )
            self.assertAlmostEqual(
                interpolated_strong_empirical_error_supersmooth_via_estimator,
                interpolated_strong_empirical_error_supersmooth,
                places=5,
            )
            self.assertAlmostEqual(
                interpolated_strong_empirical_error_smooth_via_estimator,
                interpolated_strong_empirical_error_smooth,
                places=5,
            )
            self.assertAlmostEqual(
                interpolated_strong_empirical_error_rough_via_estimator,
                interpolated_strong_empirical_error_rough,
                places=5,
            )
            interpolated_weak_empirical_error_supersmooth = models_supersmooth[
                run
            ].calculate_interpolated_weak_empirical_error(models_supersmooth[run].early_stopping_index)
            interpolated_weak_empirical_error_smooth = models_smooth[run].calculate_interpolated_weak_empirical_error(
                models_smooth[run].early_stopping_index
            )
            interpolated_weak_empirical_error_rough = models_rough[run].calculate_interpolated_weak_empirical_error(
                models_rough[run].early_stopping_index
            )
            interpolated_weak_empirical_error_supersmooth_via_estimator = np.sum(
                (
                    models_supersmooth[run].design
                    @ (models_supersmooth[run].conjugate_gradient_estimate - models_supersmooth[run].true_signal)
                )
                ** 2
            )
            interpolated_weak_empirical_error_smooth_via_estimator = np.sum(
                (
                    models_smooth[run].design
                    @ (models_smooth[run].conjugate_gradient_estimate - models_smooth[run].true_signal)
                )
                ** 2
            )
            interpolated_weak_empirical_error_rough_via_estimator = np.sum(
                (
                    models_rough[run].design
                    @ (models_rough[run].conjugate_gradient_estimate - models_rough[run].true_signal)
                )
                ** 2
            )
            self.assertAlmostEqual(
                interpolated_weak_empirical_error_supersmooth_via_estimator,
                interpolated_weak_empirical_error_supersmooth,
                places=5,
            )
            self.assertAlmostEqual(
                interpolated_weak_empirical_error_smooth_via_estimator,
                interpolated_weak_empirical_error_smooth,
                places=5,
            )
            self.assertAlmostEqual(
                interpolated_weak_empirical_error_rough_via_estimator,
                interpolated_weak_empirical_error_rough,
                places=5,
            )
