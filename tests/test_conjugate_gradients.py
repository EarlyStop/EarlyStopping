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
        self.NUMBER_RUNS = 100

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
        # Test if conjugate gradient estimate converges to true signal in the noise free model
        model_supersmooth = es.ConjugateGradients(
            self.design_noise_free,
            self.design_noise_free @ self.signal_supersmooth,
            true_signal=self.signal_supersmooth,
            true_noise_level=0,
            computation_threshold=0,
        )
        model_smooth = es.ConjugateGradients(
            self.design_noise_free,
            self.design_noise_free @ self.signal_smooth,
            true_signal=self.signal_smooth,
            true_noise_level=0,
            computation_threshold=0,
        )
        model_rough = es.ConjugateGradients(
            self.design_noise_free,
            self.design_noise_free @ self.signal_rough,
            true_signal=self.signal_rough,
            true_noise_level=0,
            computation_threshold=0,
        )
        model_supersmooth.iterate(2 * self.sample_size_noise_free)
        model_smooth.iterate(2 * self.sample_size_noise_free)
        model_rough.iterate(2 * self.sample_size_noise_free)
        self.assertAlmostEqual(
            sum((model_supersmooth.get_estimate(model_supersmooth.iteration) - self.signal_supersmooth) ** 2),
            0,
            places=5,
        )
        self.assertAlmostEqual(
            sum((model_smooth.get_estimate(model_smooth.iteration) - self.signal_smooth) ** 2), 0, places=5
        )
        self.assertAlmostEqual(
            sum((model_rough.get_estimate(model_rough.iteration) - self.signal_rough) ** 2), 0, places=5
        )

    def calculate_residual(self, response, design, conjugate_gradient_estimate):
        return np.sum((response - design @ conjugate_gradient_estimate) ** 2)

    def test_residuals(self):
        # Test if the entry in the residuals vector at the early stopping index agrees with the squared residual of the conjugate gradient estimate at the same index
        critical_value = self.noise_level**2 * self.sample_size
        models_supersmooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        max_iteration = self.sample_size
        for run in range(self.NUMBER_RUNS):
            supersmooth_early_stopping_index = models_supersmooth[run].get_discrepancy_stop(
                critical_value, max_iteration
            )
            smooth_early_stopping_index = models_smooth[run].get_discrepancy_stop(critical_value, max_iteration)
            rough_early_stopping_index = models_rough[run].get_discrepancy_stop(critical_value, max_iteration)

            supersmooth_conjugate_gradient_estimate = models_supersmooth[run].get_estimate(
                supersmooth_early_stopping_index
            )
            smooth_conjugate_gradient_estimate = models_smooth[run].get_estimate(smooth_early_stopping_index)
            rough_conjugate_gradient_estimate = models_rough[run].get_estimate(rough_early_stopping_index)

            residual_supersmooth = self.calculate_residual(
                models_supersmooth[run].response,
                models_supersmooth[run].design,
                supersmooth_conjugate_gradient_estimate,
            )
            residual_smooth = self.calculate_residual(
                models_smooth[run].response,
                models_smooth[run].design,
                smooth_conjugate_gradient_estimate,
            )
            residual_rough = self.calculate_residual(
                models_rough[run].response,
                models_rough[run].design,
                rough_conjugate_gradient_estimate,
            )
            self.assertAlmostEqual(
                residual_supersmooth,
                models_supersmooth[run].residuals[int(supersmooth_early_stopping_index)],
                places=5,
            )
            self.assertAlmostEqual(
                residual_smooth, models_smooth[run].residuals[int(smooth_early_stopping_index)], places=5
            )
            self.assertAlmostEqual(
                residual_rough, models_rough[run].residuals[int(rough_early_stopping_index)], places=5
            )

    def test_interpolation(self):
        # Test several properties of the interpolated conjugate gradients algorithm
        critical_value = self.noise_level**2 * self.sample_size
        models_supersmooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        max_iteration = self.sample_size
        interpolation = True
        for run in range(self.NUMBER_RUNS):
            supersmooth_early_stopping_index = models_supersmooth[run].get_discrepancy_stop(
                critical_value, max_iteration, interpolation
            )
            smooth_early_stopping_index = models_smooth[run].get_discrepancy_stop(
                critical_value, max_iteration, interpolation
            )
            rough_early_stopping_index = models_rough[run].get_discrepancy_stop(
                critical_value, max_iteration, interpolation
            )
            interpolated_residual_supersmooth = models_supersmooth[run].get_residual(supersmooth_early_stopping_index)
            interpolated_residual_smooth = models_smooth[run].get_residual(smooth_early_stopping_index)
            interpolated_residual_rough = models_rough[run].get_residual(rough_early_stopping_index)

            # Test if the interpolated squared residual at the discrepancy stopping index agrees with the critical value
            if supersmooth_early_stopping_index < max_iteration:
                self.assertAlmostEqual(interpolated_residual_supersmooth, critical_value, places=5)
            if smooth_early_stopping_index < max_iteration:
                self.assertAlmostEqual(interpolated_residual_smooth, critical_value, places=5)
            if rough_early_stopping_index < max_iteration:
                self.assertAlmostEqual(interpolated_residual_rough, critical_value, places=5)

            interpolated_residual_supersmooth_via_estimator = self.calculate_residual(
                models_supersmooth[run].response,
                models_supersmooth[run].design,
                models_supersmooth[run].get_estimate(supersmooth_early_stopping_index),
            )
            interpolated_residual_smooth_via_estimator = self.calculate_residual(
                models_smooth[run].response,
                models_smooth[run].design,
                models_smooth[run].get_estimate(smooth_early_stopping_index),
            )
            interpolated_residual_rough_via_estimator = self.calculate_residual(
                models_rough[run].response,
                models_rough[run].design,
                models_rough[run].get_estimate(rough_early_stopping_index),
            )

            # Test if the interpolated squared residual at the discrepancy stopping index agrees with the squared residual of the conjugate gradient estimate at the same index
            self.assertAlmostEqual(
                interpolated_residual_supersmooth_via_estimator, interpolated_residual_supersmooth, places=5
            )
            self.assertAlmostEqual(interpolated_residual_smooth_via_estimator, interpolated_residual_smooth, places=5)
            self.assertAlmostEqual(interpolated_residual_rough_via_estimator, interpolated_residual_rough, places=5)

    def test_early_stopping_index(self):
        # Test if the discrepancy stopping index for the model without interpolation agrees with the rounded up discrepancy stopping index for the interpolated model
        critical_value = self.noise_level**2 * self.sample_size
        models_supersmooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        max_iteration = self.sample_size
        for run in range(self.NUMBER_RUNS):
            early_stopping_index_supersmooth_interpolated = models_supersmooth[run].get_discrepancy_stop(
                critical_value, max_iteration, True
            )
            early_stopping_index_smooth_interpolated = models_smooth[run].get_discrepancy_stop(
                critical_value, max_iteration, True
            )
            early_stopping_index_rough_interpolated = models_rough[run].get_discrepancy_stop(
                critical_value, max_iteration, True
            )
            early_stopping_index_supersmooth_noninterpolated = models_supersmooth[run].get_discrepancy_stop(
                critical_value, max_iteration, False
            )
            early_stopping_index_smooth_noninterpolated = models_smooth[run].get_discrepancy_stop(
                critical_value, max_iteration, False
            )
            early_stopping_index_rough_noninterpolated = models_rough[run].get_discrepancy_stop(
                critical_value, max_iteration, False
            )
            self.assertAlmostEqual(
                np.ceil(early_stopping_index_supersmooth_interpolated),
                early_stopping_index_supersmooth_noninterpolated,
                places=5,
            )
            self.assertAlmostEqual(
                np.ceil(early_stopping_index_smooth_interpolated),
                early_stopping_index_smooth_noninterpolated,
                places=5,
            )
            self.assertAlmostEqual(
                np.ceil(early_stopping_index_rough_interpolated),
                early_stopping_index_rough_noninterpolated,
                places=5,
            )

    def test_empirical_oracles(self):
        # Test if the risks at the interpolated empirical oracles are smaller or equal to all risks along the integer iteration path
        # and, in addition, for supersmooth if they are smaller or equal to all risks along a real-valued grid of iteration indices

        models_supersmooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_supersmooth[:, i],
                true_signal=self.signal_supersmooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_smooth = [
            es.ConjugateGradients(
                self.design,
                self.observation_smooth[:, i],
                true_signal=self.signal_smooth,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_rough = [
            es.ConjugateGradients(
                self.design,
                self.observation_rough[:, i],
                true_signal=self.signal_rough,
                true_noise_level=self.noise_level,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        max_iteration = self.sample_size
        for run in range(self.NUMBER_RUNS):
            strong_empirical_oracle_supersmooth = models_supersmooth[run].get_strong_empirical_oracle(
                max_iteration, True
            )
            strong_empirical_oracle_smooth = models_smooth[run].get_strong_empirical_oracle(max_iteration, True)
            strong_empirical_oracle_rough = models_rough[run].get_strong_empirical_oracle(max_iteration, True)
            strong_empirical_oracle_risk_supersmooth = models_supersmooth[run].get_strong_empirical_risk(
                strong_empirical_oracle_supersmooth
            )
            strong_empirical_oracle_risk_smooth = models_smooth[run].get_strong_empirical_risk(
                strong_empirical_oracle_smooth
            )
            strong_empirical_oracle_risk_rough = models_rough[run].get_strong_empirical_risk(
                strong_empirical_oracle_rough
            )

            self.assertTrue(
                all(
                    strong_empirical_oracle_risk_supersmooth <= risk
                    for risk in models_supersmooth[run].strong_empirical_risk
                )
            )
            self.assertTrue(
                all(strong_empirical_oracle_risk_smooth <= risk for risk in models_smooth[run].strong_empirical_risk)
            )
            self.assertTrue(
                all(strong_empirical_oracle_risk_rough <= risk for risk in models_rough[run].strong_empirical_risk)
            )

            weak_empirical_oracle_supersmooth = models_supersmooth[run].get_weak_empirical_oracle(max_iteration, True)
            weak_empirical_oracle_smooth = models_smooth[run].get_weak_empirical_oracle(max_iteration, True)
            weak_empirical_oracle_rough = models_rough[run].get_weak_empirical_oracle(max_iteration, True)
            weak_empirical_oracle_risk_supersmooth = models_supersmooth[run].get_weak_empirical_risk(
                weak_empirical_oracle_supersmooth
            )
            weak_empirical_oracle_risk_smooth = models_smooth[run].get_weak_empirical_risk(
                weak_empirical_oracle_smooth
            )
            weak_empirical_oracle_risk_rough = models_rough[run].get_weak_empirical_risk(weak_empirical_oracle_rough)

            self.assertTrue(
                all(
                    weak_empirical_oracle_risk_supersmooth <= risk
                    for risk in models_supersmooth[run].weak_empirical_risk
                )
            )
            self.assertTrue(
                all(weak_empirical_oracle_risk_smooth <= risk for risk in models_smooth[run].weak_empirical_risk)
            )
            self.assertTrue(
                all(weak_empirical_oracle_risk_rough <= risk for risk in models_rough[run].weak_empirical_risk)
            )

            interpolated_weak_risk_supersmooth = []
            step_size = 0.1
            for iteration in np.arange(0, max_iteration + step_size, step_size):
                interpolated_weak_risk_supersmooth = np.append(
                    interpolated_weak_risk_supersmooth,
                    models_supersmooth[run].get_weak_empirical_risk(iteration),
                )
            interpolated_weak_risk_supersmooth = [
                risk for risk in interpolated_weak_risk_supersmooth if risk is not None
            ]
            self.assertTrue(
                all(weak_empirical_oracle_risk_supersmooth <= risk for risk in interpolated_weak_risk_supersmooth)
            )
