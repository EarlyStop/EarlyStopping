import unittest
import numpy as np
from scipy.io import loadmat
import EarlyStopping as es


class TestConjugateGradientsGravity(unittest.TestCase):
    """Tests for the conjugate gradients algorithm"""

    def setUp(self):
        # setUp is a class from unittest

        # Python implementation of the gravity test problem from the regtools toolbox, see `Hansen (2008) <http://people.compute.dtu.dk/pcha/Regutools/RTv4manual.pdf>`_ for details
        self.sample_size = 2**9
        a = 0
        b = 1
        d = 0.25  # Parameter controlling the ill-posedness: the larger, the more ill-posed, default in regtools: d = 0.25

        t = (np.arange(1, self.sample_size + 1) - 0.5) / self.sample_size
        s = ((np.arange(1, self.sample_size + 1) - 0.5) * (b - a)) / self.sample_size
        T, S = np.meshgrid(t, s)

        self.design = (
            (1 / self.sample_size)
            * d
            * (d**2 * np.ones((self.sample_size, self.sample_size)) + (S - T) ** 2) ** (-(3 / 2))
        )
        self.signal = np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t)
        self.design_times_signal = self.design @ self.signal

        # Set parameters
        self.parameter_size = self.sample_size
        self.max_iter = self.sample_size
        self.noise_level = 10 ** (-2)
        self.critical_value = self.sample_size * (self.noise_level**2)

        # Specify number of Monte Carlo runs
        self.NUMBER_RUNS = 100

        # Create observations
        noise = np.random.normal(0, self.noise_level, (self.sample_size, self.NUMBER_RUNS))
        self.observation = noise + self.design_times_signal[:, None]

    def test_noise_free_model(self):
        # Test if conjugate gradient estimate converges to true signal in the noise free model
        model = es.ConjugateGradients(
            self.design,
            self.design @ self.signal,
            true_signal=self.signal,
            true_noise_level=0,
            interpolation=False,
            computation_threshold=0,
        )
        model.iterate(2 * self.sample_size)
        self.assertAlmostEqual(sum((model.conjugate_gradient_estimate - self.signal) ** 2), 0, places=5)

    def calculate_residual(self, response, design, conjugate_gradient_estimate):
        return np.sum((response - design @ conjugate_gradient_estimate) ** 2)

    def test_residuals(self):
        # Test if the entry in the residuals vector at the discrepancy stopping index agrees with the squared residual of the conjugate gradient estimate at the same index
        models = [
            es.ConjugateGradients(
                self.design,
                self.observation[:, i],
                true_signal=self.signal,
                true_noise_level=self.noise_level,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        for run in range(self.NUMBER_RUNS):
            models[run].discrepancy_stop(self.max_iter)
            residual = self.calculate_residual(
                models[run].response,
                models[run].design,
                models[run].conjugate_gradient_estimate,
            )

            self.assertAlmostEqual(
                residual,
                models[run].residuals[int(models[run].early_stopping_index)],
                places=5,
            )

    def test_interpolation(self):
        # Test several properties of the interpolated conjugate gradients algorithm
        models = [
            es.ConjugateGradients(
                self.design,
                self.observation[:, i],
                true_signal=self.signal,
                true_noise_level=self.noise_level,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        for run in range(self.NUMBER_RUNS):
            models[run].discrepancy_stop(self.max_iter)
            interpolated_residual = models[run].calculate_interpolated_residual(models[run].early_stopping_index)

            # Test if the interpolated squared residual at the discrepancy stopping index agrees with the critical value
            if models[run].early_stopping_index < self.max_iter:
                self.assertAlmostEqual(interpolated_residual, models[run].critical_value, places=5)

            interpolated_residual_via_estimator = self.calculate_residual(
                models[run].response,
                models[run].design,
                models[run].conjugate_gradient_estimate,
            )

            # Test if the interpolated squared residual at the discrepancy stopping index agrees with the squared residual of the conjugate gradient estimate at the same index
            self.assertAlmostEqual(interpolated_residual_via_estimator, interpolated_residual, places=5)

            interpolated_strong_empirical_error = models[run].calculate_interpolated_strong_empirical_error(
                models[run].early_stopping_index
            )
            interpolated_strong_empirical_error_via_estimator = np.sum(
                (models[run].conjugate_gradient_estimate - models[run].true_signal) ** 2
            )

            # Test if the interpolated strong empirical error at the discrepancy stopping index agrees with the strong empirical error of the conjugate gradient estimate at the same index
            self.assertAlmostEqual(
                interpolated_strong_empirical_error_via_estimator,
                interpolated_strong_empirical_error,
                places=4,
            )

            interpolated_weak_empirical_error = models[run].calculate_interpolated_weak_empirical_error(
                models[run].early_stopping_index
            )
            interpolated_weak_empirical_error_via_estimator = np.sum(
                (models[run].design @ (models[run].conjugate_gradient_estimate - models[run].true_signal)) ** 2
            )

            # Test if the interpolated weak empirical error at the discrepancy stopping index agrees with the weak empirical error of the conjugate gradient estimate at the same index
            self.assertAlmostEqual(
                interpolated_weak_empirical_error_via_estimator,
                interpolated_weak_empirical_error,
                places=5,
            )

    def test_early_stopping_index(self):
        # Test if the discrepancy stopping index for the model without interpolation agrees with the rounded up discrepancy stopping index for the interpolated model
        models_interpolated = [
            es.ConjugateGradients(
                self.design,
                self.observation[:, i],
                true_signal=self.signal,
                true_noise_level=self.noise_level,
                interpolation=True,
            )
            for i in range(self.NUMBER_RUNS)
        ]
        models_noninterpolated = [
            es.ConjugateGradients(
                self.design,
                self.observation[:, i],
                true_signal=self.signal,
                true_noise_level=self.noise_level,
                interpolation=False,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        for run in range(self.NUMBER_RUNS):
            models_interpolated[run].discrepancy_stop(self.max_iter)
            models_noninterpolated[run].discrepancy_stop(self.max_iter)
            early_stopping_index_interpolated = models_interpolated[run].early_stopping_index
            early_stopping_index_noninterpolated = models_noninterpolated[run].early_stopping_index
            self.assertAlmostEqual(
                np.ceil(early_stopping_index_interpolated),
                early_stopping_index_noninterpolated,
                places=5,
            )
