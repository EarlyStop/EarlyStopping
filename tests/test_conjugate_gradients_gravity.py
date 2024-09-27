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
            computation_threshold=0,
        )
        model.iterate(2 * self.sample_size)
        self.assertAlmostEqual(sum((model.get_estimate(model.iteration) - self.signal) ** 2), 0, places=5)

    def calculate_residual(self, response, design, conjugate_gradient_estimate):
        return np.sum((response - design @ conjugate_gradient_estimate) ** 2)

    def test_residuals(self):
        # Test if the entry in the residuals vector at the discrepancy stopping index agrees with the squared residual of the conjugate gradient estimate at the same index
        critical_value = (np.sqrt(self.sample_size) + self.sample_size) * (self.noise_level**2)
        models = [
            es.ConjugateGradients(
                self.design,
                self.observation[:, i],
                true_signal=self.signal,
                true_noise_level=self.noise_level,
                computation_threshold=0,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        max_iteration = self.sample_size
        for run in range(self.NUMBER_RUNS):
            early_stopping_index = models[run].get_discrepancy_stop(critical_value, max_iteration)
            if early_stopping_index is None:
                early_stopping_index = max_iteration
            residual = self.calculate_residual(
                models[run].response,
                models[run].design,
                models[run].get_estimate(early_stopping_index),
            )

            self.assertAlmostEqual(
                residual,
                models[run].residuals[int(early_stopping_index)],
                places=5,
            )

    def test_interpolation(self):
        # Test several properties of the interpolated conjugate gradients algorithm
        critical_value = (np.sqrt(self.sample_size) + self.sample_size) * (self.noise_level**2)
        models = [
            es.ConjugateGradients(
                self.design,
                self.observation[:, i],
                true_signal=self.signal,
                true_noise_level=self.noise_level,
                computation_threshold=0,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        max_iteration = self.sample_size
        interpolation = True
        for run in range(self.NUMBER_RUNS):
            early_stopping_index = models[run].get_discrepancy_stop(critical_value, max_iteration, interpolation)
            if early_stopping_index is None:
                early_stopping_index = max_iteration
            interpolated_residual = models[run].get_residual(early_stopping_index)

            # Test if the interpolated squared residual at the discrepancy stopping index agrees with the critical value
            if early_stopping_index < max_iteration:
                self.assertAlmostEqual(interpolated_residual, critical_value, places=5)

            interpolated_residual_via_estimator = self.calculate_residual(
                models[run].response,
                models[run].design,
                models[run].get_estimate(early_stopping_index),
            )

            # Test if the interpolated squared residual at the discrepancy stopping index agrees with the squared residual of the conjugate gradient estimate at the same index
            self.assertAlmostEqual(interpolated_residual_via_estimator, interpolated_residual, places=5)

    def test_early_stopping_index(self):
        # Test if the discrepancy stopping index for the model without interpolation agrees with the rounded up discrepancy stopping index for the interpolated model
        critical_value = (np.sqrt(self.sample_size) + self.sample_size) * (self.noise_level**2)
        models = [
            es.ConjugateGradients(
                self.design,
                self.observation[:, i],
                true_signal=self.signal,
                true_noise_level=self.noise_level,
                computation_threshold=0,
            )
            for i in range(self.NUMBER_RUNS)
        ]

        max_iteration = self.sample_size
        for run in range(self.NUMBER_RUNS):
            early_stopping_index_interpolated = models[run].get_discrepancy_stop(critical_value, max_iteration, True)
            if early_stopping_index_interpolated is None:
                early_stopping_index_interpolated = max_iteration
            early_stopping_index_noninterpolated = models[run].get_discrepancy_stop(
                critical_value, max_iteration, False
            )
            if early_stopping_index_noninterpolated is None:
                early_stopping_index_noninterpolated = max_iteration
            self.assertAlmostEqual(
                np.ceil(early_stopping_index_interpolated),
                early_stopping_index_noninterpolated,
                places=5,
            )
