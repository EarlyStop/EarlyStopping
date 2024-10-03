import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import toeplitz
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import svds

from .landweber import Landweber
from .conjugate_gradients import ConjugateGradients

# import matplotlib.pyplot as plt
# from matplotlib.ticker import FixedLocator
# import seaborn as sns
# import pandas as pd


class SimulationData:

    def diagonal_data(sample_size, type="supersmooth"):
        indices = np.arange(sample_size) + 1
        design = dia_matrix(np.diag(1 / np.sqrt(indices)))
        if type == "supersmooth":
            true_signal = 5 * np.exp(-0.1 * indices)
        elif type == "smooth":
            true_signal = 5000 * np.abs(np.sin(0.01 * indices)) * indices ** (-1.6)
        elif type == "rough":
            true_signal = 250 * np.abs(np.sin(0.002 * indices)) * indices ** (-0.8)
        else:
            ValueError("Currently, only supersmooth, smooth and rough are supported.")

        response_noiseless = design @ true_signal
        return design, response_noiseless, true_signal

    def gravity(sample_size, a=0, b=1, d=0.25):
        # Parameter controlling the ill-posedness: the larger, the more ill-posed, default in regtools: d = 0.25

        t = (np.arange(1, sample_size + 1) - 0.5) / sample_size
        s = ((np.arange(1, sample_size + 1) - 0.5) * (b - a)) / sample_size
        T, S = np.meshgrid(t, s)

        design = (1 / sample_size) * d * (d**2 * np.ones((sample_size, sample_size)) + (S - T) ** 2) ** (-(3 / 2))
        true_signal = np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t)

        response_noiseless = design @ true_signal
        return design, response_noiseless, true_signal

    def heat(sample_size, kappa=1):
        """
        Test problem: inverse heat equation.

        Parameters:
        sample_size (int): Number of discretization points.
        kappa (float): Controls the ill-conditioning of the matrix.

        Returns:
        design (numpy.ndarray): The matrix representing the integral operator.
        response_noiseless (numpy.ndarray): The right-hand side vector.
        true_signal (numpy.ndarray): The exact solution vector.
        """
        # Initialization
        h = 1 / sample_size
        t = np.linspace(h / 2, 1, sample_size)  # midpoints
        c = h / (2 * kappa * np.sqrt(np.pi))
        d = 1 / (4 * kappa**2)

        # Compute the matrix A
        k = c * t ** (-1.5) * np.exp(-d / t)
        r = np.zeros(len(t))
        r[0] = k[0]
        design = toeplitz(k, r)

        # Compute the vectors x and b
        true_signal = np.zeros(sample_size)
        for i in range(1, sample_size // 2 + 1):
            ti = i * 20 / sample_size
            if ti < 2:
                true_signal[i - 1] = 0.75 * (ti**2) / 4
            elif ti < 3:
                true_signal[i - 1] = 0.75 + (ti - 2) * (3 - ti)
            else:
                true_signal[i - 1] = 0.75 * np.exp(-(ti - 3) * 2)

        true_signal[sample_size // 2 + 1 :] = 0

        response_noiseless = design @ true_signal

        return design, response_noiseless, true_signal

    def deriv2(sample_size, example=1):
        # Initialize variables and compute coefficients
        h = 1 / sample_size
        sqh = np.sqrt(h)
        h32 = h * sqh
        h2 = h**2
        sqhi = 1 / sqh
        t = 2 / 3
        design = np.zeros((sample_size, sample_size))

        # Compute the matrix A
        for i in range(1, sample_size + 1):
            design[i - 1, i - 1] = h2 * ((i**2 - i + 0.25) * h - (i - t))
            for j in range(1, i):
                design[i - 1, j - 1] = h2 * (j - 0.5) * ((i - 0.5) * h - 1)
        design = design + np.tril(design, -1).T

        # Compute the right-hand side vector b
        response_noiseless = np.zeros(sample_size)
        if example == 1:
            for i in range(1, sample_size + 1):
                response_noiseless[i - 1] = h32 * (i - 0.5) * ((i**2 + (i - 1) ** 2) * h2 / 2 - 1) / 6
        elif example == 2:
            ee = 1 - np.exp(1)
            for i in range(1, sample_size + 1):
                response_noiseless[i - 1] = sqhi * (np.exp(i * h) - np.exp((i - 1) * h) + ee * (i - 0.5) * h2 - h)
        elif example == 3:
            if sample_size % 2 != 0:
                raise ValueError("Order n must be even")
            else:
                for i in range(1, sample_size // 2 + 1):
                    s1 = i * h
                    s12 = s1**2
                    s2 = (i - 1) * h
                    s22 = s2**2
                    response_noiseless[i - 1] = sqhi * (s12 + s22 - 1.5) * (s12 - s22) / 24
                for i in range(sample_size // 2 + 1, sample_size + 1):
                    s1 = i * h
                    s12 = s1**2
                    s2 = (i - 1) * h
                    s22 = s2**2
                    response_noiseless[i - 1] = (
                        sqhi * (-(s12 + s22) * (s12 - s22) + 4 * (s1**3 - s2**3) - 4.5 * (s12 - s22) + h) / 24
                    )
        else:
            raise ValueError("Illegal value of example")

        # Compute the solution vector x
        true_signal = np.zeros(sample_size)
        if example == 1:
            for i in range(1, sample_size + 1):
                true_signal[i - 1] = h32 * (i - 0.5)
        elif example == 2:
            for i in range(1, sample_size + 1):
                true_signal[i - 1] = sqhi * (np.exp(i * h) - np.exp((i - 1) * h))
        elif example == 3:
            for i in range(1, sample_size // 2 + 1):
                true_signal[i - 1] = sqhi * ((i * h) ** 2 - ((i - 1) * h) ** 2) / 2
            for i in range(sample_size // 2 + 1, sample_size + 1):
                true_signal[i - 1] = sqhi * (h - ((i * h) ** 2 - ((i - 1) * h) ** 2) / 2)

        return design, response_noiseless, true_signal

    def phillips(sample_size):
        # Check if n is a multiple of 4
        if sample_size % 4 != 0:
            raise ValueError("The order n must be a multiple of 4")

        # Compute the matrix A using the toeplitz function
        h = 12 / sample_size
        n4 = sample_size // 4
        c = np.cos(np.arange(-1, n4 + 1) * 4 * np.pi / sample_size)
        r1 = np.zeros(sample_size)
        # Debug: Print array shapes

        r1[:n4] = h + 9 / (h * np.pi**2) * (2 * c[1 : n4 + 1] - c[:n4] - c[2 : n4 + 2])
        r1[n4] = h / 2 + 9 / (h * np.pi**2) * (np.cos(4 * np.pi / sample_size) - 1)
        design = toeplitz(r1)

        # Compute the right-hand side b
        response_noiseless = np.zeros(sample_size)
        c = np.pi / 3

        for i in range(sample_size // 2, sample_size):
            t1 = -6 + (i + 1) * h
            t2 = t1 - h
            response_noiseless[i] = (
                t1 * (6 - abs(t1) / 2)
                + ((3 - abs(t1) / 2) * np.sin(c * t1) - 2 / c * (np.cos(c * t1) - 1)) / c
                - t2 * (6 - abs(t2) / 2)
                - ((3 - abs(t2) / 2) * np.sin(c * t2) - 2 / c * (np.cos(c * t2) - 1)) / c
            )
            response_noiseless[sample_size - i - 1] = response_noiseless[i]
        response_noiseless /= np.sqrt(h)

        # Compute the solution x
        true_signal = np.zeros(sample_size)
        # s = np.arange(0, h * 5 + 10 * np.finfo(float).eps, h)
        # original version (problem with array sizes in np.arange)
        # s = np.range(n4)
        # s = np.arange(0, h * (n4+1), h)
        s = np.linspace(0, 3, n4 + 1, endpoint=True)
        true_signal[2 * n4 : 3 * n4] = (h + np.diff(np.sin(s * c)) / c) / np.sqrt(h)

        true_signal[n4 : 2 * n4] = true_signal[3 * n4 - 1 : 2 * n4 - 1 : -1]

        return design, response_noiseless, true_signal


class SimulationParameters:
    def __init__(
        self,
        design,
        true_signal,
        true_noise_level,
        max_iteration,
        monte_carlo_runs,
        noise=None,
        response_noiseless=None,
        critical_value=None,
        interpolation=False,
        computation_threshold=10 ** (-8),
        cores=5,
    ):

        self.design = design
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.max_iteration = max_iteration
        self.monte_carlo_runs = monte_carlo_runs
        self.noise = noise
        self.response_noiseless = response_noiseless
        self.critical_value = critical_value
        self.interpolation = interpolation
        self.computation_threshold = computation_threshold
        self.cores = cores
        self.validate()

    def validate(self):
        if not isinstance(self.true_signal, np.ndarray):
            raise ValueError("true_signal must be a numpy array.")
        if not (0 <= self.true_noise_level):
            raise ValueError("true_noise_level must be greater than or equal to 0.")
        if not isinstance(self.max_iteration, int) or self.max_iteration < 0:
            raise ValueError("max_iteration must be a nonnegative integer.")
        # add more cases to validate (e.g. for noise)


class SimulationWrapper:
    def __init__(
        self,
        design,
        true_signal=None,
        true_noise_level=None,
        max_iteration=1000,
        monte_carlo_runs=5,
        noise=None,
        response_noiseless=None,
        critical_value=None,
        interpolation=False,
        computation_threshold=10 ** (-8),
        cores=5,
    ):

        self.design = design
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.max_iteration = max_iteration
        self.monte_carlo_runs = monte_carlo_runs
        self.noise = noise
        self.critical_value = critical_value
        self.interpolation = interpolation
        self.computation_threshold = computation_threshold
        self.cores = cores

        if response_noiseless is None:
            self.response_noiseless = self.design @ self.true_signal
        else:
            self.response_noiseless = response_noiseless

        self.sample_size = design.shape[0]

    # def run_simulation(self, learning_rate=None):
    #     info("Running simulation.")
    #     if self.noise is None:
    #         self.noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))
    #     self.response = self.noise + (self.response_noiseless)[:, None]

    #     if learning_rate == "auto":
    #         info("Searching for viable learning rates.")
    #         self.learning_rate = self.__search_learning_rate(search_depth=10)
    #     else:
    #         self.learning_rate = 1

    #     info("Running Monte Carlo simulation.")
    #     self.results = Parallel(n_jobs=self.cores)(
    #         delayed(self.monte_carlo_wrapper)(m) for m in range(self.monte_carlo_runs)
    #     )

    #     (
    #         strong_errors,
    #         weak_errors,
    #         weak_relative_efficiency,
    #         strong_relative_efficiency,
    #         landweber_strong_bias,
    #         landweber_strong_variance,
    #         landweber_strong_error,
    #         landweber_weak_bias,
    #         landweber_weak_variance,
    #         landweber_weak_error,
    #         landweber_residuals,
    #         stopping_index_landweber,
    #         balanced_oracle_weak,
    #         balanced_oracle_strong,
    #     ) = zip(*self.results)

    #     self.__visualise_boxplot(np.vstack(strong_errors), "Strong Empirical Error at $\\tau$")
    #     self.__visualise_boxplot(np.vstack(weak_errors), "Weak Empirical Error at $\\tau$")
    #     self.__visualise_boxplot(
    #         np.vstack(weak_relative_efficiency), quantity_name="Weak relative efficiency at $\\tau$"
    #     )
    #     self.__visualise_boxplot(
    #         np.vstack(strong_relative_efficiency), quantity_name="Strong relative efficiency at $\\tau$"
    #     )

    #     np.vstack(
    #         (np.mean(landweber_residuals, axis=0), np.mean(landweber_strong_bias, axis=0)),
    #     )

    #     self.__visualise_bias_variance_tradeoff(
    #         residuals_mean=np.mean(landweber_residuals, axis=0),
    #         strong_bias2_mean=np.mean(landweber_strong_bias, axis=0),
    #         strong_variance_mean=np.mean(landweber_strong_variance, axis=0),
    #         strong_error_mean=np.mean(landweber_strong_error, axis=0),
    #         weak_bias2_mean=np.mean(landweber_weak_bias, axis=0),
    #         weak_variance_mean=np.mean(landweber_weak_variance, axis=0),
    #         weak_error_mean=np.mean(landweber_weak_error, axis=0),
    #     )

    #     return self.results

    def run_simulation_conjugate_gradients(self):
        info("Running simulation.")
        if self.noise is None:
            self.noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))
        self.response = self.noise + (self.response_noiseless)[:, None]

        info("Running Monte-Carlo simulation.")
        self.results = Parallel(n_jobs=self.cores)(
            delayed(self.monte_carlo_wrapper_conjugate_gradients)(m) for m in range(self.monte_carlo_runs)
        )

        return self.results

    # def __search_learning_rate(self, search_depth):
    #     u, s, vh = svds(self.design, k=1)
    #     largest_singular_value = s[0]
    #     initial_guess_learning_rate = 2 / largest_singular_value**2
    #     self.learning_rate_candidates = np.array([initial_guess_learning_rate / (2**i) for i in range(search_depth)])
    #     info("Determine learning rate candidates based on search depth.")

    #     results = Parallel(n_jobs=self.cores)(
    #         delayed(self.search_learning_rate_wrapper)(m) for m in range(search_depth)
    #     )

    #     learning_rate_candidates_evaluation, strong_errors = zip(*results)

    #     true_false_vector = np.array(np.vstack(learning_rate_candidates_evaluation).flatten())
    #     accepted_candidates = self.learning_rate_candidates[true_false_vector]
    #     accepted_errors = np.array(strong_errors)[true_false_vector]
    #     return accepted_candidates[np.argmin(accepted_errors)]

    # def search_learning_rate_wrapper(self, m):
    #     model_landweber = Landweber(
    #         design=self.design,
    #         response=self.response[:, 0],
    #         true_signal=self.true_signal,
    #         true_noise_level=self.true_noise_level,
    #         learning_rate=self.learning_rate_candidates[m],
    #     )

    #     model_landweber.iterate(self.max_iteration)
    #     stopping_index_landweber = model_landweber.get_discrepancy_stop(
    #         self.sample_size * (self.true_noise_level**2), self.max_iteration
    #     )

    #     converges = True
    #     for k in range((self.max_iteration - 1)):
    #         if model_landweber.residuals[k] < model_landweber.residuals[(k + 1)]:
    #             converges = False

    #     if converges is False:
    #         info(f"The estimator does not converge for learning rate: {self.learning_rate_candidates[m]}", color="red")

    #     if stopping_index_landweber == None:
    #         stopping_index_landweber = self.max_iteration

    #     return converges, model_landweber.strong_empirical_risk[stopping_index_landweber]

    # def monte_carlo_wrapper(self, m):
    #     info(f"Monte-Carlo run {m + 1}/{self.monte_carlo_runs}.")
    #     model_landweber = Landweber(
    #         design=self.design,
    #         response=self.response[:, m],
    #         true_signal=self.true_signal,
    #         true_noise_level=self.true_noise_level,
    #         learning_rate=self.learning_rate,
    #     )
    #     model_conjugate_gradients = ConjugateGradients(
    #         design=self.design,
    #         response=self.response[:, m],
    #         true_signal=self.true_signal,
    #         true_noise_level=self.true_noise_level,
    #     )

    #     model_conjugate_gradients.gather_all(self.max_iteration)
    #     model_landweber.iterate(self.max_iteration)

    #     landweber_strong_bias = model_landweber.strong_bias2
    #     landweber_strong_variance = model_landweber.strong_variance
    #     landweber_strong_risk = model_landweber.strong_risk
    #     landweber_weak_bias = model_landweber.weak_bias2
    #     landweber_weak_variance = model_landweber.weak_variance
    #     landweber_weak_risk = model_landweber.weak_risk
    #     landweber_residuals = model_landweber.residuals

    #     stopping_index_landweber = model_landweber.get_discrepancy_stop(
    #         self.sample_size * (self.true_noise_level**2), self.max_iteration
    #     )
    #     balanced_oracle_weak = model_landweber.get_weak_balanced_oracle(self.max_iteration)
    #     balanced_oracle_strong = model_landweber.get_strong_balanced_oracle(self.max_iteration)

    #     landweber_strong_empirical_risk_es = model_landweber.strong_empirical_risk[stopping_index_landweber]
    #     conjugate_gradients_strong_empirical_error_es = model_conjugate_gradients.strong_empirical_errors[
    #         model_conjugate_gradients.early_stopping_index
    #     ]

    #     landweber_weak_empirical_risk_es = model_landweber.weak_empirical_risk[stopping_index_landweber]
    #     conjugate_gradients_weak_empirical_error_es = model_conjugate_gradients.weak_empirical_errors[
    #         model_conjugate_gradients.early_stopping_index
    #     ]

    #     landweber_weak_relative_efficiency = np.sqrt(
    #         np.min(model_landweber.weak_empirical_risk) / landweber_weak_empirical_risk_es
    #     )
    #     landweber_strong_relative_efficiency = np.sqrt(
    #         np.min(model_landweber.strong_empirical_risk) / landweber_strong_empirical_risk_es
    #     )

    #     conjugate_gradients_weak_relative_efficiency = np.sqrt(
    #         np.min(model_conjugate_gradients.weak_empirical_errors) / conjugate_gradients_weak_empirical_error_es
    #     )
    #     conjugate_gradients_strong_relative_efficiency = np.sqrt(
    #         np.min(model_conjugate_gradients.strong_empirical_errors) / conjugate_gradients_strong_empirical_error_es
    #     )

    #     return (
    #         np.array([landweber_strong_empirical_risk_es, conjugate_gradients_strong_empirical_error_es]),
    #         np.array([landweber_weak_empirical_risk_es, conjugate_gradients_weak_empirical_error_es]),
    #         np.array([landweber_weak_relative_efficiency, conjugate_gradients_weak_relative_efficiency]),
    #         np.array([landweber_strong_relative_efficiency, conjugate_gradients_strong_relative_efficiency]),
    #         landweber_strong_bias,
    #         landweber_strong_variance,
    #         landweber_strong_risk,
    #         landweber_weak_bias,
    #         landweber_weak_variance,
    #         landweber_weak_risk,
    #         landweber_residuals,
    #         stopping_index_landweber,
    #         balanced_oracle_weak,
    #         balanced_oracle_strong,
    #     )

    def monte_carlo_wrapper_conjugate_gradients(self, m):
        info(f"Monte-Carlo run {m + 1}/{self.monte_carlo_runs}.")

        model_conjugate_gradients = ConjugateGradients(
            design=self.design,
            response=self.response[:, m],
            true_signal=self.true_signal,
            true_noise_level=self.true_noise_level,
            computation_threshold=self.computation_threshold,
        )

        if self.critical_value is None:
            self.critical_value = self.sample_size * (self.true_noise_level**2)

        strong_empirical_oracle = model_conjugate_gradients.get_strong_empirical_oracle(
            max_iteration=self.max_iteration, interpolation=self.interpolation
        )
        weak_empirical_oracle = model_conjugate_gradients.get_weak_empirical_oracle(
            max_iteration=self.max_iteration, interpolation=self.interpolation
        )
        stopping_index = model_conjugate_gradients.get_discrepancy_stop(
            critical_value=self.critical_value, max_iteration=self.max_iteration, interpolation=self.interpolation
        )
        strong_empirical_oracle_risk = model_conjugate_gradients.get_strong_empirical_risk(strong_empirical_oracle)
        weak_empirical_oracle_risk = model_conjugate_gradients.get_weak_empirical_risk(weak_empirical_oracle)
        strong_empirical_stopping_index_risk = model_conjugate_gradients.get_strong_empirical_risk(stopping_index)
        weak_empirical_stopping_index_risk = model_conjugate_gradients.get_weak_empirical_risk(stopping_index)
        squared_residual_at_stopping_index = model_conjugate_gradients.get_residual(stopping_index)

        strong_relative_efficiency = np.sqrt(strong_empirical_oracle_risk / strong_empirical_stopping_index_risk)
        weak_relative_efficiency = np.sqrt(weak_empirical_oracle_risk / weak_empirical_stopping_index_risk)

        terminal_iteration = model_conjugate_gradients.iteration

        return (
            strong_empirical_oracle,
            weak_empirical_oracle,
            stopping_index,
            strong_empirical_oracle_risk,
            strong_empirical_stopping_index_risk,
            weak_empirical_oracle_risk,
            weak_empirical_stopping_index_risk,
            squared_residual_at_stopping_index,
            strong_relative_efficiency,
            weak_relative_efficiency,
            terminal_iteration,
        )

    # def __visualise_bias_variance_tradeoff(
    #     self,
    #     residuals_mean,
    #     strong_bias2_mean,
    #     strong_variance_mean,
    #     strong_error_mean,
    #     weak_bias2_mean,
    #     weak_variance_mean,
    #     weak_error_mean,
    # ):

    #     fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    #     axs[0].plot(range(0, self.max_iteration + 1), residuals_mean / np.max(residuals_mean))
    #     # axs[0].axvline(x=supersmooth_m, color="red", linestyle="--")
    #     axs[0].set_xlim([0, 50])
    #     axs[0].set_ylim([0, 1])
    #     axs[0].set_xlabel("Iteration")
    #     axs[0].set_ylabel("Normalised Residuals")

    #     axs[1].plot(
    #         range(0, self.max_iteration + 1),
    #         strong_error_mean / np.max(strong_error_mean),
    #         color="orange",
    #         label="Error",
    #     )
    #     axs[1].plot(
    #         range(0, self.max_iteration + 1),
    #         strong_bias2_mean / np.max(strong_bias2_mean),
    #         label="$Bias^2$",
    #         color="grey",
    #     )
    #     axs[1].plot(
    #         range(0, self.max_iteration + 1),
    #         strong_variance_mean / np.max(strong_variance_mean),
    #         label="Variance",
    #         color="blue",
    #     )
    #     # axs[1].axvline(x=supersmooth_m, color="red", linestyle="--")
    #     # axs[1].axvline(x=supersmooth_strong_oracle, color="green", linestyle="--")
    #     axs[1].set_xlim([0, 50])
    #     axs[1].set_ylim([0, 0.5])
    #     axs[1].set_xlabel("Iteration")
    #     axs[1].set_ylabel("Normalised Strong Quantities")

    #     axs[2].plot(
    #         range(0, self.max_iteration + 1), weak_error_mean / np.max(weak_error_mean), color="orange", label="Error"
    #     )
    #     axs[2].plot(
    #         range(0, self.max_iteration + 1),
    #         weak_bias2_mean / np.max(weak_bias2_mean),
    #         label="$Bias^2$",
    #         color="grey",
    #     )
    #     axs[2].plot(
    #         range(0, self.max_iteration + 1),
    #         weak_variance_mean / np.max(weak_variance_mean),
    #         label="Variance",
    #         color="blue",
    #     )
    #     # axs[2].axvline(x=supersmooth_m, color="red", linestyle="--", label=r"$\tau$")
    #     # axs[2].axvline(x=supersmooth_weak_oracle, color="green", linestyle="--", label="$t$ (oracle)")
    #     axs[2].set_xlim([0, 50])
    #     axs[2].set_ylim([0, 0.5])
    #     axs[2].set_xlabel("Iteration")
    #     axs[2].set_ylabel("Normalised Weak Quantities")
    #     axs[2].legend()

    #     plt.tight_layout()

    #     plt.show()

    # def __visualise_boxplot(self, quantity_to_visualise, quantity_name):

    #     comparison_table_quantity = pd.DataFrame(
    #         {"conjugate_gradient": quantity_to_visualise[:, 1], "landweber": quantity_to_visualise[:, 0]}
    #     )

    #     comparison_table_quantity = pd.melt(
    #         comparison_table_quantity,
    #         value_vars=["conjugate_gradient", "landweber"],
    #     )

    #     plt.figure(figsize=(14, 12))
    #     quantity_boxplot = sns.boxplot(
    #         x="variable",
    #         y="value",
    #         hue="variable",
    #         data=comparison_table_quantity,
    #         width=0.8,
    #         palette=["tab:purple", "tab:purple"],
    #     )
    #     quantity_boxplot.set_ylabel(f"{quantity_name}", fontsize=24)  # Increase fontsize
    #     quantity_boxplot.set_xlabel("Data generating processes", fontsize=24)  # Increase fontsize

    #     # Get current tick locations and labels
    #     locations = quantity_boxplot.get_xticks()
    #     labels = [item.get_text() for item in quantity_boxplot.get_xticklabels()]
    #     # Set fixed locator and labels
    #     quantity_boxplot.xaxis.set_major_locator(FixedLocator(locations))
    #     quantity_boxplot.set_xticklabels(labels, rotation=45)

    #     quantity_boxplot.tick_params(axis="both", which="major", labelsize=24)  # Increase fontsize
    #     plt.title(f"Comparison of {quantity_name}", fontsize=28)  # Increase title fontsize
    #     plt.tight_layout()
    #     plt.legend([], [], frameon=False)
    #     plt.show()


def info(message, color="green"):
    if color == "green":
        print(f"\033[92m{message}\033[0m")
    if color == "red":
        print(f"\033[31m{message}\033[0m")
