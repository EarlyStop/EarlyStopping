import numpy as np
from joblib import Parallel, delayed
from .landweber import Landweber
from .conjugate_gradients import ConjugateGradients
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.sparse.linalg import svds


class SimulationParameters:
    def __init__(self, design, true_signal, true_noise_level, max_iterations):
        self.design = design
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.max_iterations = max_iterations
        self.validate()

    def validate(self):
        if not isinstance(self.true_signal, np.ndarray):
            raise ValueError("true_signal must be a numpy array")
        if not (0 < self.true_noise_level):
            raise ValueError("true_noise_level must be between 0 and 1")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")


class SimulationWrapper:
    def __init__(
        self, design, true_signal=None, true_noise_level=None, monte_carlo_runs=5, max_iterations=1000, cores=3
    ):
        self.design = design
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.monte_carlo_runs = monte_carlo_runs
        self.max_iterations = max_iterations
        self.cores = cores

        self.sample_size = design.shape[0]

    def run_simulation(self):
        info("Running simulation.")
        noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))
        self.response = noise + (self.design @ self.true_signal)[:, None]

        info("Searching for viable learning rates.")
        self.learning_rate = self.__search_learning_rate(search_depth=10)

        info("Running Monte Carlo simulation.")
        self.results = Parallel(n_jobs=self.cores)(
            delayed(self.monte_carlo_wrapper)(m) for m in range(self.monte_carlo_runs)
        )

        strong_errors, weak_errors = zip(*self.results)

        self.__visualise(np.vstack(strong_errors), "Strong Empirical Error at $\\tau$")
        self.__visualise(np.vstack(weak_errors), "Weak Empirical Error at $\\tau$")
        return self.results

    def __search_learning_rate(self, search_depth):
        u, s, vh = svds(self.design, k=1)
        largest_singular_value = s[0]
        initial_guess_learning_rate = 2 / largest_singular_value**2
        self.learning_rate_candidates = np.array([initial_guess_learning_rate / (2**i) for i in range(search_depth)])
        info("Determine learning rate candidates based on search depth.")

        results = Parallel(n_jobs=self.cores)(
            delayed(self.search_learning_rate_wrapper)(m) for m in range(search_depth)
        )

        learning_rate_candidates_evaluation, strong_errors = zip(*results)

        true_false_vector = np.array(np.vstack(learning_rate_candidates_evaluation).flatten())
        accepted_candidates = self.learning_rate_candidates[true_false_vector]
        accepted_errors = np.array(strong_errors)[true_false_vector]
        return accepted_candidates[np.argmin(accepted_errors)]

    def search_learning_rate_wrapper(self, m):
        model_landweber = Landweber(
            design=self.design,
            response=self.response[:, 0],
            true_signal=self.true_signal,
            true_noise_level=self.true_noise_level,
            learning_rate=self.learning_rate_candidates[m],
        )

        model_landweber.iterate(self.max_iterations)

        converges = True
        for k in range((self.max_iterations - 1)):
            if model_landweber.residuals[k] < model_landweber.residuals[(k + 1)]:
                converges = False

        if converges is False:
            info(f"The estimator does not converge for learning rate: {self.learning_rate_candidates[m]}", color="red")

        return converges, model_landweber.strong_empirical_error[model_landweber.get_early_stopping_index()]

    def monte_carlo_wrapper(self, m):
        info(f"Monte Carlo run {m + 1}/{self.monte_carlo_runs}.")
        model_landweber = Landweber(
            design=self.design,
            response=self.response[:, m],
            true_signal=self.true_signal,
            true_noise_level=self.true_noise_level,
            learning_rate=self.learning_rate,
        )
        model_conjugate_gradients = ConjugateGradients(
            design=self.design,
            response=self.response[:, m],
            true_signal=self.true_signal,
            true_noise_level=self.true_noise_level,
        )

        model_conjugate_gradients.gather_all(self.max_iterations)
        model_landweber.iterate(self.max_iterations)

        landweber_strong_empirical_error = model_landweber.strong_empirical_error[
            model_landweber.get_early_stopping_index()
        ]
        conjugate_gradients_strong_empirical_error = model_conjugate_gradients.strong_empirical_errors[
            model_conjugate_gradients.early_stopping_index
        ]

        landweber_weak_empirical_error = model_landweber.weak_empirical_error[
            model_landweber.get_early_stopping_index()
        ]
        conjugate_gradients_weak_empirical_error = model_conjugate_gradients.weak_empirical_errors[
            model_conjugate_gradients.early_stopping_index
        ]

        return (
            np.array([landweber_strong_empirical_error, conjugate_gradients_strong_empirical_error]),
            np.array([landweber_weak_empirical_error, conjugate_gradients_weak_empirical_error]),
        )

    def __visualise(self, quantity_to_visualise, quantity_name):

        comparison_table_quantity = pd.DataFrame(
            {"conjugate_gradient": quantity_to_visualise[:, 1], "landweber": quantity_to_visualise[:, 0]}
        )

        comparison_table_quantity = pd.melt(
            comparison_table_quantity,
            value_vars=["conjugate_gradient", "landweber"],
        )

        plt.figure(figsize=(14, 10))
        quantity_boxplot = sns.boxplot(
            x="variable",
            y="value",
            data=comparison_table_quantity,
            width=0.8,
            palette=["tab:purple", "tab:purple"],
        )
        quantity_boxplot.set_ylabel(f"{quantity_name}", fontsize=24)  # Increase fontsize
        quantity_boxplot.set_xlabel("Data generating processes", fontsize=24)  # Increase fontsize
        quantity_boxplot.set_xticklabels(quantity_boxplot.get_xticklabels(), rotation=45)

        quantity_boxplot.tick_params(axis="both", which="major", labelsize=24)  # Increase fontsize
        plt.title(f"Comparison of {quantity_name}", fontsize=28)  # Increase title fontsize
        plt.tight_layout()
        plt.show()


def info(message, color="green"):
    if color == "green":
        print(f"\033[92m{message}\033[0m")
    if color == "red":
        print(f"\033[31m{message}\033[0m")
