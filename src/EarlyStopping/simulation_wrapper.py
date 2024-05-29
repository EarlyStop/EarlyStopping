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
    def __init__(self,
                 design,
                 true_signal=None,
                 true_noise_level=None,
                 monte_carlo_runs=5,
                 max_iterations=1000,
                 cores=3):
        self.design = design
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.monte_carlo_runs = monte_carlo_runs
        self.max_iterations = max_iterations
        self.cores = cores

        self.sample_size = design.shape[0]


    def run_simulation(self):
        noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))
        self.response = noise + (self.design @ self.true_signal)[:, None]
        self.learning_rate = self.__search_learning_rate(search_depth=3)
        self.results = np.vstack(Parallel(n_jobs=self.cores)(delayed(self.monte_carlo_wrapper)(m) for m in range(self.monte_carlo_runs)))
        self.__visualise()
        return self.results
    
    def search_learning_rate_wrapper(self, m):
        model_landweber = Landweber(
            design=self.design, response=self.response[:, 0], true_signal=self.true_signal, true_noise_level=self.true_noise_level, learning_rate=self.learning_rate_candidates[m])
        
        model_landweber.iterate(self.max_iterations)

        converges = True
        for m in range((self.max_iterations - 1)):
            if model_landweber.residuals[m] < model_landweber.residuals[(m + 1)]:
                converges = False

        return converges

    def __search_learning_rate(self, search_depth):
        u, s, vh = svds(self.design, k=1)
        largest_singular_value = s[0]
        initial_guess_learning_rate = 2 / largest_singular_value**2
        self.learning_rate_candidates = np.array([initial_guess_learning_rate /(2**i) for i in range(search_depth)])
        print(self.learning_rate_candidates)
        learning_rate_candidates_evaluation = np.vstack(Parallel(n_jobs=self.cores)(delayed(self.search_learning_rate_wrapper)(m) for m in range(search_depth)))
        print(np.array(learning_rate_candidates_evaluation.flatten()))
        accepted_candidates = self.learning_rate_candidates[np.array(learning_rate_candidates_evaluation.flatten())]
        return np.max(accepted_candidates)


    def monte_carlo_wrapper(self, m):
        model_landweber = Landweber(
            design=self.design, response=self.response[:, m], true_signal=self.true_signal, true_noise_level=self.true_noise_level, learning_rate=self.learning_rate)
        model_conjugate_gradients = ConjugateGradients(
            design=self.design, response=self.response[:, m], true_signal=self.true_signal, true_noise_level=self.true_noise_level)

        model_conjugate_gradients.gather_all(self.max_iterations)
        model_landweber.iterate(self.max_iterations)

        landweber_strong_empirical_error = model_landweber.strong_empirical_error[model_landweber.get_early_stopping_index()]
        conjugate_gradients_strong_empirical_error = model_conjugate_gradients.strong_empirical_errors[model_conjugate_gradients.early_stopping_index]

        return np.array([landweber_strong_empirical_error, conjugate_gradients_strong_empirical_error])

    def __visualise(self):
        strong_empirical_errors_Monte_Carlo = pd.DataFrame(
            {
                "conjugate_gradient": self.results[:, 1],
                "landweber": self.results[:, 0]
            }
        )

        strong_empirical_errors_Monte_Carlo = pd.melt(
            strong_empirical_errors_Monte_Carlo,
            value_vars=["conjugate_gradient", "landweber"],
        )

        plt.figure(figsize=(14, 10))
        strong_empirical_errors_boxplot = sns.boxplot(
            x="variable",
            y="value",
            data=strong_empirical_errors_Monte_Carlo,
            width=0.8,
            palette=["tab:purple", "tab:purple"],
        )
        strong_empirical_errors_boxplot.set_ylabel("Strong Empirical Error at $\\tau$",
                                                   fontsize=24)  # Increase fontsize
        strong_empirical_errors_boxplot.set_xlabel("Data generating processes", fontsize=24)  # Increase fontsize
        strong_empirical_errors_boxplot.set_xticklabels(strong_empirical_errors_boxplot.get_xticklabels(), rotation=45)

        strong_empirical_errors_boxplot.tick_params(axis="both", which="major", labelsize=24)  # Increase fontsize
        plt.title("Comparison of Strong Empirical Errors", fontsize=28)  # Increase title fontsize
        plt.tight_layout()
        plt.show()

