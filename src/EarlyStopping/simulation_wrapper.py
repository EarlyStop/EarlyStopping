import numpy as np
from joblib import Parallel, delayed
from .landweber import Landweber
from .conjugate_gradients import ConjugateGradients
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SimulationWrapper:
    def __init__(self,
                 design,
                 true_signal=None,
                 true_noise_level=None,
                 monte_carlo_runs=5,
                 max_iterations=1000,
                 cores=6):
        self.design = design
        self.true_signal = true_signal
        self.true_noise_level = true_noise_level
        self.monte_carlo_runs = monte_carlo_runs
        self.max_iterations = max_iterations
        self.cores = cores

        self.sample_size = design.shape[0]

    def monte_carlo_wrapper(self, m, response):

        model_landweber = Landweber(
            design=self.design, response=response[:, m], true_signal=self.true_signal, true_noise_level=self.true_noise_level)
        model_conjugate_gradients = ConjugateGradients(
            design=self.design, response=response[:, m], true_signal=self.true_signal, true_noise_level=self.true_noise_level)

        model_conjugate_gradients.gather_all(self.max_iterations)
        model_landweber.iterate(self.max_iterations)

        landweber_strong_empirical_error = model_landweber.strong_empirical_error[model_landweber.get_early_stopping_index()]
        conjugate_gradients_strong_empirical_error = model_conjugate_gradients.strong_empirical_errors[model_conjugate_gradients.early_stopping_index]

        return np.array([landweber_strong_empirical_error, conjugate_gradients_strong_empirical_error])


    def run_simulation(self):

        noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))
        response = noise + (self.design @ self.true_signal)[:, None]

        self.results = np.vstack(Parallel(n_jobs=self.cores)(delayed(self.monte_carlo_wrapper)(m, response) for m in range(self.monte_carlo_runs)))

        self.__visualise()

        return self.results


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

