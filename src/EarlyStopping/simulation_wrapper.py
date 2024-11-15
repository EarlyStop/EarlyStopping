import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import toeplitz
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import svds
import pandas as pd

from .landweber import Landweber
from .conjugate_gradients import ConjugateGradients
from .truncated_svd import TruncatedSVD

# import matplotlib.pyplot as plt
# from matplotlib.ticker import FixedLocator
# import seaborn as sns
# import pandas as pd
# TODO: Update bias2 vs. bias/ mse vs risk, landweber empirical vs. non-empirical quantities for replication study
# bias2
# risk


class SimulationData:
    """
     `[Source] <https://github.com/ESFIEP/EarlyStopping/edit/main/src/EarlyStopping/simulation_wrapper.py>`_ A collection of static methods for the creation of simmulation data.

    **Description**

    Collection of serveral important examples of inverse problems.

    **References**

     `[Toolbox] <https://www.mathworks.com/matlabcentral/fileexchange/52-regtools>`_ The ill-posed inverse problems heat, deriv2, gravity, phillips are based on the Hansen-Matlab toolbox.

    **Methods**

    +----------------------------------------------------+----------------------------------------------------------------------------------------------+
    | diagonal_data(``sample_size``, ``type``)           | Creation of diagonal design with `smooth`, `supersmooth` and `rough` signals                 |
    +----------------------------------------------------+----------------------------------------------------------------------------------------------+
    | gravity(``sample_size``, ``a``, ``b``, ``d``)      | Discretised gravity operator with parameters controlling the ill-posedness  of the problem   |
    +----------------------------------------------------+----------------------------------------------------------------------------------------------+
    | heat(``sample_size``, ``kappa``)                   | Discretised heat semigroup with parameter controlling the ill-conditioning of the matrix     |
    +----------------------------------------------------+----------------------------------------------------------------------------------------------+
    | deriv2(``sample_size``, ``example``)               | Discretisation of a Fredholm integral equation whose kernel K is the Green's function        |
    +----------------------------------------------------+----------------------------------------------------------------------------------------------+
    | phillips(``sample_size``)                          | Create data based on the famous Phillips example                                             |
    +----------------------------------------------------+----------------------------------------------------------------------------------------------+
    """

    @staticmethod
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

    @staticmethod
    def gravity(sample_size, a=0, b=1, d=0.25):
        """
        Discretisation of a 1-D model problem in gravity surveying, in which
        a mass distribution :math:`f(t)` is located at depth d, while the vertical
        component of the gravity field :math:`g(s)` is measured at the surface.

        The resulting problem is a first-kind Fredholm integral equation  with kernel

        .. math::
            K(s,t) = d(d^2 + (s-t)^2)^{-3/2},

        with the right-hand side given by

        .. math::
            f(t) = \\sin(\\pi t) + 0.5\\sin(2\\pi t).

        The problem is discretized by means of the midpoint quadrature rule.

        **Parameters**

        *sample_size*: ``int``. Specifies the size of the design matrix to be generated.

        *a*: ``int``. The lower bound of the integration interval.

        *b*: ``int``. The upper bound of the integration interval.

        *d*: ``int``. The depth at which the magnetic deposit is located.

        **Returns**

        *design*: ``ndarray``. The design matrix.

        *response_noiseless*: ``ndarray``. The noiseless response. The response is produced by applying the design to the signal.

        *true_signal*: ``ndarray``. The true signal.
        """

        # Parameter controlling the ill-posedness: the larger, the more ill-posed, default in regtools: d = 0.25

        t = (np.arange(1, sample_size + 1) - 0.5) / sample_size
        s = ((np.arange(1, sample_size + 1) - 0.5) * (b - a)) / sample_size
        T, S = np.meshgrid(t, s)

        design = (1 / sample_size) * d * (d**2 * np.ones((sample_size, sample_size)) + (S - T) ** 2) ** (-(3 / 2))
        true_signal = np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t)

        response_noiseless = design @ true_signal
        return design, response_noiseless, true_signal

    @staticmethod
    def heat(sample_size, kappa=1):
        """
        A first kind Volterra integral equation with :math:`[0,1]` as integration interval. The kernel is :math:`K(s,t) = k(s-t)` with the heat kernel

        .. math::
            k(t) = \\frac{t^{-1/2}}{2\\kappa \\sqrt{\\pi}}\\exp\\left(-\\frac{1}{4 \\kappa^2 t}\\right) .

        **Parameters**

        *sample_size*: ``int``. Specifies the size of the design matrix to be generated.

        *kappa*: ``int``. Here, kappa controls the ill-conditioning of the matrix:

        - :math:`\\kappa = 5` gives a well-conditioned problem.

        - :math:`\\kappa = 1` gives an ill-conditioned problem.

        **Returns**

        *design*: ``ndarray``. The design matrix.

        *response_noiseless*: ``ndarray``. The noiseless response. An exact soltuion is constructed, and the response is produced by applying the design to the signal.

        *true_signal*: ``ndarray``. The true signal.
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

    @staticmethod
    def deriv2(sample_size, example=1):
        """
        Mildly ill-posed inverse problem based on the discretization of a
        Fredholm integral equation of the first kind whose kernel K is the
        Green's function for the second derivative:

        .. math::
             K(s,t) = \\begin{cases}s(t-1)  ,  &s <  t, \\\\ t(s-1)  ,  &s \\geq t.\\end{cases}

        The right hand side :math:`g` and the solution :math:`f` can be chosen as follows:

        1. :math:`g(s) = (s^3 - s)/6` and :math:`f(t) = t`.

        2. :math:`g(s) = \\exp(s) + (1-e)s - 1` and :math:`f(t) = \\exp(t)`.

        3. :math:`g(s) = \\begin{cases}(4s^3 - 3s)/24,  &s <  0.5,\\\\(-4s^3 + 12s^2 - 9s + 1)/24,  &s \\geq 0.5.\\end{cases}` and :math:`f(t) = \\begin{cases}t,  &t <  0.5,\\\\1-t,  &t \\geq 0.5.\\end{cases}`.


        **Parameters**

        *sample_size*: ``int``. Specifies the size of the design matrix to be generated.

        *example*: ``int``. The example to be used. Must be 1, 2 or 3.

        **Returns**

        *design*: ``ndarray``. The design matrix.

        *response_noiseless*: ``ndarray``. The noiseless response. (Based on discretisation).

        *true_signal*: ``ndarray``. The true signal.
        """
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

    @staticmethod
    def phillips(sample_size):
        """
        Discretization of the *famous* first-kind Fredholm integral equation deviced by D. L. Phillips.

        Define the function

        .. math::
             \\phi(x) = \\begin{cases}1 + \\cos(x\\pi/3), & |x| < 3,\\\\0, & |x| \\geq 3.\\end{cases}

        The kernel K, the solution f, and the right-hand side g are given by:

        - :math:`K(s,t) = \\phi(s-t)`

        - :math:`f(t) = \\phi(t)`

        - :math:`g(s) = (6-|s|)(1+0.5 \\cos(s\\pi/3)) + 9/(2\\pi)\\sin(|s|\\pi/3)`

        Both integration intervals are [-6,6].

        **Parameters**

        *sample_size*: ``int``. Specifies the size of the design matrix to be generated. Must be a multiple of 4.

        **Returns**

        *design*: ``ndarray``. The design matrix.

        *response_noiseless*: ``ndarray``. The noiseless response. (obtained from discretisation).

        *true_signal*: ``ndarray``. The true signal.
        """
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
    """
    `[Source] <https://github.com/ESFIEP/EarlyStopping/edit/main/src/EarlyStopping/simulation_wrapper.py>`_


    A class for managing and validating the parameters for simulation in the `SimulationWrapper` class. This class
    ensures the validity of input parameters required for performing simulation tasks and manages attributes
    such as design matrix, noise level, iteration limits, and parallel processing settings.

    **Parameters**

    *design*: ``ndarray``. The design matrix of the simulation, representing the model matrix.

    *true_signal*: ``ndarray``. The true signal or target vector used for generating simulated response values.

    *true_noise_level*: ``float``. The standard deviation of the normally distributed noise applied to the response.

    *max_iteration*: ``int``. Specifies the maximum number of iterations allowed for each simulation.

    *monte_carlo_runs*: ``int``. Defines the number of Monte-Carlo runs to perform in the simulation.

    *noise*: ``ndarray, optional``. Specifies an initial noise matrix; defaults to `None`, in which case noise will
    be generated as needed.

    *response_noiseless*: ``ndarray, optional``. Represents the noiseless response vector, if available. Default is `None`.

    *critical_value*: ``float, optional``. A critical threshold value for the simulation, used in criteria like
    early stopping. Default is `None`.

    *interpolation*: ``bool, default=False``. Specifies whether to use interpolation techniques within the simulation.

    *computation_threshold*: ``float, default=10 ** (-8)``. A small threshold to control numerical computations
    in the simulation procedures.

    *cores*: ``int, default=5``. Specifies the number of processor cores to use for parallel execution.
    """

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
        self.__validate()

    def __validate(self):
        if not isinstance(self.true_signal, np.ndarray):
            raise ValueError("true_signal must be a numpy array.")
        if not (0 <= self.true_noise_level):
            raise ValueError("true_noise_level must be greater than or equal to 0.")
        if not isinstance(self.max_iteration, int) or self.max_iteration < 0:
            raise ValueError("max_iteration must be a nonnegative integer.")
        # add more cases to validate (e.g. for noise)
        # TODO: Catch the case for truncated SVD that you have more iterations as max iterations in your simmulation than dimensions

class SimulationWrapper:
    """
    `[Source] <https://github.com/ESFIEP/EarlyStopping/edit/main/src/EarlyStopping/simulation_wrapper.py>`_ A wrapper class for collecting montecarlo simulation data.

    **Parameters**

    *design*: ``ndarray``. The design matrix of the simulation, representing the model matrix.

    *true_signal*: ``ndarray``. The true signal or target vector used for generating simulated response values.

    *true_noise_level*: ``float``. The standard deviation of the normally distributed noise applied to the response.

    *max_iteration*: ``int``. Specifies the maximum number of iterations to use within the simulation.

    *monte_carlo_runs*: ``int``. Defines the number of Monte-Carlo runs to perform in the simulation.

    *noise*: ``ndarray, optional``. Specifies an initial noise matrix; defaults to `None`, in which case noise will
    be generated as needed.

    *response_noiseless*: ``ndarray, optional``. Represents the noiseless response vector, if available. Default is `None`.

    *critical_value*: ``float, optional``. Critical value to be used for early stopping. Default is `None` leading to theoretical values being used.

    *interpolation*: ``bool, default=False``. Specifies whether to use interpolation within the simulation.

    *computation_threshold*: ``float, default=10 ** (-8)``. A small threshold to control numerical computations
    in the simulation procedures.

    **Methods**

    +-----------------------------------------------------------------+---------------------------------------------------------+
    |  run_simulation_truncated_svd(``diagonal``, ``data_set_name``)  | Run montecarlo simulation for truncated SVD.            |
    +-----------------------------------------------------------------+---------------------------------------------------------+
    |  run_simulation_landweber(``learning_rate``,``data_set_name``)  | Run montecarlo simulation for the Landweber iteration.  |
    +-----------------------------------------------------------------+---------------------------------------------------------+
    |  run_simulation_conjugate_gradients()                           | Run montecarlo simulation for conjugate gradients.      |
    +-----------------------------------------------------------------+---------------------------------------------------------+
    """

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

    def run_simulation_landweber(self, learning_rate=None, data_set_name=None):
        """
        Runs a simulation for an inverse problem using the Landweber iteration method.
        The function generates a noisy response based on the specified noise level and and performs a Monte-Carlo
        simulation to collect various metrics related to the estimator's performance.

        **Parameters**

        *learning_rate*: ``float`` or ``str``, optional. Specifies the learning rate for the Landweber iteration.
        If set to `"auto"`, the method will search for an optimal learning rate. Default is 1.

        *data_set_name*: ``str``, optional. If specified, the results are saved to a CSV file with this name.

        **Returns**

        *results_df*: ``pd.DataFrame``. DataFrame containing the results of the Monte-Carlo simulation.
        """
        info("Running simulation.")
        if self.noise is None:
            self.noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))
        self.response = self.noise + (self.response_noiseless)[:, None]

        if learning_rate == "auto":
            info("Searching for viable learning rates.")
            self.learning_rate = self.search_learning_rate(search_depth=10)
        else:
            self.learning_rate = 1

        info("Running Monte Carlo simulation.")
        results = Parallel(n_jobs=self.cores)(
            delayed(self.monte_carlo_wrapper_landweber)(m) for m in range(self.monte_carlo_runs)
        )

        column_names = [
            "landweber_strong_empirical_risk_es",
            "landweber_weak_empirical_risk_es",
            "landweber_weak_relative_efficiency",
            "landweber_strong_relative_efficiency",
            "landweber_strong_bias2",
            "landweber_strong_variance",
            "landweber_strong_risk",
            "landweber_weak_bias2",
            "landweber_weak_variance",
            "landweber_weak_risk",
            "landweber_residuals",
            "stopping_index_landweber",
            "balanced_oracle_weak",
            "balanced_oracle_strong",
        ]

        results_df = pd.DataFrame(results, columns=column_names)

        if data_set_name:
            results_df.to_csv(f"{data_set_name}.csv", index=False)

        return results_df

    def run_simulation_truncated_svd(self, diagonal=False, data_set_name=None):
        """
        Runs a simulation for an inverse problem using truncated Singular Value Decomposition (SVD).
        The function generates a noisy response based on the specified noise level and performs a Monte-Carlo
        simulation to collect various metrics related to the estimator's performance.

        **Parameters**

        *diagonal*: ``bool``, optional. Specifies whether to treat the matrix as diagonal. Default is False.

        *data_set_name*: ``str``, optional. If specified, the results are saved to a CSV file with this name.

        **Returns**

        *results_df*: ``pd.DataFrame``. DataFrame containing the results of the Monte-Carlo simulation.

        The resulting DataFrame can be saved to a CSV file if `data_set_name` is provided.
        """
        self.diagonal = diagonal
        info("Running simulation.")
        if self.noise is None:
            self.noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))

        self.response = self.noise + (self.response_noiseless)[:, None]
        info("Running Monte-Carlo simulation.")

        results = Parallel(n_jobs=self.cores)(
            delayed(self.monte_carlo_wrapper_truncated_svd)(m) for m in range(self.monte_carlo_runs)
        )


        # TODO-BS-2024-11-02: Add AIC stop, classical oracles, etc. as column
        column_names = [
            "strong_bias2",
            "strong_variance",
            "strong_mse",
            "weak_bias2",
            "weak_variance",
            "weak_mse",
            "residuals",
            "discrepancy_stop",
            "weak_balanced_oracle",
            "strong_balanced_oracle",
            "weak_classical_oracle",
            "strong_classical_oracle",
            "weak_relative_efficiency",
            "strong_relative_efficiency",
        ]

        results_df = pd.DataFrame(results, columns=column_names)

        if data_set_name:
            results_df.to_csv(f"{data_set_name}.csv", index=False)

        return results_df

    def monte_carlo_wrapper_truncated_svd(self, m):
        info(f"Monte-Carlo run {m + 1}/{self.monte_carlo_runs}.")
        model_truncated_svd = TruncatedSVD(
            design=self.design,
            response=self.response[:, m],
            true_signal=self.true_signal,
            true_noise_level=self.true_noise_level,
            diagonal=self.diagonal,
        )

        model_truncated_svd.iterate(self.max_iteration)

        strong_bias2 = model_truncated_svd.strong_bias2
        strong_variance = model_truncated_svd.strong_variance
        strong_mse = model_truncated_svd.strong_mse
        weak_bias2 = model_truncated_svd.weak_bias2
        weak_variance = model_truncated_svd.weak_variance
        weak_mse = model_truncated_svd.weak_mse
        residuals = model_truncated_svd.residuals

        discrepancy_stop = model_truncated_svd.get_discrepancy_stop(
            self.sample_size * (self.true_noise_level**2), self.max_iteration
        )
        weak_balanced_oracle = model_truncated_svd.get_weak_balanced_oracle(self.max_iteration)
        strong_balanced_oracle = model_truncated_svd.get_strong_balanced_oracle(self.max_iteration)

        weak_classical_oracle = np.argmin(weak_mse)
        strong_classical_oracle = np.argmin(strong_mse)


        weak_error_vector_at_stopping_time = model_truncated_svd.design @ (model_truncated_svd.get_estimate(discrepancy_stop) - model_truncated_svd.true_signal)
        weak_error_at_stopping_time        = np.sum(weak_error_vector_at_stopping_time**2)
        weak_relative_efficiency           = np.sqrt(np.min(weak_mse) / weak_error_at_stopping_time)

        strong_error_at_stopping_time = np.sum((model_truncated_svd.get_estimate(discrepancy_stop) - model_truncated_svd.true_signal)**2)
        strong_relative_efficiency    = np.sqrt(np.min(strong_mse) / strong_error_at_stopping_time)

        return (
            strong_bias2,
            strong_variance,
            strong_mse,
            weak_bias2,
            weak_variance,
            weak_mse,
            residuals,
            discrepancy_stop,
            weak_balanced_oracle,
            strong_balanced_oracle,
            weak_classical_oracle,
            strong_classical_oracle,
            weak_relative_efficiency,
            strong_relative_efficiency,
        )

    def run_simulation_conjugate_gradients(self):
        """
        Runs a simulation for an inverse problem using the Conjugate Gradients method.
        This function generates a noisy response based on the specified noise level and performs a
        Monte-Carlo simulation to collect various metrics related to the estimator's performance.

        **Parameters**

        None

        **Returns**

        *results*: ``list``. A list of results from the Monte-Carlo simulation.
        """
        info("Running simulation.")
        if self.noise is None:
            self.noise = np.random.normal(0, self.true_noise_level, (self.sample_size, self.monte_carlo_runs))
        self.response = self.noise + (self.response_noiseless)[:, None]

        info("Running Monte-Carlo simulation.")
        self.results = Parallel(n_jobs=self.cores)(
            delayed(self.monte_carlo_wrapper_conjugate_gradients)(m) for m in range(self.monte_carlo_runs)
        )

        return self.results

    def search_learning_rate(self, search_depth):
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

        model_landweber.iterate(self.max_iteration)
        stopping_index_landweber = model_landweber.get_discrepancy_stop(
            self.sample_size * (self.true_noise_level**2), self.max_iteration
        )

        converges = True
        for k in range((self.max_iteration - 1)):
            if model_landweber.residuals[k] < model_landweber.residuals[(k + 1)]:
                converges = False

        if converges is False:
            info(f"The estimator does not converge for learning rate: {self.learning_rate_candidates[m]}", color="red")

        if stopping_index_landweber == None:
            stopping_index_landweber = self.max_iteration

        return converges, model_landweber.strong_empirical_risk[stopping_index_landweber]

    def monte_carlo_wrapper_landweber(self, m):
        info(f"Monte-Carlo run {m + 1}/{self.monte_carlo_runs}.")
        model_landweber = Landweber(
            design=self.design,
            response=self.response[:, m],
            true_signal=self.true_signal,
            true_noise_level=self.true_noise_level,
            learning_rate=self.learning_rate,
        )

        model_landweber.iterate(self.max_iteration)

        landweber_strong_bias2 = model_landweber.strong_bias2
        landweber_strong_variance = model_landweber.strong_variance
        landweber_strong_risk = model_landweber.strong_risk
        landweber_weak_bias2 = model_landweber.weak_bias2
        landweber_weak_variance = model_landweber.weak_variance
        landweber_weak_risk = model_landweber.weak_risk
        landweber_residuals = model_landweber.residuals

        stopping_index_landweber = model_landweber.get_discrepancy_stop(
            self.sample_size * (self.true_noise_level**2), self.max_iteration
        )
        balanced_oracle_weak = model_landweber.get_weak_balanced_oracle(self.max_iteration)
        balanced_oracle_strong = model_landweber.get_strong_balanced_oracle(self.max_iteration)

        landweber_strong_empirical_risk_es = model_landweber.strong_empirical_risk[stopping_index_landweber]
        landweber_weak_empirical_risk_es = model_landweber.weak_empirical_risk[stopping_index_landweber]
        landweber_weak_relative_efficiency = np.sqrt(
            np.min(model_landweber.weak_empirical_risk) / landweber_weak_empirical_risk_es
        )
        landweber_strong_relative_efficiency = np.sqrt(
            np.min(model_landweber.strong_empirical_risk) / landweber_strong_empirical_risk_es
        )

        return (
            landweber_strong_empirical_risk_es,
            landweber_weak_empirical_risk_es,
            landweber_weak_relative_efficiency,
            landweber_strong_relative_efficiency,
            landweber_strong_bias2,
            landweber_strong_variance,
            landweber_strong_risk,
            landweber_weak_bias2,
            landweber_weak_variance,
            landweber_weak_risk,
            landweber_residuals,
            stopping_index_landweber,
            balanced_oracle_weak,
            balanced_oracle_strong,
        )

    def monte_carlo_wrapper_conjugate_gradients(self, m):
        info(f"Monte-Carlo run {m + 1}/{self.monte_carlo_runs}.")

        model_conjugate_gradients = ConjugateGradients(
            design=self.design,
            response=self.response[:, m],
            initial_value=None,
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


def info(message, color="green"):
    if color == "green":
        print(f"\033[92m{message}\033[0m")
    if color == "red":
        print(f"\033[31m{message}\033[0m")
