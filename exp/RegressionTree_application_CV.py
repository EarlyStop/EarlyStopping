"""
CV companion for ``RegressionTree_application.py``.

For each dataset, this script runs 5-fold cross-validation on the training set
over a grid of relative discrepancy thresholds c (where
kappa = c * Var_hat(Y_train)) and reports the c* that minimises validation MSE.
The c* values produced here are the ones to plug back into ``DATASET_CONFIG``
in the main application script.

Sample sizes / max_depth are imported from ``DATASET_CONFIG`` so the CV runs
at the same n / d / depth that the main script uses.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import EarlyStopping as es  # noqa: E402

from RegressionTree_application import (  # noqa: E402
    DATASET_CONFIG,
    LOADERS,
    maybe_subsample,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# CV procedure
# ---------------------------------------------------------------------------

def estimate_1nn_sigma2(X: np.ndarray, y: np.ndarray) -> float:
    """1-NN noise variance estimator (Devroye et al., 2018). Reference only."""
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    n = len(y)
    return float(np.dot(y, y) / n - np.dot(y, y[idx[:, 1]]) / n)


def cv_tune_kappa(
    X: np.ndarray,
    y: np.ndarray,
    kappa_grid: np.ndarray,
    max_depth: int,
    n_folds: int = 5,
    min_samples_split: int = 1,
    seed: int = 0,
):
    """5-fold CV over kappa.

    The tree is grown ONCE per fold up to ``max_depth``. Every kappa in the grid
    is then evaluated on that single tree by querying ``get_discrepancy_stop``
    and predicting at the corresponding depth -- so grid size is essentially free.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_mse = np.zeros((n_folds, len(kappa_grid)))
    cv_stop = np.zeros((n_folds, len(kappa_grid)))

    for f, (tr, va) in enumerate(kf.split(X)):
        tree = es.RegressionTree(
            design=X[tr], response=y[tr], min_samples_split=min_samples_split
        )
        tree.iterate(max_depth=max_depth)
        max_grown = len(tree.residuals) - 1

        for i, kappa in enumerate(kappa_grid):
            k = tree.get_discrepancy_stop(critical_value=kappa)
            if k is None:
                k = max_grown
            k = int(max(0, min(k, max_grown)))
            pred = tree.predict(X[va], depth=k)
            cv_mse[f, i] = np.mean((y[va] - pred) ** 2)
            cv_stop[f, i] = k

    mean_mse = cv_mse.mean(axis=0)
    best_idx = int(np.argmin(mean_mse))
    return float(kappa_grid[best_idx]), mean_mse, cv_stop.mean(axis=0)


# ---------------------------------------------------------------------------
# Per-dataset CV grids (centred around previously-found c* values)
# ---------------------------------------------------------------------------

C_GRIDS = {
    "California Housing":   (0.10, 0.20, 0.30, 0.40, 0.50),
    "Diabetes":             (0.30, 0.40, 0.50, 0.60, 0.70, 0.85),
    "Wine Quality (white)": (0.50, 0.55, 0.60, 0.65),
    "Abalone":              (0.30, 0.40, 0.50, 0.60, 0.70),
    "Superconductivity":    (0.005, 0.010, 0.020, 0.040, 0.060, 0.080, 0.120),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_cv(name: str, X: np.ndarray, y: np.ndarray, c_grid, max_depth: int,
           max_n: int | None, seed: int = 21):
    X, y = maybe_subsample(X, y, max_n=max_n, seed=0)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
    var_tr = float(np.var(y_tr))
    kappa_grid = np.array([c * var_tr for c in c_grid])

    t0 = time.perf_counter()
    kappa_star, mean_mse, mean_stop = cv_tune_kappa(
        X_tr, y_tr, kappa_grid, max_depth=max_depth, n_folds=5, seed=seed
    )
    t = time.perf_counter() - t0

    c_star = kappa_star / var_tr
    sigma2_1nn = estimate_1nn_sigma2(X_tr, y_tr)
    best_idx = int(np.argmin(mean_mse))

    print(f"[{name}]")
    print(f"  n_train={X_tr.shape[0]}  d={X_tr.shape[1]}  Var(Y_train)={var_tr:.4f}")
    print(f"  c grid             : {list(c_grid)}")
    print(f"  CV MSE / Var(Y_tr) : {[f'{m/var_tr:.3f}' for m in mean_mse]}")
    print(f"  best c* = {c_star:.3f}  (kappa* = {kappa_star:.4f}, "
          f"k_stop avg = {mean_stop[best_idx]:.1f})")
    print(f"  1-NN sigma^2 (train) : {sigma2_1nn:.4f}  "
          f"(= {sigma2_1nn / var_tr:.3f} * Var(Y_tr))")
    print(f"  CV wall-clock        : {t:.1f}s\n")

    return {
        "name":       name,
        "n_train":    X_tr.shape[0],
        "d":          X_tr.shape[1],
        "c_star":     c_star,
        "kappa_star": kappa_star,
        "var_tr":     var_tr,
        "mean_mse":   mean_mse,
        "kappa_grid": kappa_grid,
        "mean_stop":  mean_stop,
        "sigma2_1nn": sigma2_1nn,
        "t_cv":       t,
    }


def plot_cv_curves(out_path: str, results: list[dict]):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        c_grid_norm = r["kappa_grid"] / r["var_tr"]
        ax.plot(c_grid_norm, r["mean_mse"] / r["var_tr"], marker="o")
        ax.axvline(r["c_star"], color="C3", ls="--", lw=1, label=f"$c^*$={r['c_star']:.2f}")
        ax.set_xlabel(r"$c$  with  $\kappa = c \cdot \widehat{\mathrm{Var}}(Y_{\mathrm{train}})$")
        ax.set_ylabel(r"5-fold CV MSE / $\widehat{\mathrm{Var}}(Y_{\mathrm{train}})$")
        ax.set_title(r["name"])
        ax.grid(True, alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved CV-curve plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(21)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    results = []
    for name, cfg in DATASET_CONFIG.items():
        X, y = LOADERS[name]()
        r = run_cv(
            name, X, y,
            c_grid=C_GRIDS[name],
            max_depth=cfg["max_depth"],
            max_n=cfg["max_n"],
        )
        results.append(r)

    plot_cv_curves(os.path.join(out_dir, "track_cv_curves.png"), results)

    print("=" * 78)
    print("Suggested DATASET_CONFIG c_star values (from CV at current sample sizes)")
    print("=" * 78)
    for r in results:
        print(f"  {r['name']:22s}  c_star = {r['c_star']:.3f}    "
              f"(n_train={r['n_train']}, d={r['d']}, kappa*={r['kappa_star']:.4f})")
