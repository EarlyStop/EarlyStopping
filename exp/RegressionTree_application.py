"""
Empirical application: regression-tree early stopping with a pre-tuned
discrepancy threshold.

For each dataset we use a fixed relative threshold c* (where
kappa = c* * Var_hat(Y_train)). The c* values were selected once via 5-fold CV
on the training set; the CV procedure lives in the companion script
``RegressionTree_application_CV.py``.

Comparison columns:
  MSE_ES       : our ES tree stopped at the discrepancy index for kappa*.
  MSE_deep     : the same algorithm grown to ``max_depth`` (no early stopping).
  MSE_pruned   : sklearn cost-complexity-pruned CART (5-fold CV over ccp_alpha).

Relative MSE (smaller -> ES better):
  RelMSE_1 = MSE_ES / MSE_pruned
  RelMSE_2 = MSE_ES / MSE_deep

Timings (wall-clock, seconds):
  t_ES         : single ES tree growth halted at k_stop (deployment cost).
  t_deep       : single ES tree growth to max_depth.
  t_pruned     : full pruned-CART pipeline (path + 5-fold GridSearchCV + refit).
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import EarlyStopping as es  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Pre-tuned configuration
# ---------------------------------------------------------------------------
# c_star values from 5-fold CV (see RegressionTree_application_CV.py).
DATASET_CONFIG = {
    "California Housing":   dict(c_star=0.20, max_n=None, max_depth=12),  # full = 20640
    "Diabetes":             dict(c_star=0.40, max_n=None, max_depth=12),  # full = 442
    "Wine Quality (white)": dict(c_star=0.65, max_n=None, max_depth=12),  # full = 4898
    "Abalone":              dict(c_star=0.50, max_n=None, max_depth=12),  # full = 4177
    "Superconductivity":    dict(c_star=0.02, max_n=None, max_depth=20),  # full = 21263
}

MAX_ALPHAS = 100  # log-spaced cap on the pruning-path CV grid


# ---------------------------------------------------------------------------
# Dataset loaders (return full data; sub-sampling happens later)
# ---------------------------------------------------------------------------

def load_california():
    cal = fetch_california_housing()
    return cal.data.astype(float), cal.target.astype(float)


def load_diabetes_dataset():
    db = load_diabetes()
    return db.data.astype(float), db.target.astype(float)


def load_winequality_white():
    wine = fetch_openml(name="wine-quality-white", version=1, as_frame=True, parser="auto")
    return wine.data.to_numpy(dtype=float), wine.target.to_numpy(dtype=float)


def load_abalone():
    ab = fetch_openml(name="abalone", version=1, as_frame=True, parser="auto")
    df = ab.data.copy()
    target_col = ab.target_names[0] if hasattr(ab, "target_names") and ab.target_names else None
    if target_col is None or target_col not in df.columns:
        y = ab.target.to_numpy(dtype=float)
    else:
        y = df.pop(target_col).to_numpy(dtype=float)
    df = pd.get_dummies(
        df, columns=[c for c in df.columns if df[c].dtype.name == "category"], drop_first=False
    )
    return df.to_numpy(dtype=float), y


def load_superconductivity():
    sc = fetch_openml(data_id=43174, as_frame=True, parser="auto")
    return sc.data.to_numpy(dtype=float), sc.target.to_numpy(dtype=float)


LOADERS = {
    "California Housing":   load_california,
    "Diabetes":             load_diabetes_dataset,
    "Wine Quality (white)": load_winequality_white,
    "Abalone":              load_abalone,
    "Superconductivity":    load_superconductivity,
}


# ---------------------------------------------------------------------------
# Estimator wrappers
# ---------------------------------------------------------------------------

def maybe_subsample(X: np.ndarray, y: np.ndarray, max_n: int | None, seed: int = 0):
    if max_n is None or max_n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_n, replace=False)
    return X[idx], y[idx]


def evaluate_es_at_kappa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kappa: float,
    max_depth: int,
    min_samples_split: int = 1,
):
    """Grow two fresh ES trees and time them separately.

    - Deep tree: grown to ``max_depth``. Used for MSE_deep and to discover
      k_stop (the discrepancy index where residuals[k] <= kappa).
    - ES tree: a separate fresh growth halted at ``max_depth=k_stop`` -- this
      is the production cost of early stopping.
    """
    t = time.perf_counter()
    deep_tree = es.RegressionTree(
        design=X_train, response=y_train, min_samples_split=min_samples_split
    )
    deep_tree.iterate(max_depth=max_depth)
    t_deep = time.perf_counter() - t

    max_grown = len(deep_tree.residuals) - 1
    k_stop = deep_tree.get_discrepancy_stop(critical_value=kappa)
    if k_stop is None:
        k_stop = max_grown
    k_stop = int(max(0, min(k_stop, max_grown)))
    pred_deep = deep_tree.predict(X_test, depth=max_grown)
    mse_deep = float(np.mean((y_test - pred_deep) ** 2))

    t = time.perf_counter()
    es_tree = es.RegressionTree(
        design=X_train, response=y_train, min_samples_split=min_samples_split
    )
    es_tree.iterate(max_depth=max(k_stop, 0))
    t_es = time.perf_counter() - t
    pred_es = es_tree.predict(X_test, depth=k_stop)
    mse_es = float(np.mean((y_test - pred_es) ** 2))

    return {"mse_es": mse_es, "mse_deep": mse_deep, "k_stop": k_stop,
            "t_es": t_es, "t_deep": t_deep}


def smart_alpha_grid(alphas: np.ndarray, max_size: int | None = 50) -> np.ndarray:
    """Log-spaced sub-sample of the cost-complexity pruning path.

    The full pruning path returned by sklearn can have thousands of α breakpoints.
    We log-space across ``[alpha_min_positive, alpha_max]`` so that consecutive
    grid points correspond to similar *relative* shrinkage -- the same logic as
    standard regularization-path practice in lasso/ridge. ``α = 0`` is always
    kept (deepest tree). Returns the full path unchanged if it is already
    small enough, or unconditionally if ``max_size`` is ``None``.
    """
    alphas = np.unique(np.asarray(alphas, dtype=float))
    if max_size is None or len(alphas) <= max_size:
        return alphas
    has_zero = (alphas[0] == 0.0)
    pos = alphas[alphas > 0]
    if len(pos) == 0:
        return alphas
    n_log = max_size - (1 if has_zero else 0)
    log_targets = np.geomspace(pos.min(), pos.max(), n_log)
    snapped = np.unique([alphas[np.abs(alphas - t).argmin()] for t in log_targets])
    if has_zero:
        snapped = np.unique(np.concatenate([[0.0], snapped]))
    return snapped


def evaluate_pruned_cart(X_train, y_train, X_test, y_test, seed=0,
                         max_alphas: int | None = 50):
    """sklearn cost-complexity-pruned CART; ccp_alpha tuned by 5-fold CV.

    The full pruning path is filtered down to at most ``max_alphas`` log-spaced
    values via :func:`smart_alpha_grid` to keep the CV cost bounded.
    """
    t = time.perf_counter()
    path = DecisionTreeRegressor(random_state=seed).cost_complexity_pruning_path(X_train, y_train)
    alphas_full = path.ccp_alphas
    alphas = smart_alpha_grid(alphas_full, max_size=max_alphas)
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=seed),
        param_grid={"ccp_alpha": alphas},
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    ).fit(X_train, y_train)
    return {
        "mse_pruned":     float(np.mean((y_test - grid.predict(X_test)) ** 2)),
        "best_ccp_alpha": float(grid.best_params_["ccp_alpha"]),
        "n_alphas":       int(len(alphas)),
        "n_alphas_full":  int(len(alphas_full)),
        "t_pruned":       time.perf_counter() - t,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(name: str, X: np.ndarray, y: np.ndarray, c_star: float,
            max_depth: int, max_n: int | None, seed: int = 21,
            max_alphas: int | None = 50):
    X, y = maybe_subsample(X, y, max_n=max_n, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    var_tr = float(np.var(y_tr))
    kappa = c_star * var_tr

    es_out = evaluate_es_at_kappa(X_tr, y_tr, X_te, y_te,
                                  kappa=kappa, max_depth=max_depth)
    pruned_out = evaluate_pruned_cart(X_tr, y_tr, X_te, y_te,
                                      seed=seed, max_alphas=max_alphas)

    return {
        "Dataset":            name,
        "n_train":            X_tr.shape[0],
        "n_test":             X_te.shape[0],
        "d":                  X_tr.shape[1],
        "c_star":             c_star,
        "kappa":              kappa,
        "k_stop":             es_out["k_stop"],
        "MSE_ES":             es_out["mse_es"],
        "MSE_pruned":         pruned_out["mse_pruned"],
        "MSE_deep":           es_out["mse_deep"],
        "RelMSE_1=ES/pruned": es_out["mse_es"] / pruned_out["mse_pruned"],
        "RelMSE_2=ES/deep":   es_out["mse_es"] / es_out["mse_deep"],
        "t_ES_s":             es_out["t_es"],
        "t_deep_s":           es_out["t_deep"],
        "t_pruned_s":         pruned_out["t_pruned"],
        "n_alphas":           pruned_out["n_alphas"],
        "n_alphas_full":      pruned_out["n_alphas_full"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(21)
    rows = []
    for name, cfg in DATASET_CONFIG.items():
        print(f"[{name}]  loading + running with c*={cfg['c_star']} ...")
        X, y = LOADERS[name]()
        row = run_one(
            name, X, y,
            c_star=cfg["c_star"], max_depth=cfg["max_depth"], max_n=cfg["max_n"],
            max_alphas=MAX_ALPHAS,
        )
        rows.append(row)
        print(
            f"  n_train={row['n_train']:5d}  d={row['d']:3d}  k_stop={row['k_stop']:2d}  "
            f"MSE_ES={row['MSE_ES']:8.4f}  MSE_pruned={row['MSE_pruned']:8.4f}  "
            f"MSE_deep={row['MSE_deep']:8.4f}  "
            f"RelMSE_1={row['RelMSE_1=ES/pruned']:.3f}  "
            f"RelMSE_2={row['RelMSE_2=ES/deep']:.3f}  "
            f"t_ES={row['t_ES_s']:6.2f}s  t_deep={row['t_deep_s']:6.2f}s  "
            f"t_pruned={row['t_pruned_s']:6.2f}s  "
            f"(n_alphas={row['n_alphas']}/{row['n_alphas_full']})"
        )

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("Cross-dataset summary (single split; fixed c* per dataset)")
    print("=" * 90)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
