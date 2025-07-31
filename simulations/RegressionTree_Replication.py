# Clear all variables from the global namespace 
for name in list(globals()):
    if not name.startswith("_"):
        del globals()[name]

# Import required libraries and modules
import numpy as np
np.random.seed(21)
import os
import EarlyStopping as es
import importlib
import matplotlib.pyplot as plt
import RegressionTree_additive_generation as data_generation
from joblib import Parallel, delayed

# Reload modules to ensure changes are applied
importlib.reload(data_generation)
importlib.reload(es)


def methods_stopping(X_train, y_train, X_test, noise_level, noise, true_signal, true_signal_test):

    return global_ES(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        noise=noise,
        true_signal=true_signal,
        signal_test=true_signal_test,
        kappa=noise_level,
    )


def global_ES(X_train, y_train, X_test, noise, true_signal, signal_test, kappa):

    regression_tree = es.RegressionTree(
        design=X_train, response=y_train, min_samples_split=1, true_signal=true_signal, true_noise_vector=noise
    )
    regression_tree.iterate(max_depth=30)
    early_stopping_iteration = regression_tree.get_discrepancy_stop(critical_value=kappa)

    # Global ES prediction
    prediction_global_k1 = regression_tree.predict(X_test, depth=early_stopping_iteration)
    mse_global = np.mean((prediction_global_k1 - signal_test) ** 2)

    # Interpolation:
    if early_stopping_iteration == 0:
        mse_global_inter = mse_global
        print("No Interpolation done.")
    else:
        prediction_global_k = regression_tree.predict(X_test, depth=early_stopping_iteration - 1)
        residuals = regression_tree.residuals
        r2_k1 = residuals[early_stopping_iteration]
        r2_k = residuals[early_stopping_iteration - 1]
        alpha = 1 - np.sqrt(1 - (r2_k - kappa) / (r2_k - r2_k1))
        predictions_interpolated = (1 - alpha) * prediction_global_k + alpha * prediction_global_k1
        mse_global_inter = np.mean((predictions_interpolated - signal_test) ** 2)

    # Oracle on test set for interpolated global and global
    mse_global_list = []
    mse_global_interpolated_list = []
    residuals = regression_tree.residuals
    max_possible_depth = len(residuals)

    for iter in range(1, max_possible_depth):
        predictions_global_k1 = regression_tree.predict(X_test, depth=iter)
        predictions_global_k = regression_tree.predict(X_test, depth=iter - 1)
        r2_k1_test = regression_tree.residuals[iter]
        r2_k_test = regression_tree.residuals[iter - 1]
        alpha_test = 1 - np.sqrt(1 - (r2_k_test - kappa) / (r2_k_test - r2_k1_test))
        predictions_interpolated = (1 - alpha_test) * predictions_global_k + alpha_test * predictions_global_k1
        mse_global_interpolated_list.append(np.mean((predictions_interpolated - signal_test) ** 2))

        # Empirical MSE on test set
        mse_global_temp = np.mean((predictions_global_k1 - signal_test) ** 2)
        mse_global_list.append(mse_global_temp)

    global_stopping_iteration_oracle = np.argmin(mse_global_list) + 1
    oracle_tree = es.RegressionTree(
        design=X_train, response=y_train, min_samples_split=1, true_signal=true_signal, true_noise_vector=noise
    )
    oracle_tree.iterate(max_depth=global_stopping_iteration_oracle + 1)

    # Interpolated oracle:
    mse_oracle_global_interpolated = np.nanmin(mse_global_interpolated_list)
    # Global Oracle:
    mse_global_min = np.min(mse_global_list)
    # Take the min of both:
    mse_oracle_early_stopping = min(mse_oracle_global_interpolated, mse_global_min)

    return mse_global, mse_global_inter, mse_oracle_early_stopping


def single_monte_carlo_run(dgp, noise_level, run_idx):
    """Execute a single Monte Carlo run for the given DGP."""
    
    if dgp == "additive_smooth" or dgp == "additive_step" or "additive_linear" or "additive_hills":
        n_train = 1000
        n_test = 1000
        d = 30
        X_train = np.random.uniform(-2.5, 2.5, size=(n_train, d))
        X_test = np.random.uniform(-2.5, 2.5, size=(n_test, d))

    y_train, noise = data_generation.generate_data_from_X(X_train, noise_level, dgp_name=dgp, add_noise=True)
    y_test, nuisance = data_generation.generate_data_from_X(X_test, noise_level, dgp_name=dgp, add_noise=True)
    f, nuisance = data_generation.generate_data_from_X(X_train, noise_level, dgp_name=dgp, add_noise=False)
    f_test, nuisance = data_generation.generate_data_from_X(X_test, noise_level, dgp_name=dgp, add_noise=False)
    
    print(f"{dgp}, global, {run_idx}")
    
    results = methods_stopping(
        X_train,
        y_train,
        X_test,
        noise_level=noise_level,
        noise=noise,
        true_signal=f,
        true_signal_test=f_test,
    )
    
    if not isinstance(results, tuple):
        results = (results,)
    
    return results


def run_simulation_wrapper(dgp, M=200, noise_level=1):
    """Run simulation with parallel Monte Carlo runs."""
    
    # Parallelize across Monte Carlo runs (auto-detect CPU cores, leave one free for system)
    n_cores = max(1, os.cpu_count() - 1)
    results_list = Parallel(n_jobs=n_cores)(delayed(single_monte_carlo_run)(dgp, noise_level, i) for i in range(M))
    
    # Aggregate results
    mspe_list = []
    additional_metric_list = []
    additional_metric2_list = []
    
    for results in results_list:
        mspe = results[0]
        mspe_list.append(mspe)
        
        # Handle the additional metric if it exists
        if len(results) > 1:
            additional_metric_list.append(results[1])
        if len(results) > 2:
            additional_metric2_list.append(results[2])
    
    mean_mspe = np.array(mspe_list)
    mean_additional_metric = np.array(additional_metric_list) if additional_metric_list else None
    mean_additional_metric2 = np.array(additional_metric2_list) if additional_metric2_list else None
    
    # Return in the same format as before (global ES, global inter ES, global inter Oracle)
    return np.column_stack((mean_mspe, mean_additional_metric, mean_additional_metric2))





def create_custom_boxplot(data, labels, dgp_names, y_lim_lower, y_lim_upper, fig_dir, name):
    """
    Create a boxplot with custom styling and a second x-axis row for DGP names.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    # Define custom colors
    colors = ["blue", "purple"] * (len(labels) // 2)

    # Set colors for each box
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)

    # Making whiskers, caps, and medians thicker and grey
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.5)
    for cap in bp["caps"]:
        cap.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_linewidth(1.5)

    # Add a horizontal line at y=1
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.5)

    # Enable gridlines and set y-axis limits
    ax.grid(True)
    ax.set_ylim(y_lim_lower, y_lim_upper)

    # Customize tick labels
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Add second-row DGP labels beneath the x-ticks
    for i, dgp_label in enumerate(dgp_names):
        x_pos = 2 * i + 1.5  # Center position for each pair (Global, Global Int)
        plt.text(x_pos, y_lim_lower - 0.1, dgp_label, ha="center", va="top", fontsize=14, color="black")

    # Save the plot
    plt.savefig(os.path.join(fig_dir, f"regtree_{name}.png"), bbox_inches="tight", dpi=300)

    plt.tight_layout()
    plt.show()


def main():

    dgps = ["additive_smooth", "additive_step", "additive_linear", "additive_hills"]

    # Run DGPs sequentially, but parallelize Monte Carlo runs within each DGP
    results = []
    for dgp in dgps:
        print(f"Running DGP: {dgp}")
        dgp_result = run_simulation_wrapper(dgp)
        results.append(dgp_result)
    
    results = np.round(results, 6)

    # Automatically determine the directory for saving plots (same as script location)
    fig_dir = os.path.dirname(os.path.abspath(__file__))

    dgps = ["additive smooth", "additive step", "additive linear", "additive hills"]

    dgp_groups = [
        {"dgps": dgps[:2], "results": results[:2], "plot_title": "First and Second DGPs", "file_name": "plot_1"},
        {"dgps": dgps[2:], "results": results[2:], "plot_title": "Third and Fourth DGPs", "file_name": "plot_2"},
    ]

    for group in dgp_groups:
        all_data = []
        labels = []
        dgp_names = []
        for dgp_name, result in zip(group["dgps"], group["results"]):
            interpolated_oracle = result[:, 2].reshape(-1, 1)
            global_mspe = result[:, 0].reshape(-1, 1)
            global_inter_mspe = result[:, 1].reshape(-1, 1)

            relative_global = np.sqrt(interpolated_oracle / global_mspe)
            relative_inter = np.sqrt(interpolated_oracle / global_inter_mspe)

            # Filter due to numerics
            mask_inter = relative_inter > 1.0001
            filtered_relative_inter = relative_inter[~mask_inter]

            # Prepare data
            all_data.append(relative_global.flatten())
            all_data.append(filtered_relative_inter.flatten())

            # Add 'Global' and 'Global Int' for each DGP
            labels.extend(["global", "global inter"])

            # Add the DGP name for the second axis
            dgp_names.append(dgp_name)

        # Create and save the boxplot with custom labels
        create_custom_boxplot(all_data, labels, dgp_names, 0.3, 1.3, fig_dir, group["file_name"])


if __name__ == "__main__":
    main()
