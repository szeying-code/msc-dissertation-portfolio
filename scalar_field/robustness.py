"""Degree-robustness experiment (lambda selection and conformal calibration for each degree)."""

import numpy as np
import pandas as pd

from .config import PLOTS_DIR
from .cross_validation import generate_cv_residuals
from .data import process_temp_data, split_conformal_data
from .geometry import estimate_sigma_geodesic
from .plots_robustness import plot_degree_robustness_results, plot_lambda_selection_comparison
from .spherical_harmonics import SphericalHarmonicRegressor, Spherical_Harmonics
from .storage import ensure_output_dirs


def lambda_grid_search_for_rmse_and_conformal(
    train_data,
    validation_data,
    max_degree,
    lambda_values,
    grad=2,
    alpha=0.1,
    num_folds=5,
    k_sigma=20,
    target_coverage=0.90,
):
    """Select lambda using RMSE and conformal validation criteria."""
    harmonic_method = Spherical_Harmonics()
    X_train = [train_data['Theta'].values, train_data['Phi'].values]
    y_train = train_data['AverageTemperature'].values
    X_validation = [validation_data['Theta'].values, validation_data['Phi'].values]
    y_validation = validation_data['AverageTemperature'].values
    results = []
    print(f"\n--- Lambda search for L = {max_degree} ---")
    for lam in lambda_values:
        print(f"Testing lambda = {lam:.2e}")
        model = SphericalHarmonicRegressor(
            harmonic_method,
            max_degree=max_degree,
            reg_param=lam,
            grad=grad,
        )
        model.fit(X_train, y_train)
        # sigma(x) estimated using training data only
        theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
            train_data,
            max_degree=max_degree,
            reg_param=lam,
            grad=grad,
            num_folds=num_folds,
        )
        cv_rmse = np.sqrt(np.mean(residuals_cv ** 2))
        validation_predictions = model.predict(X_validation).real
        sigma_validation = estimate_sigma_geodesic(
            validation_data['Theta'].values,
            validation_data['Phi'].values,
            theta_cv,
            phi_cv,
            residuals_cv,
            k=k_sigma,
        )
        scores = np.abs(y_validation - validation_predictions) / np.maximum(
            sigma_validation,
            1e-6,
        )
        n = len(scores)
        q_level = np.ceil((1 - alpha) * (n + 1)) / n
        qhat = np.quantile(scores, q_level, method="higher")
        lower = validation_predictions - qhat * sigma_validation
        upper = validation_predictions + qhat * sigma_validation
        validation_coverage = np.mean((y_validation >= lower) & (y_validation <= upper))
        validation_width = np.mean(upper - lower)
        results.append({
            "lambda": lam,
            "cv_rmse": cv_rmse,
            "validation_coverage": validation_coverage,
            "validation_width": validation_width,
            "qhat": qhat,
        })
        print(
            f"  CV RMSE = {cv_rmse:.4f}, "
            f"validation coverage = {validation_coverage:.4f}, "
            f"width = {validation_width:.4f}"
        )
    results_df = pd.DataFrame(results)

    rmse_row = results_df.loc[results_df["cv_rmse"].idxmin()]
    rmse_lambda = rmse_row["lambda"]
    print(f"Best RMSE lambda for L = {max_degree}: {rmse_lambda:.2e}")
    print(f"Mean CV RMSE: {rmse_row['cv_rmse']:.4f}")

    valid = results_df[results_df["validation_coverage"] >= target_coverage]
    if not valid.empty:
        conformal_row = valid.loc[valid["validation_width"].idxmin()]
        print("\nSelected lambda: smallest width with coverage >= target")
    else:
        # fallback if no lambda reaches target coverage
        results_df["coverage_gap"] = np.abs(results_df["validation_coverage"] - target_coverage)
        conformal_row = results_df.sort_values(["coverage_gap", "validation_width"]).iloc[0]
        print("\nNo lambda reached target coverage; selected closest coverage with small width")
    conformal_lambda = conformal_row["lambda"]
    print(f"Best conformal lambda for L = {max_degree}: {conformal_lambda:.2e}")
    print(f"Validation coverage: {conformal_row['validation_coverage']:.4f}")
    print(f"Validation width: {conformal_row['validation_width']:.4f}")
    return rmse_lambda, conformal_lambda, results_df


def evaluate_degree_lambda(
    train_data,
    cal_data,
    test_data,
    L,
    best_lambda,
    best_grad,
    alpha,
    num_folds,
    k_sigma,
    selection_method,
):
    """Train and evaluate one degree/lambda combination."""
    harmonic_method = Spherical_Harmonics()
    X_train = [train_data['Theta'].values, train_data['Phi'].values]
    y_train = train_data['AverageTemperature'].values
    model = SphericalHarmonicRegressor(
        harmonic_method,
        max_degree=L,
        reg_param=best_lambda,
        grad=best_grad,
    )
    model.fit(X_train, y_train)

    theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
        train_data,
        max_degree=L,
        reg_param=best_lambda,
        grad=best_grad,
        num_folds=num_folds
    )

    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    sigma_cal = estimate_sigma_geodesic(
        cal_data['Theta'].values,
        cal_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma
    )
    scores = np.abs(y_cal - cal_predictions) / np.maximum(sigma_cal, 1e-6)
    n = len(scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    qhat = np.quantile(scores, q_level, method="higher")

    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_predictions = model.predict(X_test).real
    test_residuals = y_test - test_predictions
    abs_test_residuals = np.abs(test_residuals)

    standard_cal_residuals = np.abs(y_cal - cal_predictions)
    standard_qhat = np.quantile(standard_cal_residuals, 1 - alpha)
    standard_lower = test_predictions - standard_qhat
    standard_upper = test_predictions + standard_qhat
    standard_coverage = np.mean((y_test >= standard_lower) & (y_test <= standard_upper))
    standard_width = np.mean(standard_upper - standard_lower)

    sigma_test = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma
    )
    adaptive_lower = test_predictions - qhat * sigma_test
    adaptive_upper = test_predictions + qhat * sigma_test
    adaptive_covered = ((y_test >= adaptive_lower) & (y_test <= adaptive_upper))
    adaptive_coverage = np.mean(adaptive_covered)
    adaptive_width = np.mean(adaptive_upper - adaptive_lower)
    correlation = np.corrcoef(sigma_test, abs_test_residuals)[0, 1]

    conditional_df = pd.DataFrame({"sigma": sigma_test, "covered": adaptive_covered})
    conditional_df["difficulty_bin"] = pd.qcut(
        conditional_df["sigma"],
        q=6,
        labels=False,
        duplicates="drop",
    )
    conditional_results = []
    for bin_id in sorted(conditional_df["difficulty_bin"].dropna().unique()):
        bin_df = conditional_df[conditional_df["difficulty_bin"] == bin_id]
        conditional_results.append({
            "bin": int(bin_id) + 1,
            "count": len(bin_df),
            "avg_sigma": bin_df["sigma"].mean(),
            "coverage": bin_df["covered"].mean(),
        })
    conditional_results_df = pd.DataFrame(conditional_results)

    return {
        "selection_method": selection_method,
        "L": L,
        "best_lambda": best_lambda,
        "grad": best_grad,
        "standard_coverage": standard_coverage,
        "adaptive_coverage": adaptive_coverage,
        "standard_width": standard_width,
        "adaptive_width": adaptive_width,
        "correlation": correlation,
        "conditional_coverage_std": conditional_results_df['coverage'].std(),
        "worst_bin_coverage": conditional_results_df['coverage'].min(),
        "rmse": np.sqrt(np.mean(test_residuals ** 2)),
        "mae": np.mean(abs_test_residuals),
    }


def run_degree_robustness_experiment(best_grad):
    ensure_output_dirs()

    data = process_temp_data()
    train_data, validation_data, cal_data, test_data = split_conformal_data(data)

    alpha = 0.1
    num_folds = 5
    grad_value = best_grad
    k_sigma = 20

    degrees = [8, 12, 16, 20, 24, 26, 28, 30]
    lambda_values = np.logspace(-8, 2, 21)
    print("DEGREES BEING TESTED:", degrees)
    rmse_results = []
    conformal_results = []

    for L in degrees:
        print(
            f"\n========== L = {L}: "
            "selecting lambda by RMSE and conformal interval ==========\n"
        )
        rmse_lambda, conformal_lambda, lambda_search_df = lambda_grid_search_for_rmse_and_conformal(
            train_data=train_data,
            validation_data=validation_data,
            max_degree=L,
            lambda_values=lambda_values,
            grad=grad_value,
            alpha=alpha,
            num_folds=num_folds,
            k_sigma=k_sigma,
            target_coverage=0.90,
        )
        lambda_search_df.to_csv(f"{PLOTS_DIR}/lambda_search_L{L}.csv", index=False)

        evaluated = {}
        for selection_method, selected_lambda in [
            ("RMSE", rmse_lambda),
            ("Conformal", conformal_lambda),
        ]:
            lambda_key = float(selected_lambda)
            if lambda_key not in evaluated:
                evaluated[lambda_key] = evaluate_degree_lambda(
                    train_data,
                    cal_data,
                    test_data,
                    L,
                    selected_lambda,
                    grad_value,
                    alpha,
                    num_folds,
                    k_sigma,
                    selection_method,
                )
            result = evaluated[lambda_key].copy()
            result["selection_method"] = selection_method
            if selection_method == "RMSE":
                rmse_results.append(result)
            else:
                conformal_results.append(result)

    rmse_results_df = pd.DataFrame(rmse_results)
    conformal_results_df = pd.DataFrame(conformal_results)
    comparison_df = pd.concat(
        [rmse_results_df, conformal_results_df],
        ignore_index=True,
    )

    print("\n========== DEGREE ROBUSTNESS RESULTS WITH RMSE-SELECTED LAMBDA ==========\n")
    print(rmse_results_df.to_string(index=False))
    print("\n========== DEGREE ROBUSTNESS RESULTS WITH CONFORMAL-SELECTED LAMBDA ==========\n")
    print(conformal_results_df.to_string(index=False))

    rmse_results_df.to_csv(f"{PLOTS_DIR}/degree_robustness_rmse_lambda_results.csv", index=False)
    conformal_results_df.to_csv(f"{PLOTS_DIR}/degree_robustness_conformal_lambda_results.csv", index=False)
    comparison_df.to_csv(f"{PLOTS_DIR}/degree_robustness_lambda_comparison_results.csv", index=False)

    plot_degree_robustness_results(
        rmse_results_df,
        title="Robustness of Adaptive Geodesic Conformal Prediction Across Spherical Harmonic Degrees\nLambda selected by RMSE",
        output_path=f"{PLOTS_DIR}/degree_robustness_rmse_lambda_summary.png",
    )
    plot_degree_robustness_results(
        conformal_results_df,
        title="Robustness of Adaptive Geodesic Conformal Prediction Across Spherical Harmonic Degrees\nLambda selected by conformal coverage with smaller intervals",
        output_path=f"{PLOTS_DIR}/degree_robustness_conformal_lambda_summary.png",
    )
    plot_lambda_selection_comparison(
        rmse_results_df,
        conformal_results_df,
        output_path=f"{PLOTS_DIR}/degree_robustness_lambda_comparison_summary.png",
    )

    return comparison_df
