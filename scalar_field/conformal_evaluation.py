"""Coverage evaluation of standard and adaptive geodesic conformal prediction."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOTS_DIR
from .geometry import estimate_sigma_geodesic


def evaluate_standard_coverage(model, cal_data, test_data, alpha=0.1):
    """Evaluate fixed-width split conformal intervals."""
    # Calibration residuals
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_preds = model.predict(X_cal).real
    cal_residuals = np.abs(y_cal - cal_preds)
    qhat = np.quantile(cal_residuals, 1 - alpha)
    # Test predictions
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_preds = model.predict(X_test).real
    lower = test_preds - qhat
    upper = test_preds + qhat
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)
    print("Coverage:", coverage)
    print("Average width:", width)
    return coverage, width


def evaluate_coverage(model, test_data, qhat, theta_cv, phi_cv, residuals, h=0.3):
    """Evaluate adaptive conformal coverage."""
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    predictions = model.predict(X_test).real

    sigma_pred = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals,
        k=20,
    )

    lower = predictions - qhat * sigma_pred
    upper = predictions + qhat * sigma_pred
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)
    print("Coverage:", coverage)
    print("Average interval width:", width)

    # High-uncertainty analysis
    q33 = np.quantile(sigma_pred, 0.33)
    q66 = np.quantile(sigma_pred, 0.66)
    low_mask = sigma_pred <= q33
    mid_mask = (sigma_pred > q33) & (sigma_pred <= q66)
    high_mask = sigma_pred > q66
    uncertainty_regions = [
        ("Low uncertainty", low_mask),
        ("Medium uncertainty", mid_mask),
        ("High uncertainty", high_mask),
    ]
    for name, mask in uncertainty_regions:
        region_cov = np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask]))
        region_width = np.mean(upper[mask] - lower[mask])
        print(f"\n{name}:")
        print("Coverage:", region_cov)
        print("Average width:", region_width)

    return coverage, width


def difficulty_diagnostic(model, test_data, theta_cv, phi_cv, residuals_cv):
    """Measure how well sigma(x) tracks absolute prediction residuals."""
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    predictions = model.predict(X_test).real
    true_residuals = np.abs(y_test - predictions)
    sigma_pred = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )
    r = np.corrcoef(sigma_pred, true_residuals)[0, 1]
    print("\n--- DIFFICULTY DIAGNOSTIC ---")
    print("Correlation between sigma(x) and true residuals:", r)

    plt.figure(figsize=(6,5))
    plt.scatter(sigma_pred,true_residuals,alpha=0.5,s=20)
    # Linear regression trend
    m, b = np.polyfit(sigma_pred,true_residuals,1)
    x_line = np.linspace(sigma_pred.min(),sigma_pred.max(),100)
    plt.plot(x_line,m*x_line+b,'r',linewidth=2,label="Linear trend")
    plt.xlabel(r"Estimated difficulty $\hat{\sigma}(x)$")
    plt.ylabel("Absolute prediction residual")
    plt.title(f"Difficulty Estimator Validation\n"f"Correlation = {r:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/difficulty_validation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    return r


def conditional_coverage_evaluation(
    model,
    test_data,
    qhat,
    theta_cv,
    phi_cv,
    residuals_cv,
    n_bins=6,
):
    """Evaluate coverage after binning test points by estimated difficulty."""
    print("\n--- CONDITIONAL COVERAGE EVALUATION ---")
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    predictions = model.predict(X_test).real
    sigma_pred = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )

    lower = predictions - qhat * sigma_pred
    upper = predictions + qhat * sigma_pred
    covered = (y_test >= lower) & (y_test <= upper)
    df = pd.DataFrame({'sigma': sigma_pred, 'covered': covered})
    df['difficulty_bin'] = pd.qcut(df['sigma'], q=n_bins, labels=False)

    results = []
    for b in range(n_bins):
        bin_df = df[df['difficulty_bin'] == b]
        bin_cov = bin_df['covered'].mean()
        results.append({
            'bin': b + 1,
            'count': len(bin_df),
            'avg_sigma': bin_df['sigma'].mean(),
            'coverage': bin_cov,
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    coverage_std = results_df['coverage'].std()
    worst_bin_coverage = results_df['coverage'].min()
    print("\nCoverage Std Across Bins:", coverage_std)
    print("Worst Bin Coverage:", worst_bin_coverage)
    return results_df
