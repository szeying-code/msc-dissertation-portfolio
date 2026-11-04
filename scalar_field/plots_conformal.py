"""Figures comparing standard and adaptive conformal prediction."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOTS_DIR
from .geometry import estimate_sigma_geodesic, geodesic_circle_points, spherical_points_to_cartesian


def plot_standard_vs_adaptive_diagnostic(
    model,
    cal_data,
    test_data,
    qhat,
    theta_cv,
    phi_cv,
    residuals_cv,
    alpha=0.1,
    k_sigma=20,
):
    """Compare fixed-width standard CP with variable-width adaptive CP."""
    diagnostic_plot_path = f"{PLOTS_DIR}/standard_vs_adaptive_conformal_diagnostic.png"
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    standard_cal_residuals = np.abs(y_cal - cal_predictions)
    standard_qhat = np.quantile(standard_cal_residuals, 1 - alpha)

    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_predictions = model.predict(X_test).real
    residuals = y_test - test_predictions
    abs_residuals = np.abs(residuals)

    sigma_test = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma,
    )

    standard_lower = test_predictions - standard_qhat
    standard_upper = test_predictions + standard_qhat
    standard_width = standard_upper - standard_lower
    standard_covered = (y_test >= standard_lower) & (y_test <= standard_upper)

    adaptive_lower = test_predictions - qhat * sigma_test
    adaptive_upper = test_predictions + qhat * sigma_test
    adaptive_width = adaptive_upper - adaptive_lower
    adaptive_covered = (y_test >= adaptive_lower) & (y_test <= adaptive_upper)

    lat = 90 - np.degrees(test_data['Theta'].values)
    lon = np.degrees(test_data['Phi'].values)
    lon = ((lon + 180) % 360) - 180

    diagnostic_df = pd.DataFrame({
        'city': test_data['City'].values,
        'country': test_data['Country'].values,
        'lat': lat,
        'lon': lon,
        'true_temp': y_test,
        'pred_temp': test_predictions,
        'residual': residuals,
        'abs_residual': abs_residuals,
        'sigma': sigma_test,
        'standard_width': standard_width,
        'adaptive_width': adaptive_width,
        'standard_covered': standard_covered,
        'adaptive_covered': adaptive_covered,
    })
    diagnostic_df['difficulty_bin'] = pd.qcut(
        diagnostic_df['sigma'],
        q=3,
        labels=['Low difficulty', 'Medium difficulty', 'High difficulty'],
    )

    summary_df = (
        diagnostic_df
        .groupby('difficulty_bin', observed=False)
        .agg(
            count=('city', 'size'),
            mean_sigma=('sigma', 'mean'),
            mean_abs_residual=('abs_residual', 'mean'),
            standard_coverage=('standard_covered', 'mean'),
            adaptive_coverage=('adaptive_covered', 'mean'),
            standard_width=('standard_width', 'mean'),
            adaptive_width=('adaptive_width', 'mean'),
        )
        .reset_index()
    )

    diagnostic_df.to_csv(
        f"{PLOTS_DIR}/standard_vs_adaptive_city_diagnostic.csv",
        index=False,
    )
    summary_df.to_csv(
        f"{PLOTS_DIR}/standard_vs_adaptive_difficulty_summary.csv",
        index=False,
    )

    width_min = min(np.nanmin(standard_width), np.nanmin(adaptive_width))
    width_max = max(np.nanmax(standard_width), np.nanmax(adaptive_width))
    size_min, size_max = 20, 180

    def width_to_size(width):
        if width_max == width_min:
            return np.full_like(width, (size_min + size_max) / 2)
        return size_min + (width - width_min) / (width_max - width_min) * (size_max - size_min)

    standard_sizes = width_to_size(standard_width)
    adaptive_sizes = width_to_size(adaptive_width)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    standard_scatter = axs[0, 0].scatter(
        lon,
        lat,
        c=standard_width,
        s=standard_sizes,
        cmap='Blues',
        vmin=width_min,
        vmax=width_max,
        edgecolors='black',
        linewidths=0.2,
        alpha=0.75,
    )
    axs[0, 0].set_title("(a) Standard CP: Fixed Interval Width")
    axs[0, 0].set_xlabel("Longitude")
    axs[0, 0].set_ylabel("Latitude")
    axs[0, 0].grid(True, linestyle='--', alpha=0.3)

    adaptive_scatter = axs[0, 1].scatter(
        lon,
        lat,
        c=adaptive_width,
        s=adaptive_sizes,
        cmap='Reds',
        vmin=width_min,
        vmax=width_max,
        edgecolors='black',
        linewidths=0.2,
        alpha=0.75,
    )
    axs[0, 1].set_title("(b) Adaptive CP: Variable Interval Width")
    axs[0, 1].set_xlabel("Longitude")
    axs[0, 1].set_ylabel("Latitude")
    axs[0, 1].grid(True, linestyle='--', alpha=0.3)

    plt.colorbar(standard_scatter, ax=axs[0, 0], label="Interval width")
    plt.colorbar(adaptive_scatter, ax=axs[0, 1], label="Interval width")

    axs[1, 0].scatter(sigma_test, abs_residuals, alpha=0.65, s=18)
    axs[1, 0].set_title("(c) Estimated Difficulty vs Absolute Residual")
    axs[1, 0].set_xlabel(r"Estimated difficulty $\hat{\sigma}(x)$")
    axs[1, 0].set_ylabel("Absolute residual")
    axs[1, 0].grid(True, linestyle='--', alpha=0.3)

    x = np.arange(len(summary_df))
    width = 0.35
    axs[1, 1].bar(
        x - width / 2,
        summary_df['standard_coverage'],
        width,
        label='Standard',
    )
    axs[1, 1].bar(
        x + width / 2,
        summary_df['adaptive_coverage'],
        width,
        label='Adaptive',
    )
    axs[1, 1].axhline(1 - alpha, color='black', linestyle='--', linewidth=1.2)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(summary_df['difficulty_bin'], rotation=15)
    axs[1, 1].set_ylim(0, 1.05)
    axs[1, 1].set_title("(d) Coverage by Difficulty Group")
    axs[1, 1].set_ylabel("Coverage")
    axs[1, 1].legend()
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.suptitle(
        "Standard vs Adaptive Geodesic Conformal Diagnostics",
        fontsize=16,
        weight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        diagnostic_plot_path,
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()
    plt.close(fig)

    print("\n--- STANDARD VS ADAPTIVE DIAGNOSTIC BY DIFFICULTY ---")
    print(summary_df.to_string(index=False, formatters={
        'mean_sigma': '{:.3f}'.format,
        'mean_abs_residual': '{:.3f}'.format,
        'standard_coverage': '{:.3f}'.format,
        'adaptive_coverage': '{:.3f}'.format,
        'standard_width': '{:.3f}'.format,
        'adaptive_width': '{:.3f}'.format,
    }))
    print("Saved diagnostic plot.")


def plot_standard_vs_adaptive_geodesic_caps(
    model,
    cal_data,
    test_data,
    qhat,
    theta_cv,
    phi_cv,
    residuals_cv,
    alpha=0.1,
    k_sigma=20,
    max_points=10,
    output_path=f"{PLOTS_DIR}/standard_vs_adaptive_geodesic_caps.png",
):
    """Visualise fixed-size standard caps against variable-size adaptive caps."""
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    standard_qhat = np.quantile(np.abs(y_cal - cal_predictions), 1 - alpha)

    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_predictions = model.predict(X_test).real
    abs_residuals = np.abs(y_test - test_predictions)

    sigma_test = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma,
    )

    standard_width = np.full(len(test_data), 2 * standard_qhat)
    adaptive_width = 2 * qhat * sigma_test

    standard_covered = abs_residuals <= standard_qhat
    adaptive_covered = abs_residuals <= qhat * sigma_test

    sorted_idx = np.argsort(sigma_test)
    if len(sorted_idx) > max_points:
        selected_idx = np.unique(
            np.linspace(0, len(sorted_idx) - 1, max_points).astype(int)
        )
        selected_idx = sorted_idx[selected_idx]
    else:
        selected_idx = sorted_idx

    selected_widths = adaptive_width[selected_idx]
    width_min = np.nanmin(selected_widths)
    width_max = np.nanmax(selected_widths)
    min_radius = 0.10
    max_radius = 0.45

    if width_max == width_min:
        adaptive_radii = np.full(len(selected_idx), (min_radius + max_radius) / 2)
    else:
        adaptive_radii = (
            min_radius
            + (selected_widths - width_min)
            / (width_max - width_min)
            * (max_radius - min_radius)
        )
    standard_radius = np.median(adaptive_radii)

    theta_test = test_data['Theta'].values
    phi_test = test_data['Phi'].values
    all_points = spherical_points_to_cartesian(theta_test, phi_test)
    selected_points = spherical_points_to_cartesian(
        theta_test[selected_idx],
        phi_test[selected_idx],
    )

    fig = plt.figure(figsize=(15, 7))
    axes = [
        fig.add_subplot(121, projection='3d', computed_zorder=False),
        fig.add_subplot(122, projection='3d', computed_zorder=False),
    ]
    titles = [
        "(a) Standard Geodesic CP: Fixed-Size Caps",
        "(b) Adaptive Geodesic CP: Variable-Size Caps",
    ]

    sphere_theta, sphere_phi = np.meshgrid(
        np.linspace(0, np.pi, 32),
        np.linspace(0, 2 * np.pi, 64),
    )
    sphere_x = np.sin(sphere_theta) * np.cos(sphere_phi)
    sphere_y = np.sin(sphere_theta) * np.sin(sphere_phi)
    sphere_z = np.cos(sphere_theta)

    for ax, title in zip(axes, titles):
        ax.plot_wireframe(
            sphere_x,
            sphere_y,
            sphere_z,
            color='lightgray',
            linewidth=0.4,
            alpha=0.25,
        )
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            color='gray',
            s=5,
            alpha=0.15,
        )
        ax.scatter(
            selected_points[:, 0],
            selected_points[:, 1],
            selected_points[:, 2],
            marker='^',
            color='seagreen',
            edgecolors='black',
            linewidths=0.4,
            s=45,
            label='Selected test cities',
            depthshade=False,
            zorder=10,
        )
        ax.set_title(title, fontsize=11, weight='bold')
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)

    for position, data_idx in enumerate(selected_idx):
        standard_circle = geodesic_circle_points(
            theta_test[data_idx],
            phi_test[data_idx],
            standard_radius,
        )
        adaptive_circle = geodesic_circle_points(
            theta_test[data_idx],
            phi_test[data_idx],
            adaptive_radii[position],
        )
        axes[0].plot(
            standard_circle[:, 0],
            standard_circle[:, 1],
            standard_circle[:, 2],
            color='steelblue',
            linewidth=1.3,
            alpha=0.75,
        )
        axes[1].plot(
            adaptive_circle[:, 0],
            adaptive_circle[:, 1],
            adaptive_circle[:, 2],
            color='steelblue',
            linewidth=1.3 + 2.2 * adaptive_radii[position] / max_radius,
            alpha=0.8,
        )

    for ax, covered in zip(axes, [standard_covered, adaptive_covered]):
        misses = selected_idx[~covered[selected_idx]]
        if len(misses) > 0:
            miss_points = spherical_points_to_cartesian(
                theta_test[misses],
                phi_test[misses],
            )
            ax.scatter(
                miss_points[:, 0],
                miss_points[:, 1],
                miss_points[:, 2],
                marker='*',
                color='gold',
                edgecolors='black',
                linewidths=0.4,
                s=75,
                label='Interval miss',
                depthshade=False,
                zorder=20,
            )
        ax.legend(loc='lower left', fontsize=8)

    fig.suptitle(
        "Standard vs Adaptive Geodesic Conformal Cap Diagnostic",
        fontsize=15,
        weight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("Saved geodesic cap diagnostic.")
