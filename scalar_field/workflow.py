"""Main workflows: model training with conformal calibration, and field reconstruction."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch

from .config import EARTH_RADIUS_METRES, MODELS_DIR, PLOTS_DIR
from .conformal_evaluation import conditional_coverage_evaluation, difficulty_diagnostic, evaluate_coverage, evaluate_standard_coverage
from .cross_validation import generate_cv_residuals, grid_search
from .data import process_temp_data, split_conformal_data
from .geometry import estimate_sigma_geodesic, geodesic_distance
from .plots_conformal import plot_standard_vs_adaptive_diagnostic, plot_standard_vs_adaptive_geodesic_caps
from .plots_map import plot_opened_sphere_map
from .spherical_harmonics import SphericalHarmonicRegressor, Spherical_Harmonics
from .storage import ensure_output_dirs, load_model, load_qhat, save_model


def train_model():
    """Main entry point for training."""
    ensure_output_dirs()
    opt_choice = input('(1) Use grid search to optimise hyperparameters  (2) Manually enter parameters: ')
    data = process_temp_data()
    train_data, _, cal_data, test_data = split_conformal_data(data)

    print(train_data.head())
    print(cal_data.head())
    print(test_data.head())
    # Previous full-data input, kept as a reminder
    # Previous full-data target, kept as a reminder
    harmonic_method = Spherical_Harmonics()
    # Base model used by grid search
    base_model = SphericalHarmonicRegressor(
        harmonic_method,
        max_degree=1,
        reg_param=0.0,
        grad=0,
    )
    # Best gradient found by grid search
    best_grad = None
    # Use training set only for fitting
    X_train = [train_data['Theta'].values, train_data['Phi'].values]
    y_train = train_data['AverageTemperature'].values

    if opt_choice == '1':
        num_folds = int(input('Choose the number of folds to use: '))
        print("Define the parameter search space: ")
        n_lambda = int(input("Enter the number of regularisation parameters: "))
        max_lambda = int(input("Maximum regularisation parameter value (10^(?)): "))
        min_lambda = int(input("Minimum regularisation parameter value (10^(?)): "))
        lambda_values = np.logspace(int(min_lambda), int(max_lambda), int(n_lambda))

        max_grad = int(input("Enter the largest gradient to penalise: "))
        grad_values = np.arange(0, int(max_grad) + 1)

        max_degree = int(input("Enter the maximum degree of spherical harmonic: "))
        # Grid search now uses training data only
        # This avoids leaking calibration/test data into model selection
        model, best_lambda, best_grad = grid_search(
            data,
            max_degree,
            lambda_values,
            grad_values,
            base_model,
            num_folds,
        )
        print(f"\nBest lambda: {best_lambda}")
        print(f"Best gradient: {best_grad}")
        model.fit(X_train, y_train)

    elif opt_choice == '2':
        lambda_value = float(input("Enter the regularisation parameter: "))
        grad_value = int(input("Enter the gradient to penalise: "))
        max_degree = int(input("Enter the maximum degree of spherical harmonics: "))
        model = SphericalHarmonicRegressor(
            harmonic_method,
            max_degree=max_degree,
            reg_param=lambda_value,
            grad=grad_value,
        )
        model.fit(X_train, y_train)
        print(model.coefficients)
        # Use manually selected gradient
        best_grad = grad_value

    else:
        print('Invalid choice, please try again.')
        return train_model()

    alpha = 0.1
    theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
        train_data,
        max_degree=model.max_degree,
        reg_param=model.reg_param,
        grad=model.grad,
        num_folds=5,
    )
    # Build residual field for local difficulty sigma(x)
    print("Geodesic residual field prepared.")
    # Compute adaptive conformal calibration scores
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    sigma_pred = estimate_sigma_geodesic(
        cal_data['Theta'].values,
        cal_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )
    scores = np.abs(y_cal - cal_predictions) / np.maximum(sigma_pred, 1e-6)
    n = len(scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    qhat = np.quantile(scores, q_level, method="higher")

    print("\n--- STANDARD CONFORMAL ---")
    evaluate_standard_coverage(model, cal_data, test_data, alpha)
    print("\n--- ADAPTIVE CONFORMAL ---")
    evaluate_coverage(model, test_data, qhat, theta_cv, phi_cv, residuals_cv)
    plot_standard_vs_adaptive_diagnostic(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
        alpha=alpha,
    )
    plot_standard_vs_adaptive_geodesic_caps(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
        alpha=alpha,
    )

    difficulty_diagnostic(model, test_data, theta_cv, phi_cv, residuals_cv)
    conditional_coverage_evaluation(model, test_data, qhat, theta_cv, phi_cv, residuals_cv)
    print("First 10 Calibration scores:")
    print(scores[:10])
    print("Conformal threshold qhat:", qhat)
    save_model(model, "temp_model")
    with open(f"{MODELS_DIR}/temp_qhat.pkl", "wb") as f:
        pickle.dump(qhat, f)
    # Return best gradient so main() can pass it
    # to the degree robustness experiment
    return model, best_grad


def reconstruct_field(n):
    """Reconstruct the scalar field on an n x n spherical grid."""
    ensure_output_dirs()
    high_uncertainty_percentile = 95
    high_uncertainty_max_labels = 20
    data = process_temp_data()
    train_data, _, cal_data, test_data = split_conformal_data(data)
    r = EARTH_RADIUS_METRES
    
    harmonic_method = Spherical_Harmonics()
    model = load_model("temp_model")
    qhat = load_qhat("temp_qhat")
    theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
        train_data,
        max_degree=model.max_degree,
        reg_param=model.reg_param,
        grad=model.grad,
        num_folds=5,
    )
    plot_standard_vs_adaptive_diagnostic(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
    )
    plot_standard_vs_adaptive_geodesic_caps(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
    )
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(0, np.pi, n),
        np.linspace(0, 2 * np.pi, n),
        indexing="ij",
    )
    theta = theta_grid.flatten()
    phi = phi_grid.flatten()
    
    A_torch = torch.tensor(
        harmonic_method.design_matrix_gpu(theta, phi, model.max_degree),
        dtype=torch.complex128,
        device=harmonic_method.device,
    )
    coefficients_torch = torch.tensor(
        model.coefficients,
        dtype=torch.complex128,
        device=harmonic_method.device,
    )

    reconstruction_torch = A_torch @ coefficients_torch
    reconstruction = reconstruction_torch.cpu().numpy()
    reconstruction = reconstruction.reshape(theta_grid.shape)

    sigma_pred = estimate_sigma_geodesic(theta, phi, theta_cv, phi_cv, residuals_cv, k=20)
    sigma_pred = sigma_pred.reshape(theta_grid.shape)
    lower = reconstruction.real - qhat * sigma_pred
    upper = reconstruction.real + qhat * sigma_pred
    uncertainty = upper - lower

    print("Conformal interval example at first grid point:")
    print("Prediction:", reconstruction.real[0, 0])
    print("Lower:", lower[0, 0])
    print("Upper:", upper[0, 0])

    theta_data = data['Theta'].values
    phi_data = data['Phi'].values
    X_data = [theta_data, phi_data]
    pred_data = model.predict(X_data).real

    # Compare observed and predicted city temperatures
    true_temp = data['AverageTemperature'].values
    residuals = true_temp - pred_data

    # Diagnostic metrics
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Bias: {bias:.3f}")

    # Residual distribution
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.savefig(f"{PLOTS_DIR}/residual_histogram.png", dpi=300)
    plt.show()

    # Observed vs reconstructed 3D comparison
    fig = plt.figure(figsize=(14, 6))
    vmin = min(true_temp.min(), pred_data.min())
    vmax = max(true_temp.max(), pred_data.max())

    ax1 = fig.add_subplot(121, projection='3d')
    x, y, z = harmonic_method.convert_to_cartesian(theta_data, phi_data, r)
    sc1 = ax1.scatter(x, y, z, c=true_temp, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title("Observed City Temperatures")

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(x, y, z, c=pred_data, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title("Reconstructed Temperatures")
    plt.colorbar(sc1, ax=ax1, shrink=0.6)
    plt.colorbar(sc2, ax=ax2, shrink=0.6)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/observed_vs_reconstructed.png", dpi=300)
    plt.show()

    # Paper-style diagnostic plot
    latitudes = 90 - np.degrees(theta_data)  # Convert theta to latitude
    longitudes = np.degrees(phi_data)  # Convert longitude to degrees
    longitudes = ((longitudes + 180) % 360) - 180  # Convert longitude to [-180, 180]
    # Estimate adaptive uncertainty at city points
    sigma_city = estimate_sigma_geodesic(
        theta_data,
        phi_data,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )
    # Create dataframe
    plot_df = pd.DataFrame({
        'city': data['City'].values,
        'country': data['Country'].values,
        'theta': theta_data,
        'phi': phi_data,
        'lat': latitudes,
        'lon': longitudes,
        'true_temp': true_temp,
        'pred_temp': pred_data,
        'sigma': sigma_city,
    })
    # Sort by latitude for cleaner plotting
    plot_df = plot_df.sort_values('lat')
    plot_df['upper90'] = plot_df['pred_temp'] + qhat * plot_df['sigma']
    plot_df['lower90'] = plot_df['pred_temp'] - qhat * plot_df['sigma']

    # Longitude-sliced diagnostic plots
    lon_bins = [-180, -120, -60, 0, 60, 120, 180]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(6):
        lon_min = lon_bins[i]
        lon_max = lon_bins[i + 1]
        subset = plot_df[
            (plot_df['lon'] >= lon_min)
            & (plot_df['lon'] < lon_max)
        ].copy()
        subset = subset.sort_values('lat')
        ax = axes[i]

        ax.scatter(
            subset['lat'],
            subset['true_temp'],
            color='green',
            s=15,
            alpha=0.7,
            label='Observed',
        )
        ax.plot(
            subset['lat'],
            subset['pred_temp'],
            color='red',
            linewidth=2,
            label='Prediction',
        )
        ax.fill_between(
            subset['lat'],
            subset['lower90'],
            subset['upper90'],
            color='red',
            alpha=0.2,
            label='90% adaptive geodesic conformal interval',
        )

        outside_ci = subset[
            (subset['true_temp'] < subset['lower90'])
            | (subset['true_temp'] > subset['upper90'])
        ].copy()
        if not outside_ci.empty:
            outside_ci['ci_miss_size'] = np.maximum(
                outside_ci['lower90'] - outside_ci['true_temp'],
                outside_ci['true_temp'] - outside_ci['upper90'],
            )
            outside_ci = outside_ci.sort_values(
                'ci_miss_size',
                ascending=False,
            ).head(5)
            ax.scatter(
                outside_ci['lat'],
                outside_ci['true_temp'],
                color='gold',
                edgecolors='black',
                linewidths=0.5,
                s=35,
                alpha=0.90,
                zorder=5,
                label='Outside CI',
            )
            for _, row in outside_ci.iterrows():
                ax.text(
                    row['lat'] + 0.5,
                    row['true_temp'] + 0.4,
                    f"{row['city']}, {row['country']}",
                    fontsize=7,
                    color='black',
                    alpha=0.85,
                )

        ax.set_title(f'Longitude {lon_min} deg to {lon_max} deg')
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Temperature")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{PLOTS_DIR}/longitude_sliced_temperature.png", dpi=300)
    plt.show()

    plot_df['residual'] = plot_df['true_temp'] - plot_df['pred_temp']

    print("\n--- Highest Adaptive Geodesic Uncertainty Cities ---")
    high_uncertainty_threshold = np.nanpercentile(
        plot_df['sigma'],
        high_uncertainty_percentile,
    )
    high_uncertainty_cities = (
        plot_df[plot_df['sigma'] >= high_uncertainty_threshold]
        .sort_values('sigma', ascending=False)
        .head(high_uncertainty_max_labels)
        .copy()
    )
    high_uncertainty_cities['interval_width'] = (
        high_uncertainty_cities['upper90'] - high_uncertainty_cities['lower90']
    )

    density_radii_km = [250, 500, 1000]
    earth_radius_km = 6371
    all_theta = plot_df['theta'].values
    all_phi = plot_df['phi'].values

    for row_index, row in high_uncertainty_cities.iterrows():
        distances = geodesic_distance(
            row['theta'],
            row['phi'],
            all_theta,
            all_phi,
        )
        neighbour_distances = np.sort(distances[distances > 1e-12])

        for radius_km in density_radii_km:
            angular_radius = radius_km / earth_radius_km
            high_uncertainty_cities.loc[
                row_index,
                f'observations_within_{radius_km}km',
            ] = np.sum((distances <= angular_radius) & (distances > 1e-12))

        if len(neighbour_distances) >= 20:
            distance_to_20th = neighbour_distances[19] * earth_radius_km
        else:
            distance_to_20th = np.nan
        high_uncertainty_cities.loc[
            row_index,
            'distance_to_20th_neighbour_km',
        ] = distance_to_20th

    high_uncertainty_columns = [
        'city',
        'country',
        'lat',
        'lon',
        'true_temp',
        'pred_temp',
        'residual',
        'sigma',
        'lower90',
        'upper90',
        'interval_width',
        'observations_within_250km',
        'observations_within_500km',
        'observations_within_1000km',
        'distance_to_20th_neighbour_km',
    ]
    print(
        high_uncertainty_cities[high_uncertainty_columns]
        .to_string(index=False, formatters={
            'lat': '{:.2f}'.format,
            'lon': '{:.2f}'.format,
            'true_temp': '{:.2f}'.format,
            'pred_temp': '{:.2f}'.format,
            'residual': '{:.2f}'.format,
            'sigma': '{:.2f}'.format,
            'lower90': '{:.2f}'.format,
            'upper90': '{:.2f}'.format,
            'interval_width': '{:.2f}'.format,
            'observations_within_250km': '{:.0f}'.format,
            'observations_within_500km': '{:.0f}'.format,
            'observations_within_1000km': '{:.0f}'.format,
            'distance_to_20th_neighbour_km': '{:.1f}'.format,
        })
    )
    high_uncertainty_cities[high_uncertainty_columns].to_csv(
        f"{PLOTS_DIR}/high_uncertainty_cities_top5pct_top{high_uncertainty_max_labels}.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(20, 5.5))
    ax.axis('off')
    table_df = high_uncertainty_cities[high_uncertainty_columns].copy()
    numeric_columns = [
        'lat',
        'lon',
        'true_temp',
        'pred_temp',
        'residual',
        'sigma',
        'lower90',
        'upper90',
        'interval_width',
        'observations_within_250km',
        'observations_within_500km',
        'observations_within_1000km',
        'distance_to_20th_neighbour_km',
    ]
    for column in numeric_columns:
        table_df[column] = table_df[column].map(lambda value: f"{value:.2f}")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.45)
    ax.set_title(
        "Highest-Uncertainty Cities Labelled on the Map",
        fontsize=12,
        weight='bold',
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/high_uncertainty_cities_top5pct_top{high_uncertainty_max_labels}.png",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)

    shared_temp_vmin = min(np.nanmin(pred_data), np.nanmin(true_temp))
    shared_temp_vmax = max(np.nanmax(pred_data), np.nanmax(true_temp))
    print(f"Shared opened-globe temperature scale: {shared_temp_vmin:.2f} to {shared_temp_vmax:.2f} deg C")

    # Geographic residual diagnostic plot
    plt.figure(figsize=(12, 6))
    sc = plt.scatter(
        plot_df['lon'],
        plot_df['lat'],
        c=plot_df['residual'],
        cmap='coolwarm',
        s=40,
    )
    plt.colorbar(sc, label='Residual')
    top_errors = plot_df.reindex(
        plot_df['residual'].abs().sort_values(ascending=False).index
    ).head(10)
    for _, row in top_errors.iterrows():
        plt.text(row['lon'], row['lat'], row['city'], fontsize=8)

    country_centers = plot_df.groupby('country')[['lon', 'lat']].mean().reset_index()
    countries_to_label = ['Brazil', 'Australia', 'India', 'China', 'United States']
    filtered = country_centers[country_centers['country'].isin(countries_to_label)]
    for _, row in filtered.iterrows():
        plt.text(row['lon'], row['lat'], row['country'], fontsize=10, weight='bold')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Residual Map")
    plt.grid(True)
    plt.savefig(f"{PLOTS_DIR}/geographic_residual_map.png", dpi=300)
    plt.show()

    # Opened sphere map with adaptive geodesic conformal uncertainty
    plot_opened_sphere_map(
        theta_grid=theta_grid,
        phi_grid=phi_grid,
        temperature=reconstruction.real,
        uncertainty=uncertainty,
        theta_cities=theta_data,
        phi_cities=phi_data,
        city_values=pred_data,
        city_uncertainty=sigma_city,
        city_names=data["City"].values,
        country_names=data["Country"].values,
        label_uncertainty_percentile=high_uncertainty_percentile,
        max_labels=high_uncertainty_max_labels,
        cut_longitude=0,
        name="opened_globe_uncertainty",
        title="Predicted Adaptive Geodesic Conformal Uncertainty Overlay",
    )
    harmonic_method.plot_prediction_and_uncertainty(
        theta_grid,
        phi_grid,
        reconstruction.real,
        uncertainty,
        theta_data,
        phi_data,
        pred_data,
    )
