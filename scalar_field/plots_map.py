"""Opened sphere map of the reconstructed field."""

import numpy as np
import matplotlib.pyplot as plt

from .config import PLOTS_DIR


def plot_opened_sphere_map(
    theta_grid,
    phi_grid,
    temperature,
    uncertainty,
    theta_cities=None,
    phi_cities=None,
    city_values=None,
    city_uncertainty=None,
    city_names=None,
    country_names=None,
    label_uncertainty_percentile=95,
    max_labels=12,
    cut_longitude=0,
    name="opened_globe_uncertainty",
    title="Adaptive Geodesic Conformal Uncertainty Overlay",
):
    import matplotlib as mpl

    # Convert spherical coordinates
    lon_grid = np.degrees(phi_grid)
    lat_grid = 90 - np.degrees(theta_grid)
    lon_shifted = (lon_grid - cut_longitude) % 360
    # Sort latitude (north -> south) and longitude
    sort_lat_idx = np.argsort(lat_grid[:, 0])[::-1]
    sort_lon_idx = np.argsort(lon_shifted[0, :])
    unc_sorted = uncertainty[sort_lat_idx, :][:, sort_lon_idx]
    # Normalise uncertainty
    unc_low = np.nanpercentile(unc_sorted, 5)
    unc_high = np.nanpercentile(unc_sorted, 95)

    if unc_high == unc_low:
        unc_norm = np.zeros_like(unc_sorted)
    else:
        unc_norm = (unc_sorted - unc_low) / (unc_high - unc_low)
        unc_norm = np.clip(unc_norm, 0, 1)

    # Temperature colour limits
    if city_values is not None:
        vmin = np.nanmin(city_values)
        vmax = np.nanmax(city_values)
    else:
        vmin = np.nanmin(temperature)
        vmax = np.nanmax(temperature)

    # Figure
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_facecolor("white")

    # Red uncertainty overlay. Rows are latitude and columns are longitude,
    # so the array can be passed directly to imshow without transposing.
    red_rgba = plt.cm.Reds(unc_norm)
    red_rgba[..., -1] = 0.25 * unc_norm
    ax.imshow(
        red_rgba,
        extent=[cut_longitude, cut_longitude + 360, -90, 90],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
    )
    scatter = None

    # Plot cities
    if theta_cities is not None and phi_cities is not None:
        city_lat = 90 - np.degrees(theta_cities)
        city_lon = np.degrees(phi_cities)
        city_lon = (city_lon - cut_longitude) % 360 + cut_longitude
        scatter = ax.scatter(
            city_lon,
            city_lat,
            c=city_values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=12,
            edgecolors="black",
            linewidths=0.2,
            zorder=3,
        )

        if city_names is not None and country_names is not None and city_uncertainty is not None:
            city_unc = np.asarray(city_uncertainty)
            threshold = np.nanpercentile(city_unc, label_uncertainty_percentile)
            high_unc_idx = np.where(city_unc >= threshold)[0]
            high_unc_idx = high_unc_idx[np.argsort(city_unc[high_unc_idx])[::-1]]
            high_unc_idx = high_unc_idx[:max_labels]
            for i in high_unc_idx:
                label = f"{city_names[i]}, {country_names[i]}"
                ax.text(
                    city_lon[i] + 1.0,
                    city_lat[i] + 0.6,
                    label,
                    fontsize=6,
                    color="black",
                    alpha=0.8,
                    zorder=4,
                )
    # Axes
    ax.set_xlim(cut_longitude, cut_longitude + 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title, fontsize=16, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)

    # Temperature colourbar
    if scatter is not None:
        temp_cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        temp_cbar.set_label("Temperature (°C)")

    # Uncertainty colourbar
    unc_mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=unc_low, vmax=unc_high),
        cmap=plt.cm.Reds,
    )
    unc_mappable.set_array([])
    unc_cbar = plt.colorbar(unc_mappable, ax=ax, pad=0.10)
    unc_cbar.set_label("Uncertainty")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
