"""Figures for the degree-robustness experiment and lambda selection."""

import matplotlib.pyplot as plt
import os

from .config import PLOTS_DIR


def plot_degree_robustness_results(
    results_df,
    title="Robustness of Adaptive Geodesic Conformal Prediction Across Spherical Harmonic Degrees",
    output_path=f"{PLOTS_DIR}/degree_robustness_summary.png",
):

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # ======================================================
    # (a) Coverage
    # ======================================================
    axs[0, 0].plot(
        results_df["L"],
        results_df["standard_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Standard",
    )

    axs[0, 0].plot(
        results_df["L"],
        results_df["adaptive_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Adaptive",
    )

    axs[0, 0].axhline(
        y=0.90,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Target (90%)",
    )

    axs[0, 0].set_title("(a) Coverage")
    axs[0, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 0].set_ylabel("Coverage")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # ======================================================
    # (b) Interval Width
    # ======================================================
    axs[0, 1].plot(
        results_df["L"],
        results_df["standard_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Standard",
    )

    axs[0, 1].plot(
        results_df["L"],
        results_df["adaptive_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Adaptive",
    )

    axs[0, 1].set_title("(b) Average Interval Width")
    axs[0, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 1].set_ylabel("Interval Width")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # ======================================================
    # (c) Correlation
    # ======================================================
    axs[1, 0].plot(
        results_df["L"],
        results_df["correlation"],
        "-o",
        linewidth=2,
        markersize=6,
    )

    axs[1, 0].set_title(r"(c) Correlation between $\hat{\sigma}(x)$ and Residual")
    axs[1, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 0].set_ylabel("Correlation")
    axs[1, 0].grid(True)

    # ======================================================
    # (d) RMSE
    # ======================================================
    axs[1, 1].plot(
        results_df["L"],
        results_df["rmse"],
        "-o",
        linewidth=2,
        markersize=6,
    )

    axs[1, 1].set_title("(d) RMSE")
    axs[1, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 1].set_ylabel("RMSE")
    axs[1, 1].grid(True)

    # ======================================================
    # Overall Figure
    # ======================================================
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_lambda_selection_comparison(
    rmse_results_df,
    conformal_results_df,
    output_path=f"{PLOTS_DIR}/degree_robustness_lambda_comparison_summary.png",
):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(
        rmse_results_df["L"],
        rmse_results_df["adaptive_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[0, 0].plot(
        conformal_results_df["L"],
        conformal_results_df["adaptive_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[0, 0].axhline(
        y=0.90,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Target (90%)",
    )
    axs[0, 0].set_title("(a) Adaptive Coverage")
    axs[0, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 0].set_ylabel("Coverage")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(
        rmse_results_df["L"],
        rmse_results_df["adaptive_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[0, 1].plot(
        conformal_results_df["L"],
        conformal_results_df["adaptive_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[0, 1].set_title("(b) Adaptive Interval Width")
    axs[0, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 1].set_ylabel("Interval Width")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(
        rmse_results_df["L"],
        rmse_results_df["rmse"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[1, 0].plot(
        conformal_results_df["L"],
        conformal_results_df["rmse"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[1, 0].set_title("(c) Test RMSE")
    axs[1, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 0].set_ylabel("RMSE")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].semilogy(
        rmse_results_df["L"],
        rmse_results_df["best_lambda"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[1, 1].semilogy(
        conformal_results_df["L"],
        conformal_results_df["best_lambda"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[1, 1].set_title("(d) Selected Lambda")
    axs[1, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 1].set_ylabel("Lambda")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    fig.suptitle(
        "Comparison of RMSE-Selected and Conformal-Selected Lambda",
        fontsize=16,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
