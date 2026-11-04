"""Cross-validation residuals and the hyperparameter grid search (lambda and gradient penalty)."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
import torch

from .config import PLOTS_DIR
from .data import split_data
from .spherical_harmonics import SphericalHarmonicRegressor, Spherical_Harmonics


def generate_cv_residuals(data, max_degree, reg_param, grad, num_folds=5):
    """Generate out-of-fold residuals for estimating local difficulty."""
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    harmonic_method = Spherical_Harmonics()
    residuals = []
    theta_vals = []
    phi_vals = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"CV Fold {fold + 1}/{num_folds}")
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        X_train = [train_data['Theta'].values, train_data['Phi'].values]
        y_train = train_data['AverageTemperature'].values
        X_val = [val_data['Theta'].values, val_data['Phi'].values]
        y_val = val_data['AverageTemperature'].values
        # Train model on this fold
        model = SphericalHarmonicRegressor(
            harmonic_method,
            max_degree=max_degree,
            reg_param=reg_param,
            grad=grad,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val).real  # Predict validation fold
        fold_residuals = np.abs(y_val - preds)  # Absolute validation residuals
        residuals.extend(fold_residuals)
        theta_vals.extend(val_data['Theta'].values)
        phi_vals.extend(val_data['Phi'].values)
    return np.array(theta_vals), np.array(phi_vals), np.array(residuals)


# Search over lambda and gradient penalty to find the best model setting
def grid_search(data, max_degree, lambda_values, grad_values, base_model, num_folds):
    """Search over lambda and gradient penalty to find the best model setting."""
    harmonic_method = Spherical_Harmonics()
    train_data_list, validation_data_list = split_data(data, num_folds)

    print("Precomputing matrices...")

    A_training_matrices = []
    A_test_matrices = []
    train_values_list = []
    test_values_list = []

    for fold in range(num_folds):
        print(f"  Fold {fold + 1}/{num_folds}")
        train_data = train_data_list[fold]
        test_data = validation_data_list[fold]

        theta_train = train_data["Theta"].values
        phi_train = train_data["Phi"].values
        theta_test = test_data["Theta"].values
        phi_test = test_data["Phi"].values

        train_values = train_data["AverageTemperature"].values
        test_values = test_data["AverageTemperature"].values

        A_training_matrices.append(
            harmonic_method.design_matrix_gpu(theta_train, phi_train, max_degree)
        )
        A_test_matrices.append(
            harmonic_method.design_matrix_gpu(theta_test, phi_test, max_degree)
        )
        train_values_list.append(train_values)
        test_values_list.append(test_values)

    L_cache = {
        grad: harmonic_method.construct_L_cached(max_degree, grad)
        for grad in grad_values
    }
    device = harmonic_method.device
    fold_errors = []

    print("Testing parameter combinations...")

    grad_size = len(grad_values)
    lambda_size = len(lambda_values)
    total_combinations = grad_size * lambda_size * num_folds

    for fold in range(num_folds):
        A_train = torch.tensor(
            A_training_matrices[fold],
            dtype=torch.complex128,
            device=device,
        )
        A_test = torch.tensor(
            A_test_matrices[fold],
            dtype=torch.complex128,
            device=device,
        )
        train_values = torch.tensor(
            train_values_list[fold],
            dtype=torch.complex128,
            device=device,
        )
        test_values = torch.tensor(
            test_values_list[fold],
            dtype=torch.complex128,
            device=device,
        )

        ATA_train = A_train.conj().T @ A_train
        ATy = A_train.conj().T @ train_values
        comb_errors = np.zeros((grad_size, lambda_size))

        for j, reg_param in enumerate(lambda_values):
            for i, grad in enumerate(grad_values):
                if (j * grad_size + i) % 10 == 0:
                    completed = fold * lambda_size * grad_size + j * grad_size + i + 1
                    percentage = completed / total_combinations * 100
                    print(f"{percentage:.1f}%")

                L = torch.tensor(L_cache[grad], dtype=torch.complex128, device=device)
                coefficients = harmonic_method.gpu_solve(ATA_train, ATy, reg_param, L)
                predictions = A_test @ coefficients
                comb_errors[i, j] = torch.linalg.norm(predictions - test_values).item()

        fold_errors.append(comb_errors)

    mean_errors = np.mean(fold_errors, axis=0)
    print("\n===== Cross-validation errors =====")
    for i, grad in enumerate(grad_values):
        print(f"\nGradient = {grad}")
        for j, lam in enumerate(lambda_values):
            print(f"lambda = {lam:.2e}, error = {mean_errors[i, j]:.6f}")

    min_error = np.min(mean_errors)
    best_grad_index, best_lambda_index = np.unravel_index(
        np.argmin(mean_errors),
        mean_errors.shape,
    )
    log_errors = np.log10(mean_errors)
    log_min_error = np.log10(min_error)
    log_max_error = np.log10(np.max(mean_errors))
    norm_log_errors = (log_errors - log_min_error) / (log_max_error - log_min_error)

    ax = sns.heatmap(
        norm_log_errors,
        xticklabels=lambda_values,
        yticklabels=grad_values,
        cmap="viridis",
        fmt=".2f",
        cbar_kws={"label": "Error (normalized, log scale)"},
    )
    ax.set(xlabel="Regularisation Parameter", ylabel="k values")
    ax.set_title(f"Max Degree = {max_degree}")
    ax.add_patch(
        plt.Rectangle(
            (best_lambda_index, best_grad_index),
            1,
            1,
            fill=False,
            edgecolor="red",
            lw=3,
        )
    )

    plt.savefig(f"{PLOTS_DIR}/parameter_optimization_heatmap_L{max_degree}.png")
    plt.close()

    for grad_index, grad in enumerate(grad_values):
        plt.figure(figsize=(10, 6))
        lambda_errors = []

        for fold in range(num_folds):
            fold_lambda_errors = fold_errors[fold][grad_index, :]
            lambda_errors.append(fold_lambda_errors)
            plt.loglog(lambda_values, fold_lambda_errors, label=f"Fold {fold + 1}")

        plt.loglog(
            lambda_values,
            np.mean(lambda_errors, axis=0),
            label="Mean Error",
            linewidth=2,
            color="black",
        )
        plt.xlabel("Regularisation Parameter (log scale)")
        plt.ylabel("Error (log scale)")
        plt.title(f"Gradient penalised = {grad}")
        plt.legend()
        plt.savefig(f"{PLOTS_DIR}/parameter_optimization_L{max_degree}_grad_{grad}.png")
        plt.close()

    best_lambda = lambda_values[best_lambda_index]
    best_grad = grad_values[best_grad_index]
    print("\n--- GRID SEARCH RESULT ---")
    print(f"Best lambda: {best_lambda:.2e}")
    print(f"Best grad: {best_grad}")
    print(f"Minimum CV error: {min_error:.4f}")

    model = base_model
    model.set_params(max_degree=max_degree, reg_param=best_lambda, grad=best_grad)
    return model, best_lambda, best_grad
