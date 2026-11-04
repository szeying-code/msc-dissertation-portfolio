"""Spherical harmonic basis and the scikit-learn style regressor used for field reconstruction."""

import numpy as np
from scipy.special import sph_harm_y
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
import torch
from scipy import special

from .config import EARTH_RADIUS_METRES, PLOTS_DIR


class Spherical_Harmonics:
    """Mathematical tools for spherical harmonic modelling."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L_cache = {}

    def spherical_harmonics_basis(self, l, m, theta, phi):
        """Return Y_l^m(theta, phi)."""
        return sph_harm_y(l, m, theta, phi)
    
    def convert_to_cartesian(self, theta, phi, r):
        """Convert spherical coordinates to Cartesian coordinates."""
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
    def associated_legendre(self, l, m, x):
        """Compute associated Legendre values using SciPy, then return to torch."""
        x_np = x.cpu().numpy()
        P_lm = special.lpmv(m, l, x_np)
        return torch.tensor(P_lm, dtype=torch.float64, device=self.device)
    
    def design_matrix(self, theta, phi, max_degree):
        """Build the regression design matrix."""
        A = np.zeros((len(theta), (max_degree + 1)**2), dtype=np.complex128)
        idx = 0
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                A[:, idx] = self.spherical_harmonics_basis(l, m, theta, phi)
                idx += 1
        return A

    def solve_lsq(self, max_degree, data, A, e, grad):
        """Solve the regularised least-squares system."""
        if e is None:
            e = 0.0
        L = self.construct_L(max_degree, grad)
        system = A.conj().T @ A + e * L
        rhs = A.conj().T @ data
        coefficients = np.linalg.solve(system, rhs)
        return coefficients  # Estimated spherical harmonic coefficients
    
    def design_matrix_gpu(self, theta, phi, max_degree):
        """GPU-friendly design matrix used during expensive grid-search steps."""
        num_points = len(theta)
        n_coeffs = (max_degree + 1)**2

        theta_np = theta.values if hasattr(theta, 'values') else np.array(theta)
        phi_np = phi.values if hasattr(phi, 'values') else np.array(phi)

        theta_torch = torch.tensor(theta_np, dtype=torch.float64, device=self.device)
        phi_torch = torch.tensor(phi_np, dtype=torch.float64, device=self.device)

        cos_theta = torch.cos(theta_torch)

        A = torch.zeros(
            (num_points, n_coeffs),
            dtype=torch.complex128,
            device=self.device,
        )

        idx = 0
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                P_lm = self.associated_legendre(l, m, cos_theta)
                
                norm = torch.sqrt(torch.tensor(
                    (2*l + 1) / (4 * np.pi) *
                    special.factorial(l - m) / special.factorial(l + m),
                    dtype=torch.float64, device=self.device
                ))

                phase = torch.exp(1j * m * phi_torch)

                A[:, idx] = norm * P_lm * phase
                idx += 1
        return A.cpu().numpy()
    
    def construct_L(self, max_degree, power):
        """Build the diagonal penalty matrix for higher-degree harmonics."""
        L = np.zeros(((max_degree + 1)**2, (max_degree + 1)**2), dtype=np.complex128)
        idx = 0
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                L[idx][idx] = (l * (l + 1))**power
                idx += 1
        return L
    
    def construct_L_cached(self, max_degree, power):
        """Reuse penalty matrices during grid search."""
        if (max_degree, power) in self.L_cache:
            return self.L_cache[(max_degree, power)]
        
        L = self.construct_L(max_degree, power)
        self.L_cache[(max_degree, power)] = L
        return L
    
    def gpu_solve(self, AtA, rhs, e, L):
        """Solve the regularised linear system on GPU."""
        return torch.linalg.solve(AtA + e * L, rhs)
    
    def plot_prediction_and_uncertainty(
        self,
        theta_grid,
        phi_grid,
        prediction,
        uncertainty,
        theta_cities,
        phi_cities,
        city_temp,
        r=EARTH_RADIUS_METRES,
    ):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Uncertainty surface
        xh, yh, zh = self.convert_to_cartesian(theta_grid, phi_grid, r)
        unc_norm = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()))  # Normalise uncertainty
        colors = plt.cm.Reds(unc_norm)
        # Use stronger transparency where uncertainty is larger
        colors[..., -1] = unc_norm * 0.8
        ax.plot_surface(xh, yh, zh, facecolors=colors, rstride=1, cstride=1, shade=False)
        # City temperature points
        xc, yc, zc = self.convert_to_cartesian(theta_cities, phi_cities, r)
        scatter = ax.scatter(xc, yc, zc, c=city_temp, cmap='viridis', s=10)

        # Plot styling
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=-60)
        ax.grid(True)  # Keep the 3D grid readable

        # Colourbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Temperature')
        ax.set_title("Temperature Prediction with Adaptive Geodesic Conformal Uncertainty Overlay")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/prediction_uncertainty_overlay.png", dpi=300, bbox_inches='tight')
        plt.show()


class SphericalHarmonicRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, harmonic_method, max_degree=None, reg_param=None, grad=None):
        self.harmonic_method = harmonic_method
        self.max_degree = max_degree
        self.reg_param = reg_param
        self.grad = grad
        self.coefficients = None

    def set_params(self, max_degree=None, reg_param=None, grad=None):
        """Set model parameters chosen by grid search."""
        if max_degree is not None:
            self.max_degree = max_degree
        if reg_param is not None:
            self.reg_param = reg_param
        if grad is not None:
            self.grad = grad
        return self

    def fit(self, X, y):
        """Fit the spherical harmonic coefficients."""
        A = self.harmonic_method.design_matrix(X[0], X[1], self.max_degree)
        self.coefficients = self.harmonic_method.solve_lsq(self.max_degree, y, A, self.reg_param, self.grad)
        return self

    def predict(self, X):
        """Predict temperatures at spherical locations."""
        A = self.harmonic_method.design_matrix(X[0], X[1], self.max_degree)
        return A @ self.coefficients
