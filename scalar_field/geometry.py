"""Geodesic distances, local difficulty estimation and points on the sphere."""

import numpy as np


def geodesic_distance(theta1, phi1, theta2, phi2):
    cos_angle = (
        np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
        + np.cos(theta1) * np.cos(theta2)
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
    return np.arccos(cos_angle)


def estimate_sigma_geodesic(
    theta_query,
    phi_query,
    theta_ref,
    phi_ref,
    residuals,
    k=20,
    eps=1e-6,
):
    sigma_values = []
    for tq, pq in zip(theta_query, phi_query):
        d = geodesic_distance(tq, pq, theta_ref, phi_ref)
        idx = np.argsort(d)[:k]  # Indices of k nearest neighbours
        sigma = np.mean(residuals[idx])  # Average neighbour residuals
        sigma = max(sigma, eps)  # Avoid division by zero
        sigma_values.append(sigma)
    return np.array(sigma_values)


def spherical_points_to_cartesian(theta, phi, radius=1.0):
    """Convert arrays of spherical coordinates to Cartesian points."""
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z])


def geodesic_circle_points(theta, phi, angular_radius, n_points=160):
    """Create a geodesic circle on the unit sphere around one location."""
    center = spherical_points_to_cartesian(
        np.array([theta]),
        np.array([phi]),
        radius=1.0,
    )[0]

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(center, reference)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])

    u = np.cross(center, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(center, u)

    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = (
        np.cos(angular_radius) * center[None, :]
        + np.sin(angular_radius)
        * (
            np.cos(angles)[:, None] * u[None, :]
            + np.sin(angles)[:, None] * v[None, :]
        )
    )
    return circle
