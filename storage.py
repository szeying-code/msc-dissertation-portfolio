"""Saving and loading trained models and preparing the output folders."""

import pickle
import os

from .config import MODELS_DIR, PLOTS_DIR


def save_model(model, name):
    """Save the trained model so reconstruction can run later."""
    with open(f"{MODELS_DIR}/{name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_qhat(name="temp_qhat"):
    """Load the calibrated conformal threshold qhat."""
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        qhat = pickle.load(f)
    return qhat


def load_model(name):
    """Load the trained model for reconstruction mode."""
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def ensure_output_dirs():
    """Create output directories used by training and plotting."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
