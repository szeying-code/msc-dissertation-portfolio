"""Entry point: run the main training, reconstruction and robustness workflow.

Usage:
    python scalar_field_reconstruction.py

The workflow code lives in the ``scalar_field`` package (see README.md).
"""

from scalar_field.cli import main

# Re-exported so that models saved by earlier versions of this script can still be loaded.
from scalar_field.spherical_harmonics import Spherical_Harmonics, SphericalHarmonicRegressor

__all__ = ["main", "Spherical_Harmonics", "SphericalHarmonicRegressor"]


if __name__ == "__main__":
    main()
