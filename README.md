# Scalar Field Reconstruction Code

This repository contains the final code used for the dissertation experiments on spherical harmonic scalar-field reconstruction and adaptive geodesic conformal uncertainty quantification.

## Results at a glance

Headline results from the dissertation presentation (July 2003 data, 3,510 cities, 90% target coverage):

| Difficulty tertile | Standard conformal | Adaptive geodesic (proposed) | Adaptive interval width |
| --- | --- | --- | --- |
| Low difficulty | 100.0% | 91.5% | 1.48 °C |
| Medium difficulty | 95.7% | 94.9% | 3.89 °C |
| High difficulty | 76.7% | 94.8% | 8.46 °C |

* Marginal coverage looks similar (standard 90.9%, adaptive 93.7%) and mean interval widths are nearly identical (4.65 °C vs 4.59 °C), which hides the failure of the standard method in hard regions.
* The local difficulty estimate correlates with the true absolute residual at r = 0.634.
* The results hold from spherical harmonic degree L = 8 to L = 30 (correlation 0.61 to 0.68), with two different λ-selection criteria.
* 17 of the 20 highest-uncertainty cities are in the high-altitude Andes (Bolivia, Peru, northern Chile).

## Presentation

The dissertation presentation (10 to 12 minutes) is included in this repository:

* [`docs/Full_Presentation_FINAL_2.pptx`](docs/Full_Presentation_FINAL_2.pptx): the slides (PowerPoint).
* [`docs/Full_Presentation_FINAL_2.pdf`](docs/Full_Presentation_FINAL_2.pdf): the same slides as a PDF, which can be read directly on GitHub.

Structure: slides 1 to 16 are the main talk (background and methodology, contribution, results, takeaways). Slides 17 to 20 are backup slides for questions (hyperparameter selection, pipeline overview, discussion, limitations).

## Dissertation

The full dissertation, "Adaptive Geodesic Conformal Prediction for Uncertainty-Quantified Spherical Harmonic Reconstruction of Climate Fields" (65 pages), is in [`docs/dissertation/thesis.pdf`](docs/dissertation/thesis.pdf). The LaTeX source and figures are in the same folder.

## Prior Work

The spherical harmonic scalar-field reconstruction builds on the work of **Will Thomas (2025)**. The reconstruction framework and results from this work provide the starting point for the uncertainty quantification developed in this repository.

Will Thomas's dissertation code is available here:

https://github.com/Wthomas443/Dissertation

This repository extends the scalar-field reconstruction with an adaptive geodesic conformal prediction framework for uncertainty quantification.

## Files

* `scalar_field_reconstruction.py`: main training, reconstruction, conformal prediction, robustness, and plotting workflow (now a short entry point: the workflow code lives in the `scalar_field/` package, see Project Structure below).
* `check.py`: supporting check script for selected city/data diagnostics.
* `requirements.txt`: Python packages required to run the scripts.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── check.py                          # supporting check script
├── scalar_field_reconstruction.py    # entry point: python scalar_field_reconstruction.py
├── scalar_field/                     # the code, one module per step
│   ├── __main__.py                   # entry point: python -m scalar_field
│   ├── cli.py                        # menu and main()
│   ├── workflow.py                   # train_model, reconstruct_field
│   ├── config.py                     # file names, output folders and constants
│   ├── data.py                       # preprocessing and train / calibration / test splits
│   ├── spherical_harmonics.py        # spherical harmonic basis and regressor
│   ├── geometry.py                   # geodesic distance, local difficulty estimate, sphere points
│   ├── cross_validation.py           # cross-validation residuals and grid search
│   ├── conformal_evaluation.py       # coverage evaluation: standard vs adaptive
│   ├── robustness.py                 # degree-robustness experiment
│   ├── plots_conformal.py            # standard vs adaptive figures
│   ├── plots_robustness.py           # degree-robustness and lambda-selection figures
│   ├── plots_map.py                  # opened sphere map
│   └── storage.py                    # saving and loading models, output folders
└── docs/
    ├── Full_Presentation_FINAL_2.pptx
    ├── Full_Presentation_FINAL_2.pdf
    └── dissertation/                 # full dissertation: thesis.pdf, LaTeX source and figures
```

The code was reorganised from a single long script into modules so that each step can be read on its own. The function bodies, comments and printed messages are unchanged.

## Data

The raw temperature dataset is not included because it is large. Place the dataset in the repository root with this exact name:

```text
GlobalLandTemperaturesByCity.csv
```

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main script:

```bash
python scalar_field_reconstruction.py
```

The script writes generated models and figures to `models/` and `plots/`.

The same workflow can also be started with:

```bash
python -m scalar_field
```

## Main Experimental Configuration

The primary experiment uses the following configuration:

* Spherical harmonic degree: `L = 26`
* Regularisation parameter: `λ = 0.0001`
* Sobolev penalty power: `p = 2`
* Number of nearest neighbours for local difficulty estimation: `k = 20`
* Number of cross-validation folds: `5`
* Conformal miscoverage level: `α = 0.1`
* Target coverage: `90%`

### Workflow

The main experimental pipeline consists of:

1. Data preprocessing
2. Hyperparameter selection using cross-validation
3. Spherical harmonic reconstruction
4. Local difficulty estimation using geodesic k-nearest neighbours
5. Adaptive geodesic conformal calibration
6. Evaluation on a held-out test set
7. Degree-robustness analysis

### Where each step lives in the code

| Step | Module |
| --- | --- |
| Data preprocessing | `scalar_field/data.py` |
| Hyperparameter selection using cross-validation | `scalar_field/cross_validation.py` |
| Spherical harmonic reconstruction | `scalar_field/spherical_harmonics.py`, `scalar_field/workflow.py` |
| Local difficulty estimation using geodesic k-nearest neighbours | `scalar_field/geometry.py` |
| Adaptive geodesic conformal calibration | `scalar_field/workflow.py` |
| Evaluation on a held-out test set | `scalar_field/conformal_evaluation.py` |
| Degree-robustness analysis | `scalar_field/robustness.py` |
| Figures | `scalar_field/plots_*.py` |
