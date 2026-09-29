# Into the Danger Zone: Stable Extrapolation in High-Dimensional Function and Operator Learning

Code and reproducibility materials for **Into the Danger Zone: Stable Extrapolation in High-Dimensional Function and Operator Learning**.

- **Authors:** Ben Adcock, Simone Brugiapaglia, Xuemeng Wang

## Overview

This repository contains the experiments for three approximation methods:

- **ALS:** adaptive least-squares approximation of a high-dimensional function;
- **DNN:** fully connected neural networks for functions from a virtual library; and
- **FNO:** Fourier neural operators for Darcy flow and the two-dimensional incompressible Navier–Stokes equations in vorticity form.

The experiments compare in-distribution (ID) and out-of-distribution (OOD) errors as the training-set size increases. See [figure.md](figure.md) for the mapping between paper panels, scripts, configurations, and output files.

## Repository structure

```text
.
├── readme.md
├── figure.md
├── ALS/
│   ├── scripts/                # MATLAB experiment and Python plotting script
│   ├── utils/                  # ALS routines and bundled Chebfun code
│   ├── data/                   # Generated .mat files (created at runtime)
│   ├── figs/                   # Generated figures (created at runtime)
│   └── tables/                 # Fitted-rate CSV files (created at runtime)
├── DNN/
│   ├── scripts/                # Virtual-library DNN experiments
│   ├── utils/                  # Target functions
│   ├── data/DNN/               # Generated or cached datasets
│   ├── results/DNN/            # Model checkpoints and numerical results
│   └── fig/DNN/                # Generated figures
└── FNO/
    ├── scripts/                # Darcy and Navier–Stokes experiments
    ├── utils/                  # random fields and loss utilities
    ├── data/                   # Navier–Stokes datasets
    ├── models/NS/              # Navier–Stokes checkpoints
    ├── pred/NS/                # Saved Navier–Stokes predictions
    ├── results/                # Darcy and Navier–Stokes numerical results
    └── fig/                    # Darcy and Navier–Stokes figures
```

## Software requirements

The Python scripts require:

- Python 3.9 or newer;
- NumPy, SciPy, Matplotlib, PyTorch, `neuraloperator`, and h5py; and
- a LaTeX installation available to Matplotlib for scripts that set `text.usetex = True`.

The ALS data-generation script additionally requires MATLAB. It uses `parfor`, so the Parallel Computing Toolbox is recommended. The required Chebfun sources are included under `ALS/utils/`.

## Running the experiments

### ALS

The MATLAB script performs the experiments and writes `.mat` files. The Python script reads those files, fits the algebraic rates, and creates the final plots and CSV tables.

The default MATLAB configuration runs four cases: dimensions 8 and 32, each with constant-side-length (`hypercube`) and varying-side-length (`unbounded`) OOD domains. It uses 30 trials, a maximum sample budget of 15,000, and an error grid of size 10,000. Outputs are written to `ALS/data/`, `ALS/figs/`, and `ALS/tables/`.

### DNN

- `NN4virtual_library_thetacontrol.py` controls anisotropic OOD widths through `theta_values`; its paper configuration uses `wingweight_scaled_10d`.
- `NN4virtual_library_wcontrol.py` controls a common OOD box scale through `ood_scales`; run it separately with `target_name = "circuit_scaled_6d"` and `target_name = "piston_scaled_7d"` for the two Figure 4 panels.

For a run from scratch, set `generate_new_data = True` and `train_models = True`. The checked-in defaults are configured to reuse saved datasets and checkpoints (`False` for both switches), so those artifacts must already exist. Figures, results, and checkpoints are stored under `DNN/fig/DNN/` and `DNN/results/DNN/`.

### FNO: Darcy flow
 Set `generate_new_data = False` and `train_new_models = False` to reuse the matching cached data and checkpoints. Outputs are written to `FNO/fig/Darcy/` and `FNO/results/Darcy/`.

`FNO4Darcy_input_output_pairs.py` is a plotting helper intended to be called with the Darcy experiment objects; it is not an independent training entry point.

### FNO: Navier–Stokes

```bash
python3 FNO/scripts/FNO4NS_OOD.py
```


1. set `GENERATE_DATA = True`, `TRAIN_MODEL = False`, `EVALUATE_MODEL = False`, and `PLOT_ONLY = False` to generate only the datasets;
2. set `GENERATE_DATA = False`, `TRAIN_MODEL = True`, `EVALUATE_MODEL = True`, and `PLOT_ONLY = False` to train and evaluate; and
3. after results have been saved, set `PLOT_ONLY = True` for inexpensive replotting.

The default experiment uses a 64 × 64 grid, training sizes 150–4800, five trials, 200 epochs, and seed 2026. Data, checkpoints, predictions, results, and figures are stored in their corresponding `FNO/` subdirectories.

## Reproducibility notes

- The Python experiments use seed 2026 by default. The ALS MATLAB script currently does not set an explicit random seed; add and report one if bitwise repeatability is required.
- Full reproduction is computationally expensive. In particular, the ALS experiment uses 30 trials, the DNN experiments use 30 trials and 2,000 epochs, and the PDE datasets require numerical solves.
- CUDA is used when available. Several scripts also support Apple MPS and otherwise fall back to CPU.
- Configuration is stored directly in the scripts. Record any changes to target functions, distributions, sample sizes, trials, solver settings, and run switches when reporting new results.

## Paper figure guide

See [figure.md](figure.md) for panel-level commands and exact generated filenames.


