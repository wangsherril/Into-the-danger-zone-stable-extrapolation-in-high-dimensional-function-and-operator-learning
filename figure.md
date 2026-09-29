# Paper figure reproduction guide

This guide maps each main-paper panel to its experiment, active configuration, and generated file. Commands assume the current directory is the repository root unless stated otherwise.

## Figure index

| Figure/panel | Method | Experiment | Script(s) | Expected generated file |
|---|---|---|---|---|
| Fig. 1 (left) | DNN | Wing-weight function; OOD width controlled by theta | `DNN/scripts/NN4virtual_library_thetacontrol.py` | `DNN/fig/DNN/id_ood_mse_depth_15_wingweight_scaled_10d_theta.png` |
| Fig. 1 (right) | FNO | Navier–Stokes; ID alpha 2.5 and OOD alpha 2.0/2.25 | `FNO/scripts/FNO4NS_OOD.py` | `FNO/fig/NS/ns_fno3d_mse_errors.png` |
| Fig. 2 (left) | ALS | Constant side lengths (`hypercube`), dimension 8 |   `'ALS/scripts/ALS_data_generation_2isq.m', then ’ALS/scripts/ALS_plot_2isq.py'`| `ALS/figs/ALS_nonintegrable_singularity_delta_2ia_d8_a2_hypercube.png` |
| Fig. 2 (right) | ALS | Constant side lengths (`hypercube`), dimension 32 | same as above | `ALS/figs/ALS_nonintegrable_singularity_delta_2ia_d32_a2_hypercube.png` |
| Fig. 3 (left) | ALS | Varying side lengths (`unbounded`), dimension 8 | same as above | `ALS/figs/ALS_nonintegrable_singularity_delta_2ia_d8_a2_unbounded.png` |
| Fig. 3 (right) | ALS | Varying side lengths (`unbounded`), dimension 32 | same as above | `ALS/figs/ALS_nonintegrable_singularity_delta_2ia_d32_a2_unbounded.png` |
| Fig. 4 (left) | DNN | Circuit function; common OOD box scale | `DNN/scripts/NN4virtual_library_wcontrol.py` | `DNN/fig/DNN/id_ood_rel_l2_depth_15_circuit_scaled_6d.png` |
| Fig. 4 (right) | DNN | Piston function; common OOD box scale | `DNN/scripts/NN4virtual_library_wcontrol.py` | `DNN/fig/DNN/id_ood_rel_l2_depth_15_piston_scaled_7d.png` |
| Fig. 5 (left) | FNO | Darcy flow, LN training to LN OOD tests | `FNO/scripts/FNO4Darcy_LC_PC_cookie_withswitch.py` | `FNO/fig/Darcy/LN_to_LN_relative_l2_final_error_vs_training_size_IDalpha_2.png` |
| Fig. 5 (middle) | FNO | Darcy flow, LN training to PC OOD tests | same as above | `FNO/fig/Darcy/LN_to_PC_relative_l2_final_error_vs_training_size_IDalpha_2.png` |
| Fig. 5 (right) | FNO | Darcy flow, LN training to COOKIE OOD tests | same as above | `FNO/fig/Darcy/LN_to_COOKIE_relative_l2_final_error_vs_training_size_IDalpha_2.png` |


## Figure 1

### Left: DNN wing-weight experiment
For a fresh run, first set `generate_new_data = True` and `train_models = True`. To use existing data and checkpoints, leave both switches `False`.
The script also writes the training-history plot and an `.npz` result file under `DNN/fig/DNN/` and `DNN/results/DNN/`.

### Right: FNO Navier–Stokes experiment

The paper configuration uses a 64 × 64 spatial grid, training alpha 2.5, test alphas `[2.0, 2.25, 2.5]`, training sizes `[150, 300, 600, 1200, 2400, 4800]`, five trials, and MSE evaluation.

For plotting from saved results, retain:

```python
PLOT_ONLY = True
EVAL_METRICS = ["mse"]
```

This mode requires `FNO/results/NS/ns_fno3d_mse_results.npz`. For a complete data-generation/training run, follow the switch sequence in [readme.md](readme.md#fno-navierstokes).

## Figures 2 and 3: ALS

Run the MATLAB experiment from `ALS/scripts/` because it uses relative paths for `../utils` and `../data`:

For each unbounded setup, the script uses `b = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]` and side lengths `w_k = k^b`. For each hypercube setup, it uses constant side lengths `omega = [1, 1.1, 1.2, 1.3, 1.6, 1.8]`. The MATLAB stage writes one `.mat` file per dimension, geometry, and OOD parameter to `ALS/data/`. Then generate all four panels and the fitted-rate tables with:

```bash
python3 ALS/scripts/ALS_data_generation_2isq.py
```


## Figure 4: DNN virtual-library functions

Use `NN4virtual_library_wcontrol.py` twice. For a fresh run, also set `generate_new_data = True` and `train_models = True`; use `False` only when the corresponding cached dataset and checkpoints already exist.

For the left panel:

```python
target_name = "circuit_scaled_6d"
```

For the right panel:

```python
target_name = "piston_scaled_7d"
```

The remaining paper settings are:

```python
sample_sizes = [10 * 2**k for k in range(8)]
depths = [15]
width_multiplier = 10
epochs = 2000
training_loss = "rel_l2"
evaluation_metric = "rel_l2"
trials = 30
test_size = 10000
ood_scales = [1.1, 1.2, 1.4]
seed = 2026
```

## Figure 5: FNO Darcy flow

Use `FNO4Darcy_LC_PC_cookie_withswitch.py` three times. Keep `train_distribution = "LN"` and `id_test_distribution = "LN"`, and change only `ood_test_distribution` between runs:

```python
ood_test_distribution = "LN"
ood_test_distribution = "PC"
ood_test_distribution = "COOKIE"
```

The shared paper configuration is:

```python
resolution = 64
id_alpha = 2
ood_alphas = [1.75, 1.5, 1.25, 1.0]
cookie_radii = [0.14]
training_sizes = [10 * 2**k for k in range(8)]
test_size = 200
trials = 5
epochs = 100
evaluation_metric = "relative_l2"
seed = 2026
```

With `generate_new_data = True` and `train_new_models = True`, each run creates data, trains models, evaluates ID/OOD error, and writes figures and results. With both switches `False`, the matching cached datasets and checkpoints must already exist under `FNO/results/Darcy/`.


