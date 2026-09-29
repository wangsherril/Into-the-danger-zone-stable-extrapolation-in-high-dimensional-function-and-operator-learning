import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.legend_handler import HandlerTuple
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 1. Paths
# =============================================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2.2,
})

if "__file__" in globals():
    SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = (
        SCRIPT_PATH.parents[1]
        if SCRIPT_PATH.parent.name == "scripts"
        else SCRIPT_PATH.parent
    )
else:
    PROJECT_ROOT = Path.cwd().resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIGURE_DIR = PROJECT_ROOT / "fig" / "Darcy"
RESULTS_DIR = PROJECT_ROOT / "results" / "Darcy"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from neuralop.models import FNO
from utils.darcy_utils import GRF, solve_gwf
from utils.utilities import LpLoss



# =============================================================================
# 2. Parameters and setup
# =============================================================================

# Darcy data settings.
resolution = 64
id_alpha = 2
ood_alphas = [1.75, 1.5, 1.25, 1.0]
# ood_alphas = [1, 0.75, 0.5, 0.25]
cookie_radii = [0.14]
tau = 3.0

# Train on the original log-normal distribution and use the same distribution
# for the fixed ID test set. Choose "LN", "PC", or "COOKIE" for OOD tests.
train_distribution = "LN"
id_test_distribution = "LN"
ood_test_distribution = "PC"

# Dataset settings.
training_sizes = [10 * 2**k for k in range(8)]
test_size = 200
trials = 5
training_batch_size = 32
test_batch_size = 32

# Algebraic fitting.
plot_algebraic_fit = True
fit_training_sizes = training_sizes

# FNO settings.
n_modes = (12, 12)
hidden_channels = 32
projection_channel_ratio = 2

# Optimization settings.
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4
scheduler_step = 100
scheduler_gamma = 0.5

# Evaluation settings. Choose "relative_l2" or "mse".
evaluation_metric = "relative_l2"

# Output settings.
save_prediction_plots = True
save_convergence_plot = True
prediction_trial = 0
history_trial = 0
number_of_examples = 3
show_figures = False
verbose = True

# Run switches.
generate_new_data = True
train_new_models = True

# Reproducibility.
seed = 2026

experiment_name = f"{train_distribution}_to_{ood_test_distribution}"
result_name = f"{experiment_name}_{evaluation_metric}"
DATA_DIR = RESULTS_DIR / f"{experiment_name}_data"
MODEL_DIR = RESULTS_DIR / f"{experiment_name}_models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. Helper functions
# =============================================================================

def log(message=""):
    """Print only when verbose=True."""
    if verbose:
        print(message, flush=True)


def choose_device():
    """Use CUDA, Apple MPS, or CPU, in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(value):
    """Make NumPy and PyTorch reproducible."""
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def number_tag(value):
    """Convert a number to a filename- and array-key-safe string."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def metric_log_name():
    """Return the selected evaluation metric's short display name."""
    return "relative L2" if evaluation_metric == "relative_l2" else "MSE"


def metric_axis_label():
    """Return the selected evaluation metric's plot label."""
    if evaluation_metric == "relative_l2":
        return r"Relative $L^2$ error"
    return "Mean squared error (MSE)"


def make_ood_cases():
    """Return OOD cases indexed by alpha or cookie radius."""
    if ood_test_distribution == "COOKIE":
        return [
            {
                "value": radius,
                "alpha": id_alpha,
                "radius": radius,
                "label": rf"OOD, COOKIE $r={radius:g}$",
                "log_label": f"COOKIE radius={radius:g}",
                "key": f"cookie_radius_{number_tag(radius)}",
            }
            for radius in cookie_radii
        ]

    return [
        {
            "value": alpha,
            "alpha": alpha,
            "radius": None,
            "label": rf"OOD, {ood_test_distribution} $\alpha={alpha:g}$",
            "log_label": f"{ood_test_distribution} alpha={alpha:g}",
            "key": f"{ood_test_distribution.lower()}_alpha_{number_tag(alpha)}",
        }
        for alpha in ood_alphas
    ]


def validate_settings():
    """Stop early when a setting is invalid."""
    valid_distributions = {"LN", "PC", "COOKIE"}

    if train_distribution not in {"LN", "PC"}:
        raise ValueError("train_distribution must be 'LN' or 'PC'")
    if id_test_distribution not in {"LN", "PC"}:
        raise ValueError("id_test_distribution must be 'LN' or 'PC'")
    if ood_test_distribution not in valid_distributions:
        raise ValueError("ood_test_distribution must be 'LN', 'PC', or 'COOKIE'")
    if id_test_distribution != train_distribution:
        raise ValueError(
            "ID test distribution must match the training distribution"
        )
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if any(mode <= 0 or mode > resolution // 2 for mode in n_modes):
        raise ValueError(
            f"n_modes={n_modes} is incompatible with resolution={resolution}"
        )
    if not training_sizes or any(size <= 0 for size in training_sizes):
        raise ValueError("training_sizes must contain positive numbers")
    if test_size <= 0 or trials <= 0:
        raise ValueError("test_size and trials must be positive")
    if training_batch_size <= 0 or test_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if evaluation_metric not in {"relative_l2", "mse"}:
        raise ValueError("evaluation_metric must be 'relative_l2' or 'mse'")
    if number_of_examples <= 0 or number_of_examples > test_size:
        raise ValueError("number_of_examples must be between 1 and test_size")
    if not 0 <= prediction_trial < trials:
        raise ValueError("prediction_trial must be a valid trial index")
    if not 0 <= history_trial < trials:
        raise ValueError("history_trial must be a valid trial index")

    if ood_test_distribution == "COOKIE":
        if not cookie_radii or any(radius <= 0 for radius in cookie_radii):
            raise ValueError("cookie_radii must contain positive radii")
    elif not ood_alphas or any(alpha <= 0 for alpha in ood_alphas):
        raise ValueError("ood_alphas must contain positive values")

    if plot_algebraic_fit:
        if len(fit_training_sizes) < 2:
            raise ValueError("fit_training_sizes must contain at least two values")
        invalid_fit_sizes = [
            size for size in fit_training_sizes if size not in training_sizes
        ]
        if invalid_fit_sizes:
            raise ValueError(
                "Every fit_training_sizes value must also appear in "
                f"training_sizes. Invalid values: {invalid_fit_sizes}"
            )


# =============================================================================
# 4. Data generation
# =============================================================================

COOKIE_CENTERS = [
    (0.2, 0.2),
    (0.2, 0.5),
    (0.2, 0.8),
    (0.5, 0.2),
    (0.5, 0.8),
    (0.8, 0.2),
    (0.8, 0.5),
    (0.8, 0.8),
]


def generate_cookie_field(resolution, radius, parameters=None):
    """Generate one cookie diffusion-coefficient field."""
    grid = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    x1, x2 = np.meshgrid(grid, grid, indexing="ij")

    if parameters is None:
        parameters = np.random.uniform(-1.0, 1.0, size=8).astype(np.float32)

    coefficient = np.ones((resolution, resolution), dtype=np.float32)

    for parameter, (center_x, center_y) in zip(parameters, COOKIE_CENTERS):
        circle = (
            (x1 - center_x) ** 2 + (x2 - center_y) ** 2 <= radius**2
        )
        coefficient[circle] = 0.625 - 0.375 * parameter

    return coefficient

def generate_one_sample(alpha, distribution, radius=None):
    """Generate one permeability field and its Darcy solution."""
    if distribution == "COOKIE":
        if radius is None:
            raise ValueError("radius is required for COOKIE samples")
        permeability = generate_cookie_field(resolution, radius)
    else:
        log_permeability = GRF(alpha, tau, resolution)

        if distribution == "LN":
            permeability = np.exp(log_permeability)
        elif distribution == "PC":
            permeability = np.where(log_permeability >= 0.0, 2, 0.5)
        else:
            raise ValueError("distribution must be 'LN', 'PC', or 'COOKIE'")

    forcing = np.ones((resolution, resolution), dtype=np.float32)
    solution = solve_gwf(permeability, forcing)
    return permeability.astype(np.float32), solution.astype(np.float32)


def generate_dataset(
    number_of_samples,
    alpha,
    label,
    distribution,
    radius=None,
):
    """Generate a complete Darcy dataset."""
    inputs = []
    outputs = []

    for index in range(number_of_samples):
        permeability, solution = generate_one_sample(
            alpha,
            distribution,
            radius=radius,
        )
        inputs.append(permeability)
        outputs.append(solution)

        completed = index + 1
        if completed == 1 or completed % 10 == 0 or completed == number_of_samples:
            parameter_text = (
                f"radius={radius:g}"
                if distribution == "COOKIE"
                else f"alpha={alpha:g}"
            )
            log(
                f"  [data] {label:<24} | samples "
                f"{completed:>4}/{number_of_samples:<4} | {parameter_text}"
            )

    return np.stack(inputs), np.stack(outputs)


def load_or_generate_dataset(
    cache_name,
    number_of_samples,
    alpha,
    label,
    distribution,
    radius=None,
):
    """Use the switch to generate new data or load saved data."""
    data_path = DATA_DIR / f"{cache_name}.npz"

    if generate_new_data:
        inputs, outputs = generate_dataset(
            number_of_samples,
            alpha,
            label,
            distribution,
            radius=radius,
        )
        np.savez_compressed(data_path, inputs=inputs, outputs=outputs)
        log(f"  [data] Saved: {data_path}")
    else:
        saved = np.load(data_path)
        inputs = saved["inputs"]
        outputs = saved["outputs"]
        saved.close()
        log(f"  [data] Loaded: {data_path}")

    return inputs, outputs


def make_loader(inputs, outputs, batch_size, shuffle):
    """Convert NumPy arrays into a PyTorch DataLoader."""
    x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(outputs, dtype=torch.float32).unsqueeze(1)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# =============================================================================
# 5. Model, training, and evaluation
# =============================================================================

def make_model(device):
    """Create the FNO model."""
    return FNO(
        n_modes=n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        projection_channel_ratio=projection_channel_ratio,
    ).to(device)


def train_model(model, loader, device, id_loader=None, record_history=False):
    """Train one FNO and optionally record train/ID errors per epoch."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma,
    )
    loss_fn = LpLoss(d=2, p=2, size_average=True)
    print_every = max(1, epochs // 10)
    history = {"train": [], "id_test": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        scheduler.step()
        epoch_number = epoch + 1
        mean_loss = total_loss / total_samples

        if record_history:
            if id_loader is None:
                raise ValueError("id_loader is required when recording history")
            history["train"].append(test_model(model, loader, device))
            history["id_test"].append(test_model(model, id_loader, device))

        if epoch_number == 1 or epoch_number % print_every == 0 or epoch_number == epochs:
            log(
                f"    epoch {epoch_number:>5}/{epochs:<5} "
                f"| relative L2={mean_loss:.4e} "
                f"| lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    return history


def test_model(model, loader, device):
    """Return the selected mean evaluation metric over all samples."""
    model.eval()
    total_error = 0.0
    total_samples = 0
    relative_l2 = LpLoss(d=2, p=2, size_average=True)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            if evaluation_metric == "relative_l2":
                error = relative_l2(prediction, y)
            else:
                error = torch.mean((prediction - y) ** 2)
            total_error += error.item() * x.size(0)
            total_samples += x.size(0)

    return total_error / total_samples


def fit_algebraic_rate(sample_sizes, mean_errors):
    """Fit error(m) = c * m^(-q) on selected training sizes."""
    sample_sizes = np.asarray(sample_sizes, dtype=float)
    mean_errors = np.asarray(mean_errors, dtype=float)
    fit_mask = np.isin(sample_sizes, np.asarray(fit_training_sizes, dtype=float))
    fit_sizes = sample_sizes[fit_mask]
    fit_errors = mean_errors[fit_mask]

    if fit_sizes.size < 2:
        raise ValueError("At least two points are required for algebraic fitting")
    if not np.all(np.isfinite(fit_errors)):
        raise ValueError("Cannot fit non-finite error values")
    if np.any(fit_errors <= 0):
        raise ValueError("Algebraic fitting requires strictly positive errors")

    slope, intercept = np.polyfit(
        np.log(fit_sizes),
        np.log(fit_errors),
        deg=1,
    )
    q_estimate = -slope
    c_estimate = np.exp(intercept)
    fitted_errors = c_estimate * fit_sizes ** (-q_estimate)
    return q_estimate, c_estimate, fit_sizes, fitted_errors


# =============================================================================
# 6. Plotting and saving
# =============================================================================

def plot_predictions(
    model,
    id_dataset,
    ood_dataset,
    ood_case,
    training_size,
    trial,
    device,
):
    """Plot truth, prediction, and absolute error for ID and OOD samples."""
    model.eval()
    figure = plt.figure(figsize=(12, 4 * number_of_examples))

    with torch.no_grad():
        for index in range(number_of_examples):
            rows = (
                (f"ID {id_test_distribution}", id_dataset),
                (ood_case["log_label"], ood_dataset),
            )
            for distribution_index, (name, dataset) in enumerate(rows):
                x, y = dataset[index]
                prediction = model(x.unsqueeze(0).to(device))[0].cpu()
                row = 2 * index + distribution_index
                error = torch.abs(prediction[0] - y[0])
                fields = (y[0], prediction[0], error)
                headings = ("Ground truth", "FNO prediction", "Absolute error")

                for column, (field, heading) in enumerate(zip(fields, headings)):
                    axis = figure.add_subplot(
                        2 * number_of_examples,
                        3,
                        row * 3 + column + 1,
                    )
                    image = axis.imshow(field.numpy(), cmap="viridis")
                    figure.colorbar(image, ax=axis)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if row == 0:
                        axis.set_title(heading)
                    if column == 0:
                        axis.set_ylabel(f"{name}\nexample {index + 1}")

    # figure.suptitle(
    #     f"Darcy FNO: {train_distribution} training, "
    #     f"{ood_case['log_label']} OOD, "
    #     f"N_train={training_size}, trial={trial + 1}",
    #     y=0.995,
    # )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR
        / (
            f"{experiment_name}_predictions_{ood_case['key']}_IDalpha_{id_alpha}"
            f"N_{training_size}_trial_{trial + 1}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    if show_figures:
        plt.show()
    plt.close(figure)


def plot_max_size_convergence(history, training_size):
    """Plot full training and ID test errors across epochs."""
    epoch_numbers = np.arange(1, epochs + 1)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.semilogy(epoch_numbers, history["train"], label="Training error")
    axis.semilogy(
        epoch_numbers,
        history["id_test"],
        label=f"ID test error ({id_test_distribution})",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric_axis_label())
    # axis.set_title(f"Darcy FNO training history: m={training_size}")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"{result_name}_convergence_N_{training_size}_IDalpha_{id_alpha}.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show_figures:
        plt.show()
    plt.close(figure)


def save_final_error_table(
    sample_sizes,
    id_mean,
    id_std,
    ood_mean,
    ood_std,
    ood_cases,
):
    """Save final evaluation metrics as a CSV table."""
    headers = ["training_size", "id_ln_mean", "id_ln_std"]

    for case in ood_cases:
        headers.extend([
            f"ood_{case['key']}_mean",
            f"ood_{case['key']}_std",
        ])

    rows = []
    for index, size in enumerate(sample_sizes):
        row = [size, id_mean[index], id_std[index]]
        for case in ood_cases:
            key = case["key"]
            row.extend([ood_mean[key][index], ood_std[key][index]])
        rows.append(row)

    np.savetxt(
        RESULTS_DIR / f"{result_name}_final_errors.csv",
        np.asarray(rows),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )


def positive_lower_edge(mean, std):
    """Keep the shaded error band positive on a logarithmic axis."""
    return np.maximum(mean - std, 1e-12)


def plot_final_errors(
    sample_sizes,
    id_mean,
    id_std,
    ood_mean,
    ood_std,
    ood_cases,
):
    """Plot mean errors, standard deviations, and algebraic fits."""
    figure, axis = plt.subplots(figsize=(8, 5.5))
    legend_handles = []
    legend_labels = []

    id_line = axis.plot(sample_sizes, id_mean, marker="o", linewidth=2)[0]
    axis.fill_between(
        sample_sizes,
        positive_lower_edge(id_mean, id_std),
        id_mean + id_std,
        color=id_line.get_color(),
        alpha=0.2,
    )

    if plot_algebraic_fit:
        id_q, id_c, id_fit_sizes, id_fitted_errors = fit_algebraic_rate(
            sample_sizes,
            id_mean,
        )
        id_fit_line = axis.plot(
            id_fit_sizes,
            id_fitted_errors,
            linestyle="--",
            linewidth=2,
            color=id_line.get_color(),
        )[0]
        legend_handles.append((id_line, id_fit_line))
        legend_labels.append(
            f"ID, {id_test_distribution}, " + rf"${id_c:.2f}m^{{-{id_q:.2f}}}$"
        )
        log(
            f"  Fit | ID {id_test_distribution} | c={id_c:.6e} "
            f"| q={id_q:.6f} | sizes={id_fit_sizes.astype(int).tolist()}"
        )
    else:
        legend_handles.append(id_line)
        legend_labels.append(f"ID, {id_test_distribution}")

    for case in ood_cases:
        key = case["key"]
        mean_errors = np.asarray(ood_mean[key], dtype=float)
        std_errors = np.asarray(ood_std[key], dtype=float)
        ood_line = axis.plot(
            sample_sizes,
            mean_errors,
            marker="s",
            linewidth=2,
        )[0]
        axis.fill_between(
            sample_sizes,
            positive_lower_edge(mean_errors, std_errors),
            mean_errors + std_errors,
            color=ood_line.get_color(),
            alpha=0.2,
        )

        if plot_algebraic_fit:
            ood_q, ood_c, ood_fit_sizes, ood_fitted_errors = fit_algebraic_rate(
                sample_sizes,
                mean_errors,
            )
            ood_fit_line = axis.plot(
                ood_fit_sizes,
                ood_fitted_errors,
                linestyle="--",
                linewidth=2,
                color=ood_line.get_color(),
            )[0]
            legend_handles.append((ood_line, ood_fit_line))
            legend_labels.append(
                case["label"] + ", " + rf"${ood_c:.2e}m^{{-{ood_q:.2f}}}$"
            )
            log(
                f"  Fit | {case['log_label']} | c={ood_c:.6e} "
                f"| q={ood_q:.6f} "
                f"| sizes={ood_fit_sizes.astype(int).tolist()}"
            )
        else:
            legend_handles.append(ood_line)
            legend_labels.append(case["label"])

    axis.set_xlabel("Number of training samples")
    axis.set_ylabel(metric_axis_label())
    # axis.set_title(
    #     f"Darcy FNO: {id_test_distribution} ID vs "
    #     f"{ood_test_distribution} OOD"
    # )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(sample_sizes)
    axis.set_xticklabels([str(size) for size in sample_sizes])
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)

    if plot_algebraic_fit:
        axis.legend(
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )
    else:
        axis.legend(legend_handles, legend_labels)

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"{result_name}_final_error_vs_training_size_IDalpha_{id_alpha}.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show_figures:
        plt.show()
    plt.close(figure)


def save_results(
    sample_sizes,
    id_errors,
    ood_errors,
    id_mean,
    id_std,
    ood_mean,
    ood_std,
    ood_cases,
):
    """Save raw errors, statistics, and algebraic-fit parameters."""
    arrays = {
        "training_sizes": np.asarray(sample_sizes),
        "fit_training_sizes": np.asarray(fit_training_sizes),
        "evaluation_metric": np.asarray(evaluation_metric),
        "id_distribution": np.asarray(id_test_distribution),
        "id_alpha": np.asarray(id_alpha),
        "id_raw": id_errors,
        "id_mean": id_mean,
        "id_std": id_std,
        "ood_distribution": np.asarray(ood_test_distribution),
    }

    if plot_algebraic_fit:
        id_q, id_c, _, _ = fit_algebraic_rate(sample_sizes, id_mean)
        arrays["id_fit_q"] = np.asarray(id_q)
        arrays["id_fit_c"] = np.asarray(id_c)

    for case in ood_cases:
        key = case["key"]
        arrays[f"ood_{key}_raw"] = ood_errors[key]
        arrays[f"ood_{key}_mean"] = ood_mean[key]
        arrays[f"ood_{key}_std"] = ood_std[key]

        if plot_algebraic_fit:
            ood_q, ood_c, _, _ = fit_algebraic_rate(
                sample_sizes,
                ood_mean[key],
            )
            arrays[f"ood_{key}_fit_q"] = np.asarray(ood_q)
            arrays[f"ood_{key}_fit_c"] = np.asarray(ood_c)

    np.savez_compressed(
        RESULTS_DIR / f"{result_name}_results.npz",
        **arrays,
    )


# =============================================================================
# 7. Main experiment
# =============================================================================

def main():
    validate_settings()
    set_seed(seed)
    device = choose_device()
    ood_cases = make_ood_cases()

    log("\n" + "=" * 70)
    log("DARCY / FNO ID-OOD EXPERIMENT")
    log("=" * 70)
    log(f"Device             : {device}")
    log(f"Resolution         : {resolution} x {resolution}")
    log(f"Training/ID data   : {train_distribution}, alpha={id_alpha:g}")
    log(f"OOD distribution   : {ood_test_distribution}")
    if ood_test_distribution == "COOKIE":
        log(f"Cookie radii       : {cookie_radii}")
    else:
        log(f"OOD alphas         : {ood_alphas}")
    log(f"Training sizes     : {training_sizes}")
    log(f"Epochs/trials      : {epochs}/{trials}")
    log(f"FNO modes/width    : {n_modes}/{hidden_channels}")
    log(f"Evaluation metric  : {metric_log_name()}")
    log(f"Generate new data  : {generate_new_data}")
    log(f"Train new models   : {train_new_models}")
    log("=" * 70)

    # Generate fixed LN ID test data once.
    log("\n[1/3] Preparing fixed test datasets")
    x_id, y_id = load_or_generate_dataset(
        f"id_{id_test_distribution}_alpha_{number_tag(id_alpha)}_N_{test_size}",
        test_size,
        id_alpha,
        f"ID {id_test_distribution}",
        id_test_distribution,
    )
    id_loader = make_loader(x_id, y_id, test_batch_size, False)
    id_dataset = id_loader.dataset

    ood_loaders = {}
    ood_datasets = {}
    for case in ood_cases:
        x_ood, y_ood = load_or_generate_dataset(
            f"ood_{case['key']}_N_{test_size}",
            test_size,
            case["alpha"],
            case["log_label"],
            ood_test_distribution,
            radius=case["radius"],
        )
        loader = make_loader(x_ood, y_ood, test_batch_size, False)
        ood_loaders[case["key"]] = loader
        ood_datasets[case["key"]] = loader.dataset

    # Rows are training sizes; columns are trials.
    id_errors = np.zeros((len(training_sizes), trials))
    ood_errors = {
        case["key"]: np.zeros((len(training_sizes), trials))
        for case in ood_cases
    }
    saved_history = None
    maximum_training_size = max(training_sizes)

    log("\n[2/3] Training and evaluating models")
    for size_index, training_size in enumerate(training_sizes):
        log("\n" + "-" * 70)
        log(f"Training size {size_index + 1}/{len(training_sizes)}: {training_size}")
        log("-" * 70)

        for trial in range(trials):
            log(f"  Trial {trial + 1}/{trials}")
            set_seed(seed + 1000 * size_index + trial)
            model = make_model(device)
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            log(f"  Model parameters: {parameter_count:,}")
            model_path = (
                MODEL_DIR
                / f"model_N_{training_size}_trial_{trial + 1}.pt"
            )
            record_history = (
                save_convergence_plot
                and training_size == maximum_training_size
                and trial == history_trial
            )

            if train_new_models:
                x_train, y_train = load_or_generate_dataset(
                    (
                        f"train_{train_distribution}_alpha_"
                        f"{number_tag(id_alpha)}_N_{training_size}_"
                        f"trial_{trial + 1}"
                    ),
                    training_size,
                    id_alpha,
                    f"train trial={trial + 1}",
                    train_distribution,
                )
                train_loader = make_loader(
                    x_train,
                    y_train,
                    training_batch_size,
                    True,
                )
                history = train_model(
                    model,
                    train_loader,
                    device,
                    id_loader=id_loader,
                    record_history=record_history,
                )
                torch.save(model.state_dict(), model_path)
                log(f"  [model] Saved: {model_path}")

                if record_history:
                    saved_history = history

                del x_train, y_train, train_loader
            else:
                state_dict = torch.load(
                    model_path,
                    map_location=device,
                    weights_only=True,
                )
                model.load_state_dict(state_dict)
                log(f"  [model] Loaded: {model_path}")

            id_error = test_model(model, id_loader, device)
            id_errors[size_index, trial] = id_error
            log(
                f"  [test] ID  | {id_test_distribution} alpha={id_alpha:g} "
                f"| {metric_log_name()}={id_error:.4e}"
            )

            for case in ood_cases:
                key = case["key"]
                error = test_model(model, ood_loaders[key], device)
                ood_errors[key][size_index, trial] = error
                log(
                    f"  [test] OOD | {case['log_label']} "
                    f"| {metric_log_name()}={error:.4e}"
                )

                if (
                    save_prediction_plots
                    and training_size == maximum_training_size
                    and trial == prediction_trial
                ):
                    plot_predictions(
                        model,
                        id_dataset,
                        ood_datasets[key],
                        case,
                        training_size,
                        trial,
                        device,
                    )

            del model

    id_mean = id_errors.mean(axis=1)
    id_std = id_errors.std(axis=1)
    ood_mean = {
        key: values.mean(axis=1) for key, values in ood_errors.items()
    }
    ood_std = {
        key: values.std(axis=1) for key, values in ood_errors.items()
    }

    log("\n[3/3] Saving figures and results")
    sample_sizes = np.asarray(training_sizes)
    plot_final_errors(
        sample_sizes,
        id_mean,
        id_std,
        ood_mean,
        ood_std,
        ood_cases,
    )

    if save_convergence_plot and saved_history is not None:
        history_path = (
            RESULTS_DIR / f"{result_name}_max_size_convergence.npz"
        )
        plot_max_size_convergence(saved_history, maximum_training_size)
        np.savez_compressed(
            history_path,
            training_size=np.asarray(maximum_training_size),
            epochs=np.arange(1, epochs + 1),
            history_trial=np.asarray(history_trial),
            history_train=np.asarray(saved_history["train"]),
            history_id_test=np.asarray(saved_history["id_test"]),
        )

    save_results(
        sample_sizes,
        id_errors,
        ood_errors,
        id_mean,
        id_std,
        ood_mean,
        ood_std,
        ood_cases,
    )
    save_final_error_table(
        sample_sizes,
        id_mean,
        id_std,
        ood_mean,
        ood_std,
        ood_cases,
    )

    log("\nExperiment complete")
    log(f"Figures: {FIGURE_DIR}")
    log(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()





