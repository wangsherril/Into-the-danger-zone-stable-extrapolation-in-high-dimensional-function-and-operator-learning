#!/usr/bin/env python3
"""NS data generation, training and evaluation in one script.

Set GENERATE_DATA, RESUME_GENERATION, TRAIN_MODEL and EVALUATE_MODEL below.
Requires the original utils/NS_random_fields.py and utils/utilities.py.
"""
import math
import hashlib
import json
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from matplotlib.legend_handler import HandlerTuple
from neuralop.models import FNO
from utils.utilities import GaussianNormalizer

# If using Google Colab, mount Drive and set PROJECT_ROOT manually.
#
# PROJECT_ROOT = Path(
#     "/content/drive/MyDrive/OOD_numerical_experiments/FNO"
# )

# Find the project root locally or in Colab/Jupyter.
if "__file__" in globals():
    SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = (
        SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
    )
else:
    PROJECT_ROOT = Path.cwd().resolve()

# Let Python find modules in the project folder.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.NS_random_fields import GaussianRF

DATA_DIR = PROJECT_ROOT / "data"


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

# Data-generation configuration.
viscosity = 1e-3
forcing_amplitude = 0.1
forcing_wave_number = 2.0 * math.pi
estimated_reynolds_number = math.sqrt(forcing_amplitude) / (viscosity * forcing_wave_number**1.5)
resolution = 64
final_time = 50
# True: dt=1e-4; False: dt=1e-3.
strict_reference_solver = False
time_step = 1.0e-4 if strict_reference_solver else 1.0e-3  #solver time_step
record_steps = 50
training_alpha = 2.5
test_alphas = [2.0, 2.25, 2.5]
grf_tau = 7.0
training_sizes = [150, 300, 600, 1200, 2400, 4800]

test_size = 200
generation_batch_size = 20

# Parallel CPU data generation.
PARALLEL_GENERATION = True
NUM_GENERATION_WORKERS = 4

# Generation switches: overwrite takes priority over resume.
GENERATE_DATA = False
RESUME_GENERATION = False
OVERWRITE_DATA = False
RESUME_TRAINING = False

# Save each 50 new samples and the final remainder.
SAVE_EVERY = 50

from utils.utilities import MatReader, UnitGaussianNormalizer, count_params, LpLoss

FIGURE_DIR = PROJECT_ROOT / "fig" / "NS"
RESULTS_DIR = PROJECT_ROOT / "results" / "NS"
PREDICTION_DIR = PROJECT_ROOT / "pred" / "NS"
MODEL_DIR = PROJECT_ROOT / "models" / "NS"

# Training/evaluation configuration.
subsample = 1
effective_resolution = resolution // subsample
time_start = 0
number_of_time_steps = 50
time_padding = round(0.125 * number_of_time_steps)
# round(0.125 * 50) = 6
trials = 5

plot_algebraic_fit = True

fit_training_sizes = training_sizes
fourier_modes = 12
hidden_width = 32
batch_size = 32
epochs = 200
learning_rate = 1e-3
weight_decay = 0
use_scheduler = True
scheduler_step = 50
scheduler_gamma = 0.5
training_loss = "rel_l2"

# False loads saved weights and training normalization statistics.
TRAIN_MODEL = False
EVALUATE_MODEL = False
# MSE averages physical-space squared errors over samples, x, y and time.
EVAL_METRICS = ["mse"]
save_sample_plots = True
save_predictions = True
save_rel_l2_plot = True
save_linf_plot = True
save_mse_plot = True
show_table = False
show_figures = True
number_of_examples = 3
prediction_trial = 0
time_id = number_of_time_steps - 1
seed = 2026
verbose = True
PLOT_ONLY = True

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


def get_data_path(split, number_of_samples, alpha):
    """Return the file path for one NS dataset."""
    filename = (
        f"ns_{split}_"
        f"nu{viscosity:g}_"
        f"Re{estimated_reynolds_number:.2f}_"
        f"N{number_of_samples}_"
        f"T{final_time:g}_"
        f"s{resolution}_"
        f"alpha{alpha:g}_"
        f"tau{grf_tau:g}.mat"
    )

    return DATA_DIR / filename

def generation_settings():
    return {
        "resolution": resolution, "viscosity": viscosity,
        "forcing_amplitude": forcing_amplitude, "forcing_wave_number": forcing_wave_number,
        "final_time": final_time, "time_step": time_step,
        "record_steps": record_steps, "grf_tau": grf_tau,
    }

def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    """Solve the 2D vorticity equation using the reference spectral method."""
    N = w0.size(-1)  # Grid size; N must be a power of two.
    k_max = math.floor(N / 2.0)
    steps = math.ceil(T / delta_t)

    # New PyTorch replacement for torch.rfft(..., onesided=False).
    w_h = torch.view_as_real(torch.fft.fftn(w0, dim=(-2, -1)))
    f_h = torch.view_as_real(torch.fft.fftn(f, dim=(-2, -1)))
    if f_h.ndim < w_h.ndim:
        f_h = f_h.unsqueeze(0)

    record_time = math.floor(steps / record_steps)
    if record_time < 1:
        raise ValueError("record_steps cannot exceed the number of solver steps")

    k_y = torch.cat(
        (torch.arange(0, k_max, device=w0.device), torch.arange(-k_max, 0, device=w0.device))
    ).repeat(N, 1)
    k_x = k_y.transpose(0, 1)
    lap = 4 * math.pi**2 * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0
    dealias = (
        ((torch.abs(k_y) <= (2.0 / 3.0) * k_max) & (torch.abs(k_x) <= (2.0 / 3.0) * k_max))
        .float()
        .unsqueeze(0)
    )

    sol = torch.zeros(*w0.size(), record_steps, device=w0.device, dtype=w0.dtype)
    sol_t = torch.zeros(record_steps, device=w0.device, dtype=w0.dtype)
    c = 0
    t = 0.0

    for j in range(steps):
        # Stream function: solve the Poisson equation in Fourier space.
        psi_h = w_h.clone()
        psi_h[..., 0] /= lap
        psi_h[..., 1] /= lap

        # Velocity q = psi_y.
        q = psi_h.clone()
        temp = q[..., 0].clone()
        q[..., 0] = -2 * math.pi * k_y * q[..., 1]
        q[..., 1] = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(torch.view_as_complex(q), s=(N, N), dim=(-2, -1)).real

        # Velocity v = -psi_x.
        v = psi_h.clone()
        temp = v[..., 0].clone()
        v[..., 0] = 2 * math.pi * k_x * v[..., 1]
        v[..., 1] = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(torch.view_as_complex(v), s=(N, N), dim=(-2, -1)).real

        # Spatial derivatives of vorticity.
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
        w_x[..., 1] = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x), s=(N, N), dim=(-2, -1)).real

        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y), s=(N, N), dim=(-2, -1)).real

        # Nonlinear term u · grad(w), followed by dealiasing.
        F_h = torch.view_as_real(torch.fft.fftn(q * w_x + v * w_y, dim=(-2, -1)))
        F_h[..., 0] *= dealias
        F_h[..., 1] *= dealias

        # Crank--Nicolson update.
        denominator = 1.0 + 0.5 * delta_t * visc * lap
        multiplier = 1.0 - 0.5 * delta_t * visc * lap
        w_h[..., 0] = (
            -delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + multiplier * w_h[..., 0]
        ) / denominator
        w_h[..., 1] = (
            -delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + multiplier * w_h[..., 1]
        ) / denominator

        t += delta_t
        if (j + 1) % record_time == 0 and c < record_steps:
            w = torch.fft.ifftn(torch.view_as_complex(w_h), s=(N, N), dim=(-2, -1)).real
            sol[..., c] = w
            sol_t[c] = t
            c += 1

    if c != record_steps:
        raise RuntimeError(f"Recorded {c} snapshots; expected {record_steps}")

    return sol, sol_t

def generate_batch_worker(args):
    """Solve one already-sampled batch on a single CPU process."""
    batch_id, initial_vorticity_array = args

    # Avoid nested CPU threading when several worker processes run at once.
    torch.set_num_threads(1)
    device = torch.device("cpu")
    initial_vorticity = torch.from_numpy(initial_vorticity_array).to(device)

    coordinate = torch.linspace(0, 1, resolution + 1, device=device)[:-1]
    grid_x, grid_y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    forcing = forcing_amplitude * (
        torch.sin(forcing_wave_number * (grid_x + grid_y))
        + torch.cos(forcing_wave_number * (grid_x + grid_y))
    )

    solution, recorded_times = navier_stokes_2d(
        w0=initial_vorticity,
        f=forcing,
        visc=viscosity,
        T=final_time,
        delta_t=time_step,
        record_steps=record_steps,
    )

    if not torch.isfinite(solution).all():
        raise ValueError(f"Navier-Stokes solver produced NaN or Inf in batch {batch_id}")

    return (
        batch_id,
        solution.cpu().numpy(),
        recorded_times.cpu().numpy(),
    )


def save_generated_data(file_path, initial_conditions, solutions, recorded_times, count, device):
    """Save completed samples and RNG state; replace the previous file only after writing."""
    data = {
        "a": initial_conditions[:count].cpu().numpy(),
        "u": solutions[:count].cpu().numpy(),
        "t": recorded_times.cpu().numpy(),
        "rng_cpu": torch.get_rng_state().numpy(),
        "rng_device": device.type,
        "generation_settings": json.dumps(generation_settings(), sort_keys=True),
    }
    numpy_state = np.random.get_state()
    data.update(
        rng_numpy_name=numpy_state[0], rng_numpy_keys=numpy_state[1],
        rng_numpy_pos=numpy_state[2], rng_numpy_has_gauss=numpy_state[3],
        rng_numpy_cached=numpy_state[4],
    )
    if device.type == "cuda":
        data["rng_cuda"] = torch.cuda.get_rng_state(device).cpu().numpy()

    temporary_path = file_path.with_suffix(".tmp.mat")
    scipy.io.savemat(temporary_path, data)
    temporary_path.replace(file_path)
    log(f"    saved {count} samples: {file_path.name}")

def generate_dataset(file_path, number_of_samples, alpha, device):
    """Generate one dataset, optionally continuing from its saved sample count."""
    saved = None
    generated_samples = 0
    if file_path.exists() and RESUME_GENERATION and not OVERWRITE_DATA:
        saved = scipy.io.loadmat(file_path)
        generated_samples = saved["a"].shape[0]
        if saved["u"].shape != (generated_samples, resolution, resolution, record_steps):
            raise ValueError(f"Saved dataset has incompatible dimensions: {file_path}")
        if "generation_settings" in saved:
            if json.loads(saved["generation_settings"].item()) != generation_settings():
                raise ValueError(f"Generation settings differ from saved data: {file_path}")

        log(f"    loaded: {file_path.name} ({generated_samples} samples)")

    random_field = GaussianRF(2, resolution, alpha=alpha, tau=grf_tau, device=device)
    if saved is not None:
        if "rng_cpu" in saved and saved["rng_device"].item() == device.type:
            torch.set_rng_state(torch.from_numpy(saved["rng_cpu"].ravel()))
            if device.type == "cuda":
                torch.cuda.set_rng_state(torch.from_numpy(saved["rng_cuda"].ravel()), device)
            np.random.set_state((
                saved["rng_numpy_name"].item(), saved["rng_numpy_keys"].ravel(),
                int(saved["rng_numpy_pos"].item()), int(saved["rng_numpy_has_gauss"].item()),
                float(saved["rng_numpy_cached"].item()),
            ))
        else:
            # Legacy files/device changes cannot replay the old RNG stream exactly.
            set_seed(int(np.random.SeedSequence().generate_state(1)[0]))
            log("    RNG state unavailable on this device; using a fresh continuation stream")

    if generated_samples >= number_of_samples:
        log(f"    complete: {file_path.name}")
        return

    coordinate = torch.linspace(0, 1, resolution + 1, device=device)[:-1]
    grid_x, grid_y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    forcing = forcing_amplitude * (
        torch.sin(forcing_wave_number * (grid_x + grid_y))
        + torch.cos(forcing_wave_number * (grid_x + grid_y))
    )
    initial_conditions = torch.zeros(number_of_samples, resolution, resolution, device=device)
    solutions = torch.zeros(number_of_samples, resolution, resolution, record_steps, device=device)
    if saved is not None:
        initial_conditions[:generated_samples] = torch.as_tensor(saved["a"], device=device)
        solutions[:generated_samples] = torch.as_tensor(saved["u"], device=device)
    starting_samples = generated_samples
    next_save = min(generated_samples + SAVE_EVERY, number_of_samples)
    batch_number = 0
    start_time = default_timer()

    if PARALLEL_GENERATION:
        if device.type != "cpu":
            raise ValueError("PARALLEL_GENERATION=True requires CPU data generation")

        while generated_samples < number_of_samples:
            target_samples = next_save
            scheduled_samples = generated_samples
            batch_requests = []
            batch_initial_conditions = {}

            # Keep random-field sampling in the main process so the RNG stream
            # and resume behavior stay the same as in serial generation.
            while scheduled_samples < target_samples:
                batch_number += 1
                current_batch_size = min(
                    generation_batch_size, target_samples - scheduled_samples
                )
                initial_vorticity = (
                    random_field.sample(current_batch_size)
                    / resolution**2
                )
                batch_initial = initial_vorticity.cpu().numpy()
                batch_requests.append((batch_number, batch_initial))
                batch_initial_conditions[batch_number] = batch_initial
                scheduled_samples += current_batch_size

            completed = {}
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=NUM_GENERATION_WORKERS,
                mp_context=ctx,
            ) as executor:
                futures = [
                    executor.submit(generate_batch_worker, request)
                    for request in batch_requests
                ]

                for future in as_completed(futures):
                    result = future.result()
                    completed[result[0]] = result

            for current_batch_id, _ in batch_requests:
                _, batch_solution, batch_times = completed[current_batch_id]
                batch_initial = batch_initial_conditions[current_batch_id]
                current_batch_size = batch_initial.shape[0]
                next_index = generated_samples + current_batch_size
                initial_conditions[generated_samples:next_index] = torch.from_numpy(batch_initial)
                solutions[generated_samples:next_index] = torch.from_numpy(batch_solution)
                generated_samples = next_index
                recorded_times = torch.from_numpy(batch_times)

                elapsed_time = default_timer() - start_time
                sample_rate = (
                    (generated_samples - starting_samples) / elapsed_time
                    if elapsed_time > 0
                    else float("inf")
                )
                log(
                    f"    batch {current_batch_id:>3} "
                    f"| samples "
                    f"{generated_samples:>4}/{number_of_samples:<4} "
                    f"| elapsed={elapsed_time:7.2f}s "
                    f"| rate={sample_rate:6.2f}/s"
                )

            if generated_samples == next_save:
                save_generated_data(
                    file_path, initial_conditions, solutions, recorded_times, generated_samples, device
                )
                next_save = min(generated_samples + SAVE_EVERY, number_of_samples)

        return

    while generated_samples < number_of_samples:
        batch_number += 1
        current_batch_size = min(generation_batch_size, next_save - generated_samples)

        # Preserve the supplied sampler scale (ifftn with norm="forward").
        initial_vorticity = (
            random_field.sample(current_batch_size)
            / resolution**2
        )
        solution, recorded_times = navier_stokes_2d(
            w0=initial_vorticity,
            f=forcing,
            visc=viscosity,
            T=final_time,
            delta_t=time_step,
            record_steps=record_steps,
        )

        if not torch.isfinite(solution).all():
            raise ValueError("Navier-Stokes solver produced NaN or Inf")

        next_index = generated_samples + current_batch_size
        initial_conditions[generated_samples:next_index] = initial_vorticity
        solutions[generated_samples:next_index] = solution
        generated_samples = next_index
        elapsed_time = default_timer() - start_time
        sample_rate = (generated_samples - starting_samples) / elapsed_time if elapsed_time > 0 else float("inf")
        log(
            f"    batch {batch_number:>3} "
            f"| samples "
            f"{generated_samples:>4}/{number_of_samples:<4} "
            f"| elapsed={elapsed_time:7.2f}s "
            f"| rate={sample_rate:6.2f}/s"
        )

        if generated_samples == next_save:
            save_generated_data(
                file_path, initial_conditions, solutions, recorded_times, generated_samples, device
            )
            next_save = min(generated_samples + SAVE_EVERY, number_of_samples)


def load_results():
    """Load previously saved evaluation results."""
    results = {}

    for metric in EVAL_METRICS:
        file_path = RESULTS_DIR / f"ns_fno3d_{metric}_results.npz"

        if not file_path.exists():
            raise FileNotFoundError(f"Results not found: {file_path}")

        with np.load(file_path) as data:
            saved_sizes = data["training_sizes"]
            saved_alphas = data["test_alphas"]

            if not np.array_equal(saved_sizes, np.asarray(training_sizes)):
                raise ValueError("Saved training_sizes do not match current settings")

            if not np.allclose(saved_alphas, np.asarray(test_alphas)):
                raise ValueError("Saved test_alphas do not match current settings")

            results[metric] = {}

            for alpha in test_alphas:
                alpha_name = str(alpha).replace(".", "p")
                key = f"{metric}_alpha_{alpha_name}_raw"
                results[metric][alpha] = data[key]

    return results

def prepare_datasets(device):
    """Generate missing datasets when requested."""
    if not GENERATE_DATA:
        return

    dataset_requests = [("train", size, training_alpha) for size in training_sizes]
    dataset_requests += [("test", test_size, alpha) for alpha in test_alphas]

    for split, number_of_samples, alpha in dataset_requests:
        file_path = get_data_path(split, number_of_samples, alpha)

        if file_path.exists() and not OVERWRITE_DATA and not RESUME_GENERATION:
            log(f"    reuse: {file_path.name}")
            continue

        log(
            f"    generating {split.upper():5s} "
            f"| N={number_of_samples:<4} "
            f"| alpha={alpha:g}"
        )
        generate_dataset(
            file_path=file_path, number_of_samples=number_of_samples, alpha=alpha, device=device
        )


def validate_settings():
    """Stop early when a setting is invalid."""
    if resolution <= 0 or resolution & (resolution - 1):
        raise ValueError("resolution must be a positive power of two")

    if subsample <= 0 or resolution % subsample != 0:
        raise ValueError("subsample must be a positive divisor of resolution")

    if final_time <= 0 or time_step <= 0:
        raise ValueError("final_time and time_step must be positive")

    total_solver_steps = math.ceil(final_time / time_step)

    if not 0 < record_steps <= total_solver_steps:
        raise ValueError("record_steps cannot exceed the number of solver steps")

    if time_start < 0 or number_of_time_steps <= 0:
        raise ValueError(
            "time_start must be nonnegative and number_of_time_steps must be positive"
        )

    if time_start + number_of_time_steps > record_steps:
        raise ValueError("The requested FNO time interval exceeds record_steps")

    if fourier_modes <= 0 or fourier_modes > effective_resolution // 2:
        raise ValueError(
            f"fourier_modes={fourier_modes} is incompatible with "
            f"effective_resolution={effective_resolution}"
        )

    if fourier_modes > number_of_time_steps // 2 + 1:
        raise ValueError(
            f"fourier_modes={fourier_modes} is incompatible with "
            f"number_of_time_steps={number_of_time_steps}"
        )

    if not training_sizes or any(size <= 0 for size in training_sizes):
        raise ValueError("training_sizes must contain positive numbers")

    if test_size <= 0 or trials <= 0:
        raise ValueError("test_size and trials must be positive")

    if training_alpha not in test_alphas:
        raise ValueError("test_alphas must include training_alpha for ID testing")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")

    if training_loss not in {"rel_l2", "abs_l2", "mse"}:
        raise ValueError("training_loss must be 'rel_l2', 'abs_l2', or 'mse'")

    if EVALUATE_MODEL:
        valid_metrics = {"rel_l2", "final_rel_l2", "linf", "mse"}

        if not EVAL_METRICS:
            raise ValueError("EVAL_METRICS cannot be empty")

        if any(metric not in valid_metrics for metric in EVAL_METRICS):
            raise ValueError(
                "EVAL_METRICS can only contain 'rel_l2', 'final_rel_l2', 'linf', and 'mse'"
            )

        if not 0 < number_of_examples <= test_size:
            raise ValueError("number_of_examples must be between 1 and test_size")

        if not 0 <= prediction_trial < trials:
            raise ValueError("prediction_trial must be a valid trial index")

        if not 0 <= time_id < number_of_time_steps:
            raise ValueError("time_id must index the selected time snapshots")

        if plot_algebraic_fit:
            if len(fit_training_sizes) < 2:
                raise ValueError("fit_training_sizes must contain at least two values")

            invalid_fit_sizes = [size for size in fit_training_sizes if size not in training_sizes]

            if invalid_fit_sizes:
                raise ValueError(
                    "Every value in fit_training_sizes must also "
                    f"appear in training_sizes. Invalid values: "
                    f"{invalid_fit_sizes}"
                )


def print_configuration(device):
    """Print the experiment settings."""
    log("\n" + "=" * 70)
    log("NAVIER-STOKES FNO3D ID/OOD EXPERIMENT")
    log("=" * 70)
    log(f"Device            : {device}")
    log(f"Spatial grid      : {resolution} x {resolution}")
    log(f"Effective grid    : {effective_resolution} x {effective_resolution}")
    log(f"Final time        : {final_time:g}")
    log(f"Time step         : {time_step:g}")
    log(f"Reference solver  : {strict_reference_solver}")
    log(f"Recorded steps    : {record_steps}")
    log(f"FNO time steps    : {number_of_time_steps}")
    log(f"Reynolds number   : {estimated_reynolds_number:.4f} (forcing-based estimate)")
    log(f"Viscosity         : {viscosity:g}")
    log(f"Training alpha    : {training_alpha:g}")
    log(f"Test alphas       : {test_alphas}")
    log(f"Training sizes    : {training_sizes}")
    log(f"Test size         : {test_size}")
    log(f"Trials            : {trials}")
    log(f"Fourier modes     : {fourier_modes}")
    log(f"Hidden width      : {hidden_width}")
    log("Activation        : GELU")
    log(f"Epochs            : {epochs}")
    log(f"Batch size        : {batch_size}")
    log(f"Training loss     : {training_loss}")
    log(f"Evaluation        : {EVAL_METRICS}")
    log(f"Retrain / evaluate: {TRAIN_MODEL} / {EVALUATE_MODEL}")
    log("=" * 70)


l2_loss = LpLoss(size_average=True)


def fno_config():
    return {
        "n_modes": (fourier_modes,) * 3,
        # "n_modes": (12, 12, 2)
        "in_channels": 4, "out_channels": 1,
        "hidden_channels": hidden_width, "n_layers": 4,
        "lifting_channel_ratio": 1, "projection_channel_ratio": 2,
        "positional_embedding": None, "use_channel_mlp": False,
    }

class FNO3d(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.fno = FNO(**(fno_config() if config is None else config))

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)

        x = F.pad(x, (0, time_padding))  # 50 -> 56

        x = self.fno(x)

        x = x[..., :-time_padding]       # 56 -> 50

        return x.permute(0, 2, 3, 4, 1)


class TrajectoryDataset(Dataset):
    """Build each space-time input only when it is needed, saving memory."""

    def __init__(self, initial_fields, outputs):
        self.initial_fields = initial_fields
        self.outputs = outputs
        grid_x = (
            (torch.arange(effective_resolution, dtype=torch.float32) / effective_resolution)
            .reshape(effective_resolution, 1, 1, 1)
            .expand(effective_resolution, effective_resolution, number_of_time_steps, 1)
        )
        grid_y = (
            (torch.arange(effective_resolution, dtype=torch.float32) / effective_resolution)
            .reshape(1, effective_resolution, 1, 1)
            .expand(effective_resolution, effective_resolution, number_of_time_steps, 1)
        )

        # Preserve the normalized time coordinate (0, 1] for the selected outputs.
        grid_t = (
            torch.linspace(1.0 / number_of_time_steps, 1.0, number_of_time_steps)
            .reshape(1, 1, number_of_time_steps, 1)
            .expand(effective_resolution, effective_resolution, number_of_time_steps, 1)
        )
        self.grid = torch.cat((grid_x, grid_y, grid_t), dim=-1)

    def __len__(self):
        return self.initial_fields.shape[0]

    def __getitem__(self, index):
        initial_field = self.initial_fields[index]
        repeated_field = initial_field.reshape(
            effective_resolution, effective_resolution, 1, 1
        ).expand(effective_resolution, effective_resolution, number_of_time_steps, 1)
        model_input = torch.cat((self.grid, repeated_field), dim=-1)

        return model_input, self.outputs[index]


def load_dataset(split, number_of_samples, alpha):
    """Read and validate the selected spatial and temporal data slices."""
    file_path = get_data_path(split, number_of_samples, alpha)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}. Run with GENERATE_DATA=True first.")
    reader = MatReader(str(file_path))
    inputs = reader.read_field("a")[:number_of_samples, ::subsample, ::subsample]
    outputs = reader.read_field("u")[
        :number_of_samples, ::subsample, ::subsample,
        time_start : time_start + number_of_time_steps
    ]
    expected_input_shape = (number_of_samples, effective_resolution, effective_resolution)
    expected_output_shape = (*expected_input_shape, number_of_time_steps)
    if inputs.shape != expected_input_shape:
        raise ValueError(f"Incomplete or incompatible {split} data: expected {expected_input_shape}, got {inputs.shape}")

    if outputs.shape != expected_output_shape:
        raise ValueError(f"Unexpected {split} output shape: {outputs.shape}")

    return inputs, outputs


def load_training_data(training_size):
    """Fit both normalizers on training data and encode inputs and outputs."""
    train_input, train_output = load_dataset("train", training_size, training_alpha)
    input_normalizer = UnitGaussianNormalizer(train_input)
    output_normalizer = UnitGaussianNormalizer(train_output)
    train_input = input_normalizer.encode(train_input)
    train_output = output_normalizer.encode(train_output)
    return train_input, train_output, input_normalizer, output_normalizer


def load_test_data(alpha, input_normalizer):
    """Encode test inputs with training statistics; keep targets in physical units."""
    test_input, test_output = load_dataset("test", test_size, alpha)
    test_input = input_normalizer.encode(test_input)
    return test_input, test_output


def make_loader(inputs, outputs, loader_batch_size, shuffle):
    """Create a DataLoader from input and output tensors."""
    dataset = TrajectoryDataset(inputs, outputs)

    return DataLoader(dataset, batch_size=loader_batch_size, shuffle=shuffle)


def selected_training_loss(prediction, target):

    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    """Return the selected training loss."""
    if training_loss == "mse":
        return F.mse_loss(prediction, target)

    if training_loss == "rel_l2":
        return l2_loss.rel(prediction_flat, target_flat)

    if training_loss == "abs_l2":
        return l2_loss.abs(prediction_flat, target_flat)




def calculate_metrics(prediction, target):
    """Calculate the selected evaluation metrics."""
    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    metrics = {}

    if "mse" in EVAL_METRICS:
        # Mean squared error over all test samples, spatial points, and selected times.
        metrics["mse"] = F.mse_loss(prediction, target)

    if "rel_l2" in EVAL_METRICS:
        metrics["rel_l2"] = l2_loss.rel(prediction_flat, target_flat)

    if "final_rel_l2" in EVAL_METRICS:
        final_prediction = prediction[..., -1].reshape(prediction.shape[0], -1)
        final_target = target[..., -1].reshape(target.shape[0], -1)
        metrics["final_rel_l2"] = l2_loss.rel(final_prediction, final_target)

    if "linf" in EVAL_METRICS:
        metrics["linf"] = torch.max(torch.abs(prediction_flat - target_flat), dim=1).values.mean()

    return metrics


def fit_algebraic_rate(metric_values):
    """Fit error(m) = c * m^(-q) on selected training sizes."""
    sample_sizes = np.asarray(training_sizes, dtype=float)
    errors = np.asarray(metric_values, dtype=float)

    if errors.shape != sample_sizes.shape:
        raise ValueError("metric_values must have the same length as training_sizes")

    fit_mask = np.isin(sample_sizes, np.asarray(fit_training_sizes, dtype=float))
    fit_sizes = sample_sizes[fit_mask]
    fit_errors = errors[fit_mask]

    if fit_sizes.size < 2:
        raise ValueError("At least two points are required for algebraic fitting")

    if not np.all(np.isfinite(fit_errors)):
        raise ValueError("Cannot fit non-finite error values")

    if np.any(fit_errors <= 0):
        raise ValueError("Algebraic fitting requires strictly positive errors")

    # log(error) = log(c) - q log(m)
    slope, intercept = np.polyfit(np.log(fit_sizes), np.log(fit_errors), deg=1)
    q_estimate = -slope
    c_estimate = np.exp(intercept)
    fitted_errors = c_estimate * fit_sizes ** (-q_estimate)

    return (q_estimate, c_estimate, fit_sizes, fitted_errors)


def train_model(
    model,
    train_loader,
    output_normalizer,
    device,
    training_size,
    trial,
):
    """Train one FNO3d model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )

    print_every = 10
    # max(1, epochs // 10)


    checkpoint_path = MODEL_DIR / (
        f"checkpoint_N{training_size}_trial{trial + 1}.pt"
    )

    # Load checkpoint if available
    if RESUME_TRAINING and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint["model_state_dict"]

        if "_metadata" in state_dict:
            state_dict = state_dict.copy()
            state_dict.pop("_metadata")

        model.load_state_dict(state_dict)

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if scheduler is not None:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        log(f"  Loaded checkpoint: {checkpoint_path.name}")

    for epoch in range(epochs):
        model.train()

        epoch_start = default_timer()
        total_loss = 0.0
        total_samples = 0

        for x, encoded_target in train_loader:
            x = x.to(device)
            encoded_target = encoded_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            encoded_prediction = model(x).squeeze(-1)

            # Decode before calculating the physical-space loss.
            prediction = output_normalizer.decode(encoded_prediction)
            target = output_normalizer.decode(encoded_target)
            loss = selected_training_loss(prediction, target)
            # loss = selected_training_loss(encoded_prediction, encoded_target)
            loss.backward()
            optimizer.step()
            current_batch_size = x.shape[0]
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        if scheduler is not None:
            scheduler.step()

        epoch_number = epoch + 1

        if (epoch + 1) % 20 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict() if scheduler else None,
            }, checkpoint_path)


        if epoch_number == 1 or epoch_number % print_every == 0 or epoch_number == epochs:
            elapsed_time = default_timer() - epoch_start
            mean_loss = total_loss / total_samples
            current_lr = optimizer.param_groups[0]["lr"]
            log(
                f"    epoch "
                f"{epoch_number:>5}/{epochs:<5} "
                f"| time={elapsed_time:7.2f}s "
                f"| {training_loss}={mean_loss:.4e} "
                f"| lr={current_lr:.3e}"
            )


def test_model(model, test_loader, output_normalizer, device):
    """Predict the test set and calculate evaluation metrics."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for x, target in test_loader:
            x = x.to(device)
            encoded_prediction = model(x).squeeze(-1)
            prediction = output_normalizer.decode(encoded_prediction)
            predictions.append(prediction.cpu())
            targets.append(target.cpu())

    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    sample_metrics = calculate_metrics(prediction, target)
    mean_metrics = {metric: values.mean().item() for metric, values in sample_metrics.items()}

    return prediction, target, mean_metrics


def plot_sample_predictions(plot_data, ood_alpha, training_size, trial):
    """Plot selected ID and OOD samples together, matching Darcy."""
    figure = plt.figure(figsize=(12, 4 * number_of_examples))
    id_target, id_prediction = plot_data[training_alpha]
    ood_target, ood_prediction = plot_data[ood_alpha]

    for index in range(number_of_examples):
        for distribution_index, (name, target, prediction) in enumerate(
            (("ID", id_target, id_prediction), ("OOD", ood_target, ood_prediction))
        ):
            truth = target[index, :, :, time_id]
            predicted = prediction[index, :, :, time_id]
            fields = (truth, predicted, torch.abs(predicted - truth))
            row = 2 * index + distribution_index

            for column, (field, heading) in enumerate(
                zip(fields, ("Ground truth", "FNO prediction", "Absolute error"))
            ):
                axis = figure.add_subplot(2 * number_of_examples, 3, row * 3 + column + 1)
                image = axis.imshow(field.numpy(), cmap="viridis")
                figure.colorbar(image, ax=axis)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(heading)
                if column == 0:
                    axis.set_ylabel(f"{name} {index + 1}")

    # figure.suptitle(
    #     f"Navier--Stokes FNO3d: OOD alpha={ood_alpha:g}, "
    #     f"N_train={training_size}, trial={trial + 1}",
    #     y=0.995,
    # )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"predictions_alpha_{ood_alpha:g}_N_{training_size}_trial_{trial + 1}.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show_figures:
        plt.show()

    plt.close(figure)


def metric_label(metric):
    """Return a readable plot label."""
    if metric == "mse":
        return "MSE"

    if metric == "linf":
        return r"$L^\infty$ error"

    if metric == "final_rel_l2":
        return r"Final-time relative $L^2$ error"

    return r"Relative $L^2$ error"


def should_save_metric_plot(metric):
    """Return whether one metric plot should be saved."""
    if metric in {"rel_l2", "final_rel_l2"}:
        return save_rel_l2_plot

    if metric == "mse":
        return save_mse_plot

    if metric == "linf":
        return save_linf_plot

    return False


def calculate_w2_distances():
  
    frequencies = np.fft.fftfreq(resolution, d=1.0 / resolution)
    k1, k2 = np.meshgrid(frequencies, frequencies, indexing="ij")
    operator_eigenvalues = 4.0 * np.pi**2 * (k1**2 + k2**2) + grf_tau**2

    def covariance_sqrt(alpha):
        # Match NS_random_fields.py, including its unnormalized inverse FFT.

        amplitudes = (
            np.sqrt(2.0)
            * grf_tau**(alpha - 1.0)
            * operator_eigenvalues**(-alpha / 2.0)
        )
        amplitudes[0, 0] = 0.0
        variances = (amplitudes**2).reshape(
            subsample, effective_resolution, subsample, effective_resolution
        ).sum(axis=(0, 2))
        return np.sqrt(variances)

    id_spectrum = covariance_sqrt(training_alpha)
    return {
        alpha: float(np.linalg.norm(covariance_sqrt(alpha) - id_spectrum))
        for alpha in test_alphas
    }


def plot_errors(metric, metric_results, w2_distances):
    """Plot one ID/OOD metric and its algebraic fits."""
    if show_table:
        figure = plt.figure(figsize=(11, 6))
        grid = figure.add_gridspec(1, 2, width_ratios=[4, 1.2], wspace=0.08)
        axis = figure.add_subplot(grid[0, 0])
        table_axis = figure.add_subplot(grid[0, 1])
        table_axis.axis("off")
    else:
        figure, axis = plt.subplots(figsize=(8, 6))

    table_rows = []
    row_colors = []
    legend_handles = []
    legend_labels = []

    for alpha in reversed(test_alphas):
        distribution_label = "ID" if alpha == training_alpha else "OOD"
        raw_errors = np.asarray(metric_results[alpha], dtype=float)
        mean_errors = raw_errors.mean(axis=1)
        std_errors = raw_errors.std(axis=1)
        marker = "o" if alpha == training_alpha else "s"

        data_line = axis.plot(
            training_sizes,
            mean_errors,
            marker=marker,
            linewidth=2
        )[0]

        axis.fill_between(
            training_sizes,
            np.maximum(mean_errors - std_errors, 1e-12),
            mean_errors + std_errors,
            color=data_line.get_color(),
            alpha=0.2,
        )
        base_label = rf"{distribution_label}, $\alpha={alpha:g}$"
        if show_table:
            w2 = w2_distances[alpha]
            table_rows.append([
                f"{distribution_label}, {alpha:g}",
                f"{w2:.4f}",
                f"{mean_errors[-1]:.4e}",
            ])
            row_colors.append(data_line.get_color())

        if plot_algebraic_fit:
            (q_estimate, c_estimate, fit_sizes, fitted_errors) = fit_algebraic_rate(mean_errors)
            fit_line = axis.plot(
                fit_sizes, fitted_errors, linestyle="--", linewidth=2, color=data_line.get_color()
            )[0]

            # One legend entry for the data and its fit.
            legend_handles.append((data_line, fit_line))
            legend_labels.append(
                base_label + ", " + rf"${c_estimate:.0f}" + rf"m^{{-{q_estimate:.2f}}}$"
            )
            log(
                f"  Fit | metric={metric} "
                f"| alpha={alpha:g} "
                f"| distribution={distribution_label} "
                f"| c={c_estimate:.6e} "
                f"| q={q_estimate:.6f} "
                f"| sizes={fit_sizes.astype(int).tolist()}"
            )

        else:
            legend_labels.append(base_label)
            legend_handles.append(data_line)


    axis.set_xlabel("Number of training samples")
    axis.set_ylabel(metric_label(metric))
    # axis.set_title(f"Navier-Stokes FNO3d: {metric_label(metric)}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(training_sizes)
    axis.set_xticklabels([str(size) for size in training_sizes])
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)

    if plot_algebraic_fit:
        axis.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        axis.legend(legend_handles, legend_labels)


    if show_table:
        table = table_axis.table(
            cellText=table_rows,
            colLabels=[r"$\alpha$", r"$W_2$", f"Final {metric}"],
            cellLoc="center", colLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.5)
        for row, color in enumerate(row_colors, start=1):
            for column in range(3):
                table[row, column].get_text().set_color(color)
        table_axis.set_title(r"$W_2$ distance to ID")
        note = f"Final error: m={training_sizes[-1]}\nRaw initial fields; grid L2 norm"
        table_axis.text(0.5, 0.03, note, ha="center", va="bottom",
                        transform=table_axis.transAxes, fontsize=7)
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.9)
    else:
        figure.tight_layout()

    if should_save_metric_plot(metric):
        figure.savefig(FIGURE_DIR / f"ns_fno3d_{metric}_errors.png", dpi=300, bbox_inches="tight")

    if show_figures:
        plt.show()

    plt.close(figure)


def model_settings():
    """Settings that must match when loading a trained model."""
    return {
        "fno_config": fno_config(),
        "resolution": resolution, "subsample": subsample,
        "time_start": time_start, "number_of_time_steps": number_of_time_steps,
        "viscosity": viscosity, "forcing_amplitude": forcing_amplitude,
        "forcing_wave_number": forcing_wave_number, "final_time": final_time,
        "time_step": time_step, "record_steps": record_steps,
        "training_alpha": training_alpha, "grf_tau": grf_tau,
        "time_padding": time_padding,
    }


def get_model_path(training_size, trial):
    data_name = get_data_path("train", training_size, training_alpha).stem
    config_id = hashlib.sha256(
        json.dumps(model_settings(), sort_keys=True).encode()
    ).hexdigest()[:16]
    return MODEL_DIR / f"{data_name}_cfg{config_id}_trial{trial + 1}.pt"


def save_model(model, input_normalizer, output_normalizer, training_size, trial):
    """Save weights and normalization statistics for independent evaluation."""
    def normalizer_state(normalizer):
        return {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in vars(normalizer).items()
        }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_normalizer": normalizer_state(input_normalizer),
        "output_normalizer": normalizer_state(output_normalizer),
        "settings": model_settings(),
        "training_loss": training_loss,
    }
    file_path = get_model_path(training_size, trial)
    temporary_path = file_path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(file_path)
    log(f"  Saved model: {file_path.name}")


def load_model(training_size, trial, device):
    """Restore weights and normalizers without loading the training dataset."""
    file_path = get_model_path(training_size, trial)
    if not file_path.exists():
        raise FileNotFoundError(f"Model not found: {file_path}. Run with TRAIN_MODEL=True first.")
    with torch.serialization.safe_globals([F.gelu]):
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
    if checkpoint["settings"] != model_settings():
        raise ValueError(f"Current model/data settings differ from the checkpoint: {file_path}")

    # model = FNO3d(checkpoint["settings"]["fno_config"]).to(device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    model = FNO3d(checkpoint["settings"]["fno_config"]).to(device)

    state_dict = checkpoint["model_state_dict"]
    if "_metadata" in state_dict:
        state_dict = state_dict.copy()
        state_dict.pop("_metadata")
    model.load_state_dict(state_dict, strict=True)

    input_normalizer = UnitGaussianNormalizer.__new__(UnitGaussianNormalizer)
    output_normalizer = UnitGaussianNormalizer.__new__(UnitGaussianNormalizer)
    input_normalizer.__dict__.update(checkpoint["input_normalizer"])
    output_normalizer.__dict__.update(checkpoint["output_normalizer"])
    output_normalizer.mean = output_normalizer.mean.to(device)
    output_normalizer.std = output_normalizer.std.to(device)

    log(f"  Loaded model: {file_path.name}")
    return model, input_normalizer, output_normalizer


def save_prediction_file(prediction, training_size, test_alpha, distribution_label, trial):
    """Save a complete predicted test dataset."""
    file_name = (
        f"prediction_N{training_size}_test_alpha{test_alpha:g}_"
        f"{distribution_label}_trial{trial + 1}.mat"
    )
    scipy.io.savemat(PREDICTION_DIR / file_name, {"prediction": prediction.numpy()})


def save_results(results, w2_distances):
    """Save each metric separately so changing EVAL_METRICS preserves earlier results."""
    for metric in EVAL_METRICS:
        arrays = {
            "training_sizes": np.asarray(training_sizes),
            "test_alphas": np.asarray(test_alphas),
            "evaluation_metric": np.asarray(metric),
            "w2_to_id": np.asarray([w2_distances[alpha] for alpha in test_alphas]),
            "w2_norm": np.asarray("sqrt(mean(field**2)); raw initial fields"),
            "w2_resolution": np.asarray(effective_resolution),
        }
        if plot_algebraic_fit:
            arrays["fit_training_sizes"] = np.asarray(fit_training_sizes)

        for alpha in test_alphas:
            alpha_name = str(alpha).replace(".", "p")
            errors = np.asarray(results[metric][alpha])
            mean_errors = errors.mean(axis=1)
            arrays[f"{metric}_alpha_{alpha_name}_raw"] = errors
            arrays[f"{metric}_alpha_{alpha_name}_mean"] = mean_errors
            arrays[f"{metric}_alpha_{alpha_name}_std"] = errors.std(axis=1)
            if plot_algebraic_fit:
                q_estimate, c_estimate, _, _ = fit_algebraic_rate(mean_errors)
                arrays[f"{metric}_fit_q_alpha_{alpha_name}"] = np.asarray(q_estimate)
                arrays[f"{metric}_fit_c_alpha_{alpha_name}"] = np.asarray(c_estimate)

        np.savez_compressed(RESULTS_DIR / f"ns_fno3d_{metric}_results.npz", **arrays)


def run_experiment(device):
    """Train and test one model for every training size and trial."""
    results = {
        metric: {alpha: np.zeros((len(training_sizes), trials)) for alpha in test_alphas}
        for metric in EVAL_METRICS
    }
    maximum_training_size = max(training_sizes)

    for size_index, training_size in enumerate(training_sizes):
        plot_data = {}
        log("\n" + "-" * 70)
        log(f"Training size {size_index + 1}/{len(training_sizes)}: {training_size}")
        log("-" * 70)
        if TRAIN_MODEL:
            train_input, train_output, input_normalizer, output_normalizer = load_training_data(
                training_size
            )
            output_normalizer.mean = output_normalizer.mean.to(device)
            output_normalizer.std = output_normalizer.std.to(device)

        for trial in range(trials):
            log(f"  Trial {trial + 1}/{trials}")
            if TRAIN_MODEL:
                set_seed(seed + trial)
                train_loader = make_loader(train_input, train_output, batch_size, shuffle=True)
                model = FNO3d().to(device)
                log(f"  Model parameters: {count_params(model):,}")
                # train_model(model, train_loader, output_normalizer, device)
                train_model(model,
                            train_loader,
                            output_normalizer,
                            device,
                            training_size,
                            trial,)
                save_model(model, input_normalizer, output_normalizer, training_size, trial)
            else:
                model, input_normalizer, output_normalizer = load_model(
                    training_size, trial, device
                )

            if not EVALUATE_MODEL:
                del model
                continue

            for test_alpha in test_alphas:
                distribution_label = "ID" if test_alpha == training_alpha else "OOD"
                test_input, test_output = load_test_data(test_alpha, input_normalizer)
                test_loader = make_loader(
                    test_input, test_output, loader_batch_size=8, shuffle=False
                )
                prediction, target, test_metrics = test_model(
                    model, test_loader, output_normalizer, device
                )

                for metric, error in test_metrics.items():
                    results[metric][test_alpha][size_index, trial] = error
                    log(
                        f"  {distribution_label:3s} | alpha={test_alpha:g} "
                        f"| {metric}={error:.6e}"
                    )

                if (
                    save_sample_plots
                    and training_size == maximum_training_size
                    and trial == prediction_trial
                ):
                    plot_data[test_alpha] = (target, prediction)

                if save_predictions:
                    save_prediction_file(
                        prediction, training_size, test_alpha, distribution_label, trial
                    )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if EVALUATE_MODEL and save_sample_plots and training_size == maximum_training_size:
            for ood_alpha in test_alphas:
                if ood_alpha != training_alpha:
                    plot_sample_predictions(plot_data, ood_alpha, training_size, prediction_trial)

    return results


def main():
    for folder in (DATA_DIR, FIGURE_DIR, RESULTS_DIR, PREDICTION_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    if GENERATE_DATA:
        if generation_batch_size <= 0 or SAVE_EVERY <= 0:
            raise ValueError("generation_batch_size and SAVE_EVERY must be positive")
        if PARALLEL_GENERATION and NUM_GENERATION_WORKERS <= 0:
            raise ValueError("NUM_GENERATION_WORKERS must be positive")
        if final_time <= 0 or time_step <= 0:
            raise ValueError("final_time and time_step must be positive")
        if resolution <= 0 or resolution & (resolution - 1):
            raise ValueError("resolution must be a positive power of two")
        if not 0 < record_steps <= math.ceil(final_time / time_step):
            raise ValueError("record_steps must fit within the solver steps")
        if not training_sizes or any(size <= 0 for size in training_sizes) or test_size <= 0:
            raise ValueError("Dataset sizes must be positive")
        # Parallel generation uses independent CPU processes; serial mode preserves the original path.
        generation_device = (
            torch.device("cpu")
            if PARALLEL_GENERATION
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        log(f"\nPreparing data on {generation_device}")
        prepare_datasets(generation_device)

    if TRAIN_MODEL or EVALUATE_MODEL:
        validate_settings()
    device = choose_device()
    print_configuration(device)
    if PLOT_ONLY:
        log("\nLoading saved evaluation results")
        results = load_results()

    else:
        if not TRAIN_MODEL and not EVALUATE_MODEL:
            log("\nData preparation finished; training and evaluation disabled")
            return

        log("\n[1/2] Training or loading models; evaluating when enabled")
        results = run_experiment(device)

        if not EVALUATE_MODEL:
            log("\nTraining complete; models saved")
            return

    log("\n[2/2] Saving figures and results")
    w2_distances = calculate_w2_distances()
    for alpha, distance in w2_distances.items():
        log(f"  W2 to ID | alpha={alpha:g} | grid L2={distance:.6e}")

    for metric in EVAL_METRICS:
        if should_save_metric_plot(metric) or show_figures:
            plot_errors(metric, results[metric], w2_distances)

    save_results(results, w2_distances)
    log("\nExperiment complete")
    log(f"Figures    : {FIGURE_DIR}")
    log(f"Results    : {RESULTS_DIR}")
    log(f"Predictions: {PREDICTION_DIR}")
    log(f"Models     : {MODEL_DIR}")


if __name__ == "__main__":
    main()
