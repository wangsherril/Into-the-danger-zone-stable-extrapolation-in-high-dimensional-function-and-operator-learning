import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from matplotlib.legend_handler import HandlerTuple
import math

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


# If using google colab, switch the directory to the project folder.
# from google.colab import drive
# drive.mount("/content/drive")
# Project_ROOT = Path("/content/drive/MyDrive/OOD_numerical_experiments/NN") 

from pathlib import Path
import sys

# find the project root
if "__file__" in globals():
    SCRIPT_PATH = Path(__file__).resolve()  # current python file

    PROJECT_ROOT = (
        SCRIPT_PATH.parents[1]             # if the file is in scripts
        if SCRIPT_PATH.parent.name == "scripts"
        else SCRIPT_PATH.parent            # if the file is in the project root
    )
else:
    PROJECT_ROOT = Path.cwd().resolve()     # for colab

# let python find modules in the project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import the target function
from utils.functions import get_target_function

# folders for figures and results
FIGURE_DIR = PROJECT_ROOT / "fig" / "DNN"
RESULTS_DIR = PROJECT_ROOT / "results" / "DNN"

# create the folders if they do not exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================= 
# 2. Parameters & SETUP
# ============================================= 

target_name =  "piston_scaled_7d"

'''
List of functions for the virtual library
    "robotsq_scaled_8d": TargetFunction("robotsq_scaled_8d", 8, robotsq_scaled_8d),
    "wingweight_scaled_10d": TargetFunction("wingweight_scaled_10d", 10, wingweight_scaled_10d),
    "circuit_scaled_6d": TargetFunction("circuit_scaled_6d", 6, circuit_scaled_6d),
    "piston_scaled_7d": TargetFunction("piston_scaled_7d", 7, piston_scaled_7d)
'''

# Training sample sizes
sample_sizes = [10 * 2**k for k in range(8)] #  [10, 20, 40, 80, 160, 320, 640, 1280]
wasserstein_p = 2

# NN settings
depths = [15] # Number of hidden layers.
width_multiplier = 10 # Width = depth × width_multiplier.
epochs = 2000
learning_rate = 1e-4

# Training loss options: "rel_l2", "abs_l2", or "mse"
training_loss = "rel_l2"

# Evaluation metric options: "rel_l2", "linf", or "mse"
evaluation_metric = "rel_l2"
# Repeated experiments and testing
trials = 30
test_size = 10000 # Testing sample sizes for both ID and OOD
ood_scales = [1.1,1.2,1.4]  # OOD domains are [-scale, scale]^d. The ID domain corresponds to scale=1.


# Training and ID inputs range
box_low = -1
box_high = 1

# Reproducibility and others
seed = 2026
show_figures = True
show_table = True
verbose = True
history_trial = 1 # Record epoch history for the desinated trial. Now, the first trial.

# Data settings
generate_new_data = False

# If False, skip training and load saved results directly for plotting.
train_models = False

fit_min_m = 80


DATA_DIR = PROJECT_ROOT / "data" / "DNN"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / f"{target_name}_dataset.npz"

MODEL_DIR = RESULTS_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================= 
# 3. Helper functions
# ============================================= 

def log(message=""):
    """Print only when verbose=True."""
    if verbose:
        print(message, flush=True)

def choose_device():
    """Use CUDA or CPU, in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(value):
    """Make an experiment reproducible."""
    np.random.seed(value)          # seed for numpy
    torch.manual_seed(value)       # seed for pytorch on CPU

    if torch.cuda.is_available():  # check if GPU is available
        torch.cuda.manual_seed_all(value)  # seed for all GPUs


def validate_settings():
    """Stop early when a setting is invalid."""
    if not sample_sizes or any(m <= 0 for m in sample_sizes):
        raise ValueError("sample_sizes must contain positive numbers")
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("depths must contain positive numbers")
    if width_multiplier <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("width_multiplier, epochs, and learning_rate must be positive")
    if training_loss not in {"rel_l2", "abs_l2", "mse"}:
        raise ValueError(
            "training_loss must be 'rel_l2', 'abs_l2', or 'mse'"
        )

    if evaluation_metric not in {"rel_l2", "linf", "mse"}:
        raise ValueError(
            "evaluation_metric must be 'rel_l2', 'linf', or 'mse'"
        )
    if trials <= 0 or test_size <= 0:
        raise ValueError("trials and test_size must be positive")
    if box_low >= box_high:
        raise ValueError("box_low must be smaller than box_high")
    if any(scale <= 1 for scale in ood_scales):
        raise ValueError("Every OOD scale must be greater than 1")


def make_random_data(target, number_of_samples, dimension, low, high, device, data_seed):
    """sample x uniformly from the domain and calculate y=target(x)"""
    generator= torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    # generate random inputs in [low, high]
    x = low + (high - low) * torch.rand(
        number_of_samples,
        dimension,
        generator=generator,
    )
    x = x.to(device)

    # evaluate the target function without gradients
    with torch.no_grad():
        y = target(x)

    # change (n,) to (n, 1)
    if y.ndim == 1:
        y = y[:, None]

    # check the output shape
    expected_shape = (number_of_samples, 1)
    if y.shape != expected_shape:
        raise ValueError(
            f"Target returned shape {tuple(y.shape)}; expected {expected_shape}"
        )

    return x, y

# ============================================= 
# 4. Model and evaluation
# ============================================= 

def make_model(dimension, depth, width, device):
    """Create a fully connected network with tanh activations."""

    layers = []                 # store all network layers
    input_width = dimension     # number of input features

    # create hidden layers
    for _ in range(depth):
        layers.append(nn.Linear(input_width, width))  # linear layer
        layers.append(nn.Tanh())                      # activation function
        input_width = width                           # input size of the next layer

    # create the output layer
    layers.append(nn.Linear(width, 1))

    # combine all layers and move the model to CPU or GPU
    return nn.Sequential(*layers).to(device)


def absolute_l2(prediction, target):
    """return l2:||prediction-target||_2."""
    return torch.linalg.vector_norm(prediction.reshape(-1) - target.reshape(-1))

def relative_l2(prediction, target):
    """return relative l2: ||prediction-target||_2 / ||target||_2."""
    numerator = absolute_l2(prediction, target)
    denominator = torch.linalg.vector_norm(target.reshape(-1)).clamp_min(1e-12)
    return numerator / denominator

def mse_loss(prediction, target):
    """Return mean squared error."""
    return torch.mean((prediction - target) ** 2)


def linf_error(prediction, target):
    """Return ||prediction - target||_infinity."""
    return torch.max(
        torch.abs(prediction.reshape(-1) - target.reshape(-1))
    )

def selected_training_loss(prediction, target):
    """Select the training loss specified in the settings section."""
    if training_loss == "abs_l2":
        return absolute_l2(prediction, target)

    if training_loss == "mse":
        return mse_loss(prediction, target)

    return relative_l2(prediction, target)

def selected_evaluation_metric(prediction, target):
    """Select the evaluation metric specified in the settings section."""
    if evaluation_metric == "linf":
        return linf_error(prediction, target)

    if evaluation_metric == "mse":
        return mse_loss(prediction, target)

    return relative_l2(prediction, target)

def train_model(model, x_train, y_train, x_id, y_id, record_history):
    """train one model and record error for every epoch"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.999) # final learning rate is 0.00001352
    history = {"train": [], "id_test": []}
    print_every = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = selected_training_loss(prediction, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if record_history: #record error for every epoch
            model.eval()
            with torch.no_grad():
                history["train"].append(
                    mse_loss(model(x_train), y_train).item()
                )
                history["id_test"].append(
                    mse_loss(model(x_id), y_id).item()
                )

        epoch_number = epoch + 1
        if epoch_number == 1 or epoch_number % print_every == 0 or epoch_number == epochs:
            log(
                f"    epoch {epoch_number:>5}/{epochs:<5} "
                f"| {training_loss}={loss.item():.4e}"
            )

    return history


def test_model(model, x, y):
    """Evaluate a trained model using the selected evaluation metric."""
    model.eval()

    with torch.no_grad():
        prediction = model(x)
        error = selected_evaluation_metric(prediction, y)

    return error.item()


# ============================================= 
# 5. Dataset
# ============================================= 
def save_dataset(
    x_train_all,
    y_train_all,
    x_id,
    y_id,
    ood_data,
):
    arrays = {
        "x_train": x_train_all.cpu().numpy(),
        "y_train": y_train_all.cpu().numpy(),
        "x_id": x_id.cpu().numpy(),
        "y_id": y_id.cpu().numpy(),
    }

    for scale, (x, y) in ood_data.items():
        name = str(scale).replace(".", "p")
        arrays[f"x_ood_{name}"] = x.cpu().numpy()
        arrays[f"y_ood_{name}"] = y.cpu().numpy()

    np.savez_compressed(DATA_FILE, **arrays)

def load_dataset(device):
    data = np.load(DATA_FILE)

    x_train_all = torch.from_numpy(data["x_train"]).float().to(device)
    y_train_all = torch.from_numpy(data["y_train"]).float().to(device)

    x_id = torch.from_numpy(data["x_id"]).float().to(device)
    y_id = torch.from_numpy(data["y_id"]).float().to(device)

    ood_data = {}

    for scale in ood_scales:
        name = str(scale).replace(".", "p")

        x = torch.from_numpy(data[f"x_ood_{name}"]).float().to(device)
        y = torch.from_numpy(data[f"y_ood_{name}"]).float().to(device)

        ood_data[scale] = (x, y)

    return x_train_all, y_train_all, x_id, y_id, ood_data


# Fit
def fit_algebraic_rate(mean_vals):

    """Fit error ~ c * m^(-q) using sample sizes m >= fit_min_m."""
    m_vals = np.asarray(sample_sizes, dtype=float)
    mean_vals = np.asarray(mean_vals, dtype=float)

    idx = np.where(m_vals >= fit_min_m)[0]

    if len(idx) < 2:
        raise ValueError(
            "At least two points with m >= fit_min_m are required"
        )

    p = np.polyfit(
        np.log10(m_vals[idx]),
        np.log10(mean_vals[idx]),
        1,
    )

    q_est = -p[0]
    c_est = 10 ** p[1]
    fitted_vals = c_est * m_vals ** (-q_est)

    return q_est, c_est, fitted_vals

# ============================================= 
# 6. Plotting
# ============================================= 

def positive_lower_edge(mean, std):
    """Draw the shaded error band positive on a log axis"""
    return np.maximum(mean - std, 1e-16)

def plot_errors(depth, results):
    legend_handles = []
    legend_labels = []
    if evaluation_metric == "mse":
        ylabel = "MSE"
        metric_title = "MSE"
    elif evaluation_metric == "rel_l2":
        ylabel = r"Relative $L^2$ error"
        metric_title = r"Relative $L^2$"
    else:
        ylabel = r"$L^\infty$ error"
        metric_title = r"$L^\infty$"

    """plot ID and all OOD mean errors with +- one standard deviation with color band"""
    x = np.asarray(sample_sizes)

    if show_table:
        fig = plt.figure(figsize=(11, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.2], wspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        table_ax = fig.add_subplot(gs[0, 1])
        table_ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    line_id, = ax.plot(
        x,
        results["id_mean"],
        marker="o",
        linewidth=2,
    )

    q_id, c_id, id_fit = fit_algebraic_rate(results["id_mean"])

    fit_id, = ax.plot(
        x,
        id_fit,
        linestyle="--",
        color=line_id.get_color(),
    )

    legend_handles.append((line_id, fit_id))
    legend_labels.append(
        fr"ID, $\theta=0.0$, ${c_id:.2f}m^{{-{q_id:.2f}}}$"
    )

    ax.fill_between(
        x,
        positive_lower_edge(results["id_mean"], results["id_std"]),
        results["id_mean"] + results["id_std"],
        color=line_id.get_color(),
        alpha=0.2,
    )

    for scale in ood_scales:
        mean = results["ood_mean"][scale]
        std = results["ood_std"][scale]

        line_ood, = ax.plot(
            x,
            mean,
            marker="s",
            linewidth=2,
        )

        q_ood, c_ood, ood_fit = fit_algebraic_rate(mean)

        fit_ood, = ax.plot(
            x,
            ood_fit,
            linestyle="--",
            color=line_ood.get_color(),
        )

        legend_handles.append((line_ood, fit_ood))
        legend_labels.append(
            fr"OOD, $w={scale:g}$, ${c_ood:.2f}m^{{-{q_ood:.2f}}}$"
        )
        # std filled color band
        ax.fill_between(
            x,
            positive_lower_edge(mean, std),
            mean + std,
            color=line_ood.get_color(),
            alpha=0.2,
        )

 
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [str(value) for value in sample_sizes])
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel(ylabel)
    # ax.set_title(
    #     f"ID/OOD {metric_title} errors: "
    #     f"depth={depth}, width={depth * width_multiplier}"
    # )
    # ax.set_title(f"ID/OOD errors: depth={depth}, width={depth * width_multiplier}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(
    legend_handles,
    legend_labels,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    if show_table:
        target = get_target_function(target_name)
        dimension = target.domain_dim

        table_data = [["ID", "0.0000", f"{results['id_mean'][-1]:.4e}"]]
        for scale in ood_scales:
            wp = abs(scale - 1.0) * math.sqrt(
                dimension / 3.0
            )
            table_data.append(
                [f"{scale:g}", f"{wp:.4f}", f"{results['ood_mean'][scale][-1]:.4e}"]
            )

        table = table_ax.table(
            cellText=table_data,
            colLabels=[r"$w$", r"$W_2$", r"Final MSE"],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.5)
        table_ax.set_title(r"$W_2$ distance to ID")

    fig.tight_layout()
    fig.savefig(
    FIGURE_DIR / f"id_ood_{training_loss}_depth_{depth}_{target_name}.png",
    dpi=300,
    bbox_inches="tight",
    )

    if show_figures:
        plt.show()
    plt.close(fig)


def plot_history(depth, history):
    """Plot training and ID test error across epochs for the training set with max(n_train)"""
    epoch_numbers = np.arange(1, epochs+1)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.semilogy(epoch_numbers, history["train"], label="Training error")
    ax.semilogy(epoch_numbers, history["id_test"], label="ID test error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"Training history: m={max(sample_sizes)}, depth={depth}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"history_depth_{depth}_{training_loss}_{target_name}.png", dpi=300)
    if show_figures:
        plt.show()
    plt.close(fig)

def save_results(depth, results, history):
    """save plotted numbers so the experiment does not need to be """
    arrays = {
        "sample_sizes": np.asarray(sample_sizes),
        "id_mean": results["id_mean"],
        "id_std": results["id_std"],
        "history_train": np.asarray(history["train"]),
        "history_id_test": np.asarray(history["id_test"]),
    }

    for scale in ood_scales:
        name = str(scale).replace(".", "p")
        arrays[f"ood_w{name}_mean"] = results["ood_mean"][scale]
        arrays[f"ood_w{name}_std"] = results["ood_std"][scale]

    np.savez_compressed(
        RESULTS_DIR
        / f"results_{evaluation_metric}_depth_{depth}_{target_name}.npz",
        **arrays,
    )

def load_results(depth):
    """Load saved results so figures can be remade without retraining."""
    result_file = (
        RESULTS_DIR
        / f"results_{evaluation_metric}_depth_{depth}_{target_name}.npz"
    )

    if not result_file.exists():
        raise FileNotFoundError(
            f"Saved result file not found: {result_file}"
        )

    data = np.load(result_file)

    if not np.array_equal(data["sample_sizes"], np.asarray(sample_sizes)):
        raise ValueError(
            "Saved sample_sizes do not match the current sample_sizes"
        )

    results = {
        "id_mean": data["id_mean"],
        "id_std": data["id_std"],
        "ood_mean": {},
        "ood_std": {},
    }

    for scale in ood_scales:
        name = str(scale).replace(".", "p")
        results["ood_mean"][scale] = data[f"ood_w{name}_mean"]
        results["ood_std"][scale] = data[f"ood_w{name}_std"]

    history = {
        "train": data["history_train"],
        "id_test": data["history_id_test"],
    }

    return results, history

from scipy.optimize import linear_sum_assignment


# ============================================= 
# Wasserstein metric 


def empirical_wp(x, y, p, max_samples):
    """
    Empirical Wp between two independently sampled,
    equally weighted distributions in d dimensions.
    """
    if p < 1:
        raise ValueError("p must be at least 1")

    n = min(
        x.shape[0],
        y.shape[0],
        max_samples,
    )

    x_index = torch.randperm(
        x.shape[0],
        device=x.device,
    )[:n]

    y_index = torch.randperm(
        y.shape[0],
        device=y.device,
    )[:n]

    x_small = x[x_index].detach().cpu()
    y_small = y[y_index].detach().cpu()

    # Pairwise Euclidean distances.
    distance = torch.cdist(
        x_small,
        y_small,
        p=2,
    ).numpy()

    # Cost matrix: ||x-y||_2^p.
    cost = distance ** p

    # Optimal one-to-one assignment.
    row_index, column_index = linear_sum_assignment(cost)

    wp_power = cost[
        row_index,
        column_index,
    ].mean()

    return wp_power ** (1.0 / p)

# ============================================= 
# 7. Main experiment
# ============================================= 

def main():

    validate_settings()
    set_seed(seed)
    device = choose_device()

    target = get_target_function(target_name)
    dimension = target.domain_dim # read dimension

    log("\n" + "=" * 70)
    log("DNN ID/OOD EXPERIMENT")
    log("=" * 70)
    log(f"Evaluation      : {evaluation_metric}")
    log(f"Target          : {target_name}")
    log(f"Dimension       : {dimension}")
    log(f"Device          : {device}")
    log(f"Sample sizes    : {sample_sizes}")
    log(f"Depths          : {depths}")
    log(f"OOD scales      : {ood_scales}")
    log(f"Epochs/trials   : {epochs}/{trials}")
    log(f"Training loss   : {training_loss}")
    log("=" * 70)
    max_train = max(sample_sizes)

    # if not train_models:
    #     for depth in depths:
    #         results, saved_history = load_results(depth)
    #         plot_errors(depth, results)
    #         plot_history(depth, saved_history)

    #     log("\nLoaded saved results and regenerated figures")
    #     log(f"Figures: {FIGURE_DIR}")
    #     log(f"Results: {RESULTS_DIR}")
    #     return

    if generate_new_data:

        log("Generating dataset...")

        x_train_all, y_train_all = make_random_data(
            target,
            max_train,
            dimension,
            box_low,
            box_high,
            device,
            seed,
        )

        x_id, y_id = make_random_data(
            target,
            test_size,
            dimension,
            box_low,
            box_high,
            device,
            seed + 10000,
        )

        ood_data = {}

        for index, scale in enumerate(ood_scales):
            ood_data[scale] = make_random_data(
                target,
                test_size,
                dimension,
                scale * box_low,
                scale * box_high,
                device,
                seed + 20000 + index,
            )

        save_dataset(
            x_train_all,
            y_train_all,
            x_id,
            y_id,
            ood_data,
        )

    else:

        log(f"Loading dataset from {DATA_FILE}")
        x_train_all, y_train_all, x_id, y_id, ood_data = load_dataset(device)



    # If one wants to compare models with different depths, with width = depth * 10
    for depth in depths:
        width = depth * width_multiplier
        log(f"\nDepth={depth}, width={width}")

        # Rows correspond to sample sizes; columns correspond to trials.
        id_errors = np.zeros((len(sample_sizes), trials))
        ood_errors = {
            scale: np.zeros((len(sample_sizes), trials))
            for scale in ood_scales
        }
        saved_history = None

        for sample_index, number_of_samples in enumerate(sample_sizes):
            log(f"\n  Training samples: {number_of_samples}")
            for trial in range(trials):
                log(f"  Trial {trial + 1}/{trials}")

                # Training data are identical across depths for fair comparison.
                data_seed = seed + 1000 * trial + sample_index
                model_seed = seed + 100_000 * depth + 1000 * trial + sample_index

                # x_train, y_train = make_random_data(
                #     target,
                #     number_of_samples,
                #     dimension,
                #     box_low,
                #     box_high,
                #     device,
                #     data_seed,
                # )
                x_train = x_train_all[:number_of_samples]
                y_train = y_train_all[:number_of_samples]

                set_seed(model_seed)
                model = make_model(dimension, depth, width, device)

                if train_models:
                    record_history = (
                        number_of_samples == max(sample_sizes)
                        and trial == history_trial
                    )

                    history = train_model(
                        model,
                        x_train,
                        y_train,
                        x_id,
                        y_id,
                        record_history,
                    )
                    model_path = (
                        MODEL_DIR
                        / (
                            f"model_depth_{depth}_m_{number_of_samples}_"
                            f"trial_{trial}_{target_name}.pt"
                        )
                    )
                    torch.save(model.state_dict(), model_path)
                    log(f"    Saved model: {model_path.name}")

                    if record_history:
                        saved_history = history

                else:
                    model_path = (
                        MODEL_DIR
                        / (
                            f"model_depth_{depth}_m_{number_of_samples}_"
                            f"trial_{trial}_{target_name}.pt"
                        )
                    )

                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Saved model not found: {model_path}"
                        )

                    state_dict = torch.load(
                        model_path,
                        map_location=device,
                        weights_only=True,
                    )

                    model.load_state_dict(state_dict)
                    model.eval()
                id_errors[sample_index, trial] = test_model(model, x_id, y_id)
                # log(f"    ID relative L2: {id_errors[sample_index, trial]:.4e}")
                log(
                    f"    ID {evaluation_metric}: "
                    f"{id_errors[sample_index, trial]:.4e}"
                )

                for scale, (x_ood, y_ood) in ood_data.items():
                    error = test_model(model, x_ood, y_ood)
                    ood_errors[scale][sample_index, trial] = error
                    log(
                        f"    OOD w={scale:g} {evaluation_metric}: "
                        f"{error:.4e}"
                    )
                del model, x_train, y_train


        if train_models:
            if saved_history is None:
                raise RuntimeError("Training history was not recorded")
        else:
            result_file = (
                RESULTS_DIR
                / f"results_{evaluation_metric}_depth_{depth}_{target_name}.npz"
            )

            if not result_file.exists():
                raise FileNotFoundError(
                    f"Saved result file not found: {result_file}"
                )

            with np.load(result_file) as data:
                saved_history = {
                    "train": data["history_train"],
                    "id_test": data["history_id_test"],
                }
        results = {
            # Mean ID error over all trials for each sample size.
            # Shape: (n_sample_sizes,)
            "id_mean": id_errors.mean(axis=1),

            # Standard deviation of ID error over all trials.
            # Measures the variability across repeated experiments.
            "id_std": id_errors.std(axis=1),

            # Mean OOD error over all trials for each OOD scale.
            # Returns a dictionary:
            # {scale -> mean error array}
            "ood_mean": {
                scale: values.mean(axis=1)
                for scale, values in ood_errors.items()
            },

            # Standard deviation of OOD error over all trials
            # for each OOD scale.
            # Returns a dictionary:
            # {scale -> std error array}
            "ood_std": {
                scale: values.std(axis=1)
                for scale, values in ood_errors.items()
            },
        }
        plot_errors(depth, results)
        plot_history(depth, saved_history)
        save_results(depth, results, saved_history)

    log("\nExperiment complete")
    log(f"Figures: {FIGURE_DIR}")
    log(f"Results: {RESULTS_DIR}")

def calculate_wasserstein():
    # Wasserstein distance

    target = get_target_function(target_name)
    dimension = target.domain_dim # read dimension
    device = choose_device()

    n_wp = 100
    for scale in ood_scales:
        if wasserstein_p == 2:
            wp = abs(scale - 1.0) * math.sqrt(
                dimension / 3.0
            )
            method = "analytical"

        else:
            x = 2 * torch.rand(
                n_wp,
                dimension,
                device=device,
            ) - 1

            y = scale * (
                2 * torch.rand(
                    n_wp,
                    dimension,
                    device=device,
                ) - 1
            )

            wp = empirical_wp(
                x,
                y,
                p=wasserstein_p,
                max_samples=n_wp,
            )
            method = "empirical"

        log(
            f"scale={scale:g}, "
            f"{method} W{wasserstein_p}={wp:.6e}"
        )
        
if __name__ == "__main__":
    main()
