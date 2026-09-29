 #!/usr/bin/env python3
"""Plot ALS results and theoretical-rate curves."""

# Standard library
import csv
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy.io import loadmat

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.2,
})

# =============================================================================
# 1. PATHS, OUTPUT DIRECTORIES, AND GLOBAL PLOTTING SETTINGS
# =============================================================================

if "__file__" in globals():
    SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = (
        SCRIPT_PATH.parents[1]
        if SCRIPT_PATH.parent.name == "scripts"
        else SCRIPT_PATH.parent
    )
else:
    PROJECT_ROOT = Path.cwd().resolve()

DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figs"
TABLE_DIR = PROJECT_ROOT / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Plot/display controls.
show_figures = True
maximum_markers = 0

# Fitting control:
# Only data points with m >= FIT_MIN_M are used to estimate the algebraic
# convergence rate. The fitted curve is then evaluated on the full m-grid.
FIT_MIN_M = 80


# =============================================================================
# 2. HELPERS FOR READING AND PROCESSING MATLAB RESULTS
# =============================================================================

def scalar(data, name):
    """Read one MATLAB scalar."""
    return float(np.asarray(data[name]).squeeze())


def vector(data, name):
    """Read one MATLAB vector."""
    return np.asarray(data[name], dtype=float).reshape(-1)


def text(data, name):
    """Read one MATLAB character array."""
    value = data[name]
    if isinstance(value, str):
        return value
    return "".join(np.asarray(value).astype(str).reshape(-1)).strip()


def ood_type_from_file(path):
    """Read the OOD type from the MATLAB filename."""
    if path.stem.endswith("_hypercube"):
        return "hypercube"
    if path.stem.endswith(("_unbounded", "_unbdounded", "_powerinc")):
        return "unbounded"
    return "unknown"


def fit_algebraic_rate(m_values, error_values, fit_min_m=FIT_MIN_M):
    """Fit error ≈ C m^(-q) on the tail m >= fit_min_m."""
    fit_mask = m_values >= fit_min_m

    if np.count_nonzero(fit_mask) < 2:
        raise ValueError(
            f"At least two points with m >= {fit_min_m} are required "
            "to estimate the convergence rate."
        )

    fit_m = m_values[fit_mask]
    fit_error = error_values[fit_mask]

    slope, intercept = np.polyfit(
        np.log(fit_m),
        np.log(fit_error),
        1,
    )

    q_fit = -slope
    c_fit = np.exp(intercept)
    fitted_curve = c_fit * m_values ** (-q_fit)

    return q_fit, c_fit, fitted_curve


def load_one_result(path):
    """Load one ID/OOD case saved by MATLAB and prepare it for plotting."""
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ood_type = ood_type_from_file(path)

    # Select the theoretical quantities stored by MATLAB for this OOD setup.
    if ood_type == "hypercube":
        theory_q_name = "q_ana2"
        theory_curve_name = "conv_the"
    elif ood_type == "unbounded":
        theory_q_name = "q_ana"
        theory_curve_name = "conv_est"
    else:
        raise ValueError(f"Cannot determine OOD type from {path.name}")

    # Common experiment metadata and precomputed plotting arrays.
    result = {
        "path": path,
        "fun_name": text(data, "fun_name"),
        "d": int(round(scalar(data, "d"))),
        "a_val": scalar(data, "a_val"),
        "b_val": scalar(data, "b_val"),
        "w": vector(data, "w"),
        "ood_type": ood_type,
        "m": vector(data, "m_vals_all"),
        "mean": vector(data, "mean_vals_all"),
        "lower": vector(data, "curve_min_vals_all"),
        "upper": vector(data, "curve_max_vals_all"),
        "theory_q": scalar(data, theory_q_name),
        "theory": vector(data, theory_curve_name),
    }

    # Use one common key for the analytical exponent, regardless of setup.
    if ood_type == "hypercube":
        result["q_ana"] = scalar(data, "q_ana2")
    else:
        result["q_ana"] = scalar(data, "q_ana")

    # Basic consistency checks before taking logarithms or fitting.
    number_of_points = result["m"].size
    for name in ("mean", "lower", "upper", "theory"):
        if result[name].size != number_of_points:
            raise ValueError(
                f"{path.name}: {name} has the wrong number of values"
            )

    if (
        np.any(result["m"] <= 0)
        or np.any(result["mean"] <= 0)
        or np.any(result["lower"] <= 0)
        or np.any(result["upper"] <= 0)
        or np.any(result["theory"] <= 0)
    ):
        raise ValueError(f"{path.name}: log-log plot values must be positive")

    # Compute the fitted convergence rate entirely in Python.
    # NOTE: result["mean"] is the MATLAB-saved mean curve used for plotting.
    (
        result["q_fit"],
        result["c_fit"],
        result["fit"],
    ) = fit_algebraic_rate(
        result["m"],
        result["mean"],
        fit_min_m=FIT_MIN_M,
    )

    # The table currently reports q_ana as the theoretical rate.
    # Retain the older alternative formula below for reference.
    # if ood_type == "unbounded":
    result["q_theory"] = result["q_ana"]


    # hypercube: b=1 is ID; unbounded/powerinc: b=0 is ID.
    result["is_id"] = np.allclose(result["w"], 1.0)
    return result


def load_all_results():
    """Load and group all MATLAB files."""
    paths = sorted(DATA_DIR.glob("ALS_*.mat"))
    if not paths:
        raise FileNotFoundError(
            f"No ALS result files found in {DATA_DIR}. "
            "Run the MATLAB file first."
        )

    groups = {}
    for path in paths:
        result = load_one_result(path)
        key = (
            result["fun_name"],
            result["d"],
            result["a_val"],
            result["ood_type"],
        )
        groups.setdefault(key, []).append(result)
        print(f"Loaded {path}")

    return groups


# =============================================================================
# 3. LABELS AND FIGURE GENERATION
# =============================================================================

# -------------------------------------------------------------------------
# Legacy label format retained for reference.
# -------------------------------------------------------------------------

# Current legend label format.
def distribution_label(result):
    if result["is_id"]:
        if result["ood_type"] == "unbounded":
            return r"\textsc{ID}, $b=0$"
        elif result["ood_type"] == "hypercube":
            return r"\textsc{ID}, $\omega=1$"
    elif result["ood_type"] == "hypercube":
        return rf"\textsc{{OOD}}, $\omega={result['b_val']:g}$"
    else:
        return rf"\textsc{{OOD}}, $b={result['b_val']:g}$"
    
def plot_final_errors(key, results):
    """Plot ALS error curves, uncertainty bands, and fitted reference curves."""
    fun_name, dimension, a_val, ood_type = key
    results = sorted(
        results,
        key=lambda result: (not result["is_id"], result["b_val"]),
    )

    # ---------------------------------------------------------------------
    # Legacy theoretical-curve selection.
    #
    # This block is currently not used by the active plotting branch below,
    # but it is intentionally retained because the commented plotting option
    # still refers to bottom_theory_result.
    # ---------------------------------------------------------------------
    # If several hypercube theoretical curves have q_ana = -1, draw only
    # the bottom one. All solid ALS curves are still drawn.
    q_minus_one = (
        [
            result
            for result in results
            if np.isclose(result["q_ana"], -1.0)
        ]
        if ood_type == "hypercube"
        else []
    )
    bottom_theory_result = (
        min(q_minus_one, key=lambda result: result["theory"][-1])
        if q_minus_one
        else None
    )

    # ---------------------------------------------------------------------
    # Create figure and legend containers.
    # ---------------------------------------------------------------------
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    legend_handles = []
    legend_labels = []

    # ---------------------------------------------------------------------
    # Plot one solid ALS curve and one dotted fitted curve per distribution.
    # ---------------------------------------------------------------------
    for result in results:
        marker = "o" if result["is_id"] else "s"

        # Show only a small, evenly spaced subset of markers. The line still
        # uses every saved data point.
        if maximum_markers > 0:
            marker_indices = np.unique(
                np.linspace(
                    0,
                    result["m"].size - 1,
                    min(maximum_markers, result["m"].size),
                    dtype=int,
                )
            )
        else:
            marker_indices = []

        # Solid line: ALS result.
        error_line = axis.plot(
            result["m"],
            result["mean"],
            marker=marker,
            markevery=marker_indices,
            linestyle="-",
            linewidth=2,
        )[0]
        color = error_line.get_color()

        # Shaded band: one geometric standard deviation around the mean.
        axis.fill_between(
            result["m"],
            np.maximum(result["lower"], 1e-16),
            result["upper"],
            color=color,
            alpha=0.2,
        )

        # -----------------------------------------------------------------
        # Dotted reference curve.
        #
        # Active behavior: use the fitted curve for every case.
        # Legacy behavior is retained below as comments.
        # -----------------------------------------------------------------
        reference_curve = result["fit"]
        show_reference = True

        reference_line = None
        if show_reference:
            reference_line = axis.plot(
                result["m"],
                reference_curve,
                linestyle=":",
                linewidth=2,
                color=color,
            )[0]

        label = distribution_label(result)
        legend_handles.append(error_line)
        legend_labels.append(label)

    # ---------------------------------------------------------------------
    # Axes and grid.
    # ---------------------------------------------------------------------
    axis.set_xlabel(r"Number of training samples, $m$")
    axis.set_ylabel(r"Relative $L^\infty$ error")
    # axis.set_title(r"ALS: ID/OOD errors")
    axis.set_xscale("log")
    axis.set_yscale("log")

    all_m_values = np.concatenate([result["m"] for result in results])
    axis.set_xlim(all_m_values.min(), all_m_values.max())

    axis.grid(True, which="both", linestyle="--", linewidth=0.5)

    # ---------------------------------------------------------------------
    # Legend.
    #
    # theory_description is retained because the older legend construction
    # below still uses it.
    # ---------------------------------------------------------------------

    theory_description = (
        rf"dotted: $C_{{fit}}m^{{-q_{{fit}}}}$, fit on $m\geq {FIT_MIN_M}$"
    )
    fit_handle = Line2D(
        [0], [0],
        linestyle=":",
        linewidth=2.2,
        color="black",
    )

    legend_handles.append(fit_handle)
    legend_labels.append(r"$C_{\mathrm{fit}}m^{-q_{\mathrm{fit}}}$")

    axis.legend(
        legend_handles,
        legend_labels,
        handlelength=3.0,
        loc="lower left",
        frameon=True,
        fontsize=14,
    )

    # ---------------------------------------------------------------------
    # Save/display figure.
    # ---------------------------------------------------------------------
    figure.tight_layout()

    a_tag = f"{a_val:g}".replace(".", "p").replace("-", "m")
    output_path = FIGURE_DIR / (
        f"ALS_{fun_name}_d{dimension}_a{a_tag}_{ood_type}.png"
    )
    figure.savefig(output_path, dpi=400, bbox_inches="tight")

    if show_figures:
        plt.show()
    plt.close(figure)
    print(f"Saved {output_path}")


# =============================================================================
# 4. FIT-RATE TABLE OUTPUT
# =============================================================================

def save_fit_tables(groups):
    """Save fitted convergence rates for unbounded and hypercube cases."""

    for ood_type in ("unbounded", "hypercube"):

        rows = []

        # Collect one row per saved ID/OOD experiment.
        for results in groups.values():
            for result in results:

                if result["ood_type"] != ood_type:
                    continue

                parameter_name = "omega" if ood_type == "hypercube" else "b"

                rows.append(
                    {
                        "function": result["fun_name"],
                        "d": result["d"],
                        "a": result["a_val"],
                        parameter_name: result["b_val"],
                        "distribution": "ID" if result["is_id"] else "OOD",
                        "fit_min_m": FIT_MIN_M,
                        "q_fit": result["q_fit"],
                        "q_theory": result["q_theory"],
                    }
                )

        if not rows:
            continue

        parameter_name = "omega" if ood_type == "hypercube" else "b"

        rows.sort(
            key=lambda row: (
                row["function"],
                row["d"],
                row["a"],
                row[parameter_name],
            )
        )

        # Write one CSV per OOD geometry.
        output_path = TABLE_DIR / f"ALS_{ood_type}_fit_slopes.csv"

        fieldnames = [
            "function",
            "d",
            "a",
            parameter_name,
            "distribution",
            "fit_min_m",
            "q_fit",
            "q_theory",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "q_fit": f"{row['q_fit']:.6f}",
                        "q_theory": f"{row['q_theory']:.6f}",
                    }
                )

        print(f"\n{ood_type.capitalize()} fit slopes")
        print(
            f"{'Function':24s} {'d':>3s} "
            f"{parameter_name:>9s} {'m_min':>7s} "
            f"{'q_fit':>12s} {'q_theory':>12s}"
        )

        for row in rows:
            print(
                f"{row['function']:24s} "
                f"{row['d']:3d} "
                f"{row[parameter_name]:9.3g} "
                f"{row['fit_min_m']:7d} "
                f"{row['q_fit']:12.6f} "
                f"{row['q_theory']:12.6f}"
            )

        print(f"Saved {output_path}")


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    groups = load_all_results()
    for key, results in groups.items():
        plot_final_errors(key, results)
    save_fit_tables(groups)


if __name__ == "__main__":
    main()
