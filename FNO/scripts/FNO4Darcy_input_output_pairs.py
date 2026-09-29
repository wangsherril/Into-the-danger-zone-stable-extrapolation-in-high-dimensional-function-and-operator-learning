"""Plot representative ID/OOD Darcy input-output pairs.

The plotting logic is kept separate from the training script. Pass the
experiment's existing parameters into ``plot_darcy_distribution_pairs`` so
there is only one source of truth for the configuration.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def generate_cookie_field(resolution, radius):
    """Generate one COOKIE permeability field."""
    grid = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    x1, x2 = np.meshgrid(grid, grid, indexing="ij")
    parameters = np.random.uniform(-1.0, 1.0, size=8).astype(np.float32)
    permeability = np.ones((resolution, resolution), dtype=np.float32)

    for parameter, (center_x, center_y) in zip(parameters, COOKIE_CENTERS):
        circle = (
            (x1 - center_x) ** 2 + (x2 - center_y) ** 2 <= radius**2
        )
        permeability[circle] = 0.625 - 0.375 * parameter

    return permeability


def generate_permeability_field(
    grf_sampler,
    resolution,
    tau,
    alpha,
    distribution,
    radius=None,
):
    """Generate one LN, PC, or COOKIE permeability field."""
    if distribution == "COOKIE":
        if radius is None:
            raise ValueError("radius is required for COOKIE samples")
        return generate_cookie_field(resolution, radius)

    log_permeability = grf_sampler(alpha, tau, resolution)
    if distribution == "LN":
        permeability = np.exp(log_permeability)
    elif distribution == "PC":
        permeability = np.where(log_permeability >= 0.0, 12 / 6, 3 / 6)
    else:
        raise ValueError("distribution must be 'LN', 'PC', or 'COOKIE'")

    return np.asarray(permeability, dtype=np.float32)


def plot_darcy_distribution_pairs(
    *,
    grf_sampler,
    darcy_solver,
    resolution,
    id_alpha,
    ood_alphas,
    cookie_radii,
    tau,
    seed,
    figure_dir,
    number_of_samples=1,
    ood_alpha=None,
    cookie_radius=None,
    dpi=150,
    show=False,
):
    """Plot corresponding inputs and outputs for the ID/OOD distributions.

    Parameters come directly from the main experiment. By default, the figure
    uses the smallest configured OOD alpha and the first COOKIE radius.
    Plotting uses an isolated NumPy random stream and therefore does not change
    subsequent data generation or model-training randomness.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be positive")
    if not ood_alphas or any(alpha <= 0 for alpha in ood_alphas):
        raise ValueError("ood_alphas must contain positive values")
    if not cookie_radii or any(radius <= 0 for radius in cookie_radii):
        raise ValueError("cookie_radii must contain positive radii")

    selected_ood_alpha = min(ood_alphas) if ood_alpha is None else ood_alpha
    selected_cookie_radius = (
        cookie_radii[0] if cookie_radius is None else cookie_radius
    )
    if selected_ood_alpha <= 0 or selected_cookie_radius <= 0:
        raise ValueError("Selected OOD alpha and COOKIE radius must be positive")

    specifications = [
        {
            "distribution": "LN",
            "alpha": id_alpha,
            "radius": None,
        },
        {
            "distribution": "LN",
            "alpha": selected_ood_alpha,
            "radius": None,
        },
        {
            "distribution": "COOKIE",
            "alpha": id_alpha,
            "radius": selected_cookie_radius,
        },
        {
            "distribution": "PC",
            "alpha": selected_ood_alpha,
            "radius": None,
        },
    ]

    # Preserve the experiment's RNG state so creating the figure cannot alter
    # later datasets, model initializations, or trials.
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed + 100_000)
        inputs_by_distribution = []
        outputs_by_distribution = []
        forcing = np.ones((resolution, resolution), dtype=np.float32)
        for specification in specifications:
            inputs = []
            outputs = []
            for _ in range(number_of_samples):
                permeability = generate_permeability_field(
                    grf_sampler,
                    resolution,
                    tau,
                    specification["alpha"],
                    specification["distribution"],
                    radius=specification["radius"],
                )
                solution = darcy_solver(permeability, forcing)
                inputs.append(permeability)
                outputs.append(np.asarray(solution, dtype=np.float32))

            inputs_by_distribution.append(np.stack(inputs))
            outputs_by_distribution.append(np.stack(outputs))
    finally:
        np.random.set_state(numpy_state)

    figure, axes = plt.subplots(
        2 * number_of_samples,
        len(specifications),
        figsize=(14, 5.0 * number_of_samples),
        squeeze=False,
    )

    output_min = min(float(fields.min()) for fields in outputs_by_distribution)
    output_max = max(float(fields.max()) for fields in outputs_by_distribution)
    if np.isclose(output_min, output_max):
        output_max = output_min + 1.0

    output_image = None
    for column, (specification, inputs, outputs) in enumerate(
        zip(
            specifications,
            inputs_by_distribution,
            outputs_by_distribution,
        )
    ):
        # Scale every input distribution independently so that the spatial
        # structure remains visible. For log-normal fields, robust percentiles
        # prevent isolated extreme values from flattening the full image.
        if specification["distribution"] == "LN":
            input_min, input_max = np.quantile(inputs, [0.01, 0.99])
        else:
            input_min = float(inputs.min())
            input_max = float(inputs.max())
        if np.isclose(input_min, input_max):
            input_max = input_min + 1.0

        input_image = None
        for sample_index, (permeability, solution) in enumerate(
            zip(inputs, outputs)
        ):
            input_row = 2 * sample_index
            output_row = input_row + 1
            input_axis = axes[input_row, column]
            output_axis = axes[output_row, column]

            input_image = input_axis.imshow(
                permeability,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="viridis",
                vmin=input_min,
                vmax=input_max,
                interpolation="nearest",
            )
            output_image = output_axis.imshow(
                solution,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="magma",
                vmin=output_min,
                vmax=output_max,
                interpolation="nearest",
            )

            for axis in (input_axis, output_axis):
                axis.set_aspect("equal")
                axis.set_xticks([])
                axis.set_yticks([])

            if sample_index == number_of_samples - 1:
                output_axis.set_xlabel(r"$x_1$")
                output_axis.set_xticks([0.0, 0.5, 1.0])

            if column == 0:
                input_axis.set_ylabel(
                    f"Sample {sample_index + 1}\nInput " + r"$a(\boldsymbol{x})$"
                )
                output_axis.set_ylabel(
                    f"Sample {sample_index + 1}\nOutput " + r"$u(\boldsymbol{x})$"
                )
                input_axis.set_yticks([0.0, 0.5, 1.0])
                output_axis.set_yticks([0.0, 0.5, 1.0])

        input_colorbar = figure.colorbar(
            input_image,
            ax=axes[0::2, column].ravel().tolist(),
            location="right",
            shrink=0.82,
            pad=0.02,
        )
        input_colorbar.set_label(r"$a(\boldsymbol{x})$")

    output_colorbar = figure.colorbar(
        output_image,
        ax=axes[1::2, :].ravel().tolist(),
        location="right",
        shrink=0.85,
        pad=0.02,
    )
    output_colorbar.set_label(r"Darcy solution $u(\boldsymbol{x})$")

    figure.subplots_adjust(
        left=0.08,
        right=0.93,
        bottom=0.10,
        top=0.98,
        wspace=0.20,
        hspace=0.14,
    )
    output_path = (
        Path(figure_dir)
        / f"darcy_train_test_distribution_pairs_IDalpha_{id_alpha}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        pil_kwargs={"compress_level": 1},
    )

    if show:
        plt.show()
    plt.close(figure)
    return output_path


# from plot_darcy_distributions import plot_darcy_distribution_pairs

plot_darcy_distribution_pairs(
    grf_sampler=GRF,
    darcy_solver=solve_gwf,
    resolution=resolution,
    id_alpha=id_alpha,
    ood_alphas=ood_alphas,
    cookie_radii=cookie_radii,
    tau=tau,
    seed=seed,
    figure_dir=FIGURE_DIR,
    number_of_samples=1,
    dpi=150,
    show=False,
)
