"""Visualize horizontal slices from a three-dimensional PINN inversion.

The inversion result archive is self-contained, so this module intentionally
depends only on NumPy and Matplotlib. It does not import JAX, Lightweaver, or
the forward solver and is therefore safe to run on a headless login node.

By default four figures are written:

* input/truth thermodynamics: temperature, electron density, total hydrogen
  density, and LOS velocity;
* inferred thermodynamics;
* input/truth Cartesian magnetic components; and
* inferred Cartesian magnetic components.

Every figure has three height rows and one column per physical component: four
columns for thermodynamics and three for the magnetic field. The input and
inferred figures share identical color normalization so they can be compared
directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


DEFAULT_RESULT = Path("data") / "pinn_3d_inversion.npz"
DEFAULT_OUTPUT_DIR = Path("figures") / "pinn_3d"
DEFAULT_HEIGHT_FRACTIONS = (0.2, 0.5, 0.8)

_ARCHIVE_FIELD_KEYS = {
    "temperature": "temperature_K",
    "ne": "electron_density_m3",
    "nhtot": "hydrogen_density_m3",
    "vz": "los_velocity_m_s",
    "b": "magnetic_field_T",
    "gamma": "inclination_rad",
    "chi": "azimuth_rad",
}


@dataclass(frozen=True)
class InversionSlices:
    """Validated arrays required by the horizontal-slice figures."""

    x: np.ndarray
    y: np.ndarray
    height_m: np.ndarray
    truth: dict[str, np.ndarray]
    inferred: dict[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.x.size, self.y.size, self.height_m.size)


class ComponentSpec(NamedTuple):
    key: str
    title: str
    cmap: str
    signed: bool


THERMODYNAMIC_SPECS = (
    ComponentSpec("temperature", r"Temperature $T$ [K]", "inferno", False),
    ComponentSpec(
        "log_ne",
        r"Electron density $\log_{10}(n_e\,[\mathrm{m}^{-3}])$",
        "viridis",
        False,
    ),
    ComponentSpec(
        "log_nhtot",
        r"Total hydrogen density $\log_{10}(n_{\mathrm{H,tot}}\,[\mathrm{m}^{-3}])$",
        "viridis",
        False,
    ),
    ComponentSpec("vz", r"LOS velocity $v_z$ [km s$^{-1}$]", "RdBu_r", True),
)

MAGNETIC_SPECS = (
    ComponentSpec("bx", r"$B_x$ [G]", "RdBu_r", True),
    ComponentSpec("by", r"$B_y$ [G]", "RdBu_r", True),
    ComponentSpec("bz", r"$B_z$ (LOS) [G]", "RdBu_r", True),
)


def _load_axis(archive, key: str, *, minimum_size: int) -> np.ndarray:
    axis = np.asarray(archive[key], dtype=float)
    if axis.ndim != 1 or axis.size < minimum_size:
        raise ValueError(
            f"{key} must be one-dimensional with at least {minimum_size} values"
        )
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{key} must contain only finite values")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{key} must be strictly increasing")
    return axis


def load_inversion_slices(path: str | Path = DEFAULT_RESULT) -> InversionSlices:
    """Load and validate fields needed by the requested figures."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"inversion result archive does not exist: {path}")

    required = {"x_normalized", "y_normalized", "height_m"}
    for prefix in ("truth_", "inferred_"):
        required.update(f"{prefix}{key}" for key in _ARCHIVE_FIELD_KEYS.values())

    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                "inversion result is missing visualization fields: "
                + ", ".join(sorted(missing))
            )
        x = _load_axis(archive, "x_normalized", minimum_size=1)
        y = _load_axis(archive, "y_normalized", minimum_size=1)
        height_m = _load_axis(archive, "height_m", minimum_size=3)
        expected_shape = (x.size, y.size, height_m.size)
        states: dict[str, dict[str, np.ndarray]] = {}
        for state, prefix in (("truth", "truth_"), ("inferred", "inferred_")):
            fields = {}
            for name, archive_key in _ARCHIVE_FIELD_KEYS.items():
                field = np.asarray(archive[f"{prefix}{archive_key}"], dtype=float)
                if field.shape != expected_shape:
                    raise ValueError(
                        f"{prefix}{archive_key} has shape {field.shape}; "
                        f"expected {expected_shape}"
                    )
                if not np.all(np.isfinite(field)):
                    raise ValueError(f"{prefix}{archive_key} must be finite")
                fields[name] = field
            states[state] = fields

    for state, fields in states.items():
        if np.any(fields["temperature"] <= 0.0):
            raise ValueError(f"{state} temperature must be strictly positive")
        if np.any(fields["ne"] <= 0.0):
            raise ValueError(f"{state} electron density must be strictly positive")
        if np.any(fields["nhtot"] <= 0.0):
            raise ValueError(
                f"{state} total hydrogen density must be strictly positive"
            )
        if np.any(fields["b"] < 0.0):
            raise ValueError(f"{state} magnetic-field strength cannot be negative")
        if np.any((fields["gamma"] < 0.0) | (fields["gamma"] > np.pi)):
            raise ValueError(f"{state} magnetic inclination must lie in [0, pi]")

    return InversionSlices(
        x=x,
        y=y,
        height_m=height_m,
        truth=states["truth"],
        inferred=states["inferred"],
    )


def resolve_height_indices(
    height_m,
    *,
    indices: tuple[int, int, int] | list[int] | None = None,
    heights_km: tuple[float, float, float] | list[float] | None = None,
    fractions: tuple[float, float, float] | list[float] | None = None,
) -> np.ndarray:
    """Resolve exactly three requested heights to distinct depth indices."""

    height_m = np.asarray(height_m, dtype=float)
    if height_m.ndim != 1 or height_m.size < 3:
        raise ValueError("height_m must contain at least three grid points")
    selectors = sum(value is not None for value in (indices, heights_km, fractions))
    if selectors > 1:
        raise ValueError("choose only indices, heights_km, or fractions")
    if selectors == 0:
        fractions = DEFAULT_HEIGHT_FRACTIONS

    if indices is not None:
        selected = np.asarray(indices, dtype=int)
        if selected.shape != (3,):
            raise ValueError("exactly three height indices are required")
        if np.any((selected < 0) | (selected >= height_m.size)):
            raise ValueError(f"height indices must lie in [0, {height_m.size - 1}]")
    else:
        if heights_km is not None:
            targets_m = 1000.0 * np.asarray(heights_km, dtype=float)
            if targets_m.shape != (3,) or not np.all(np.isfinite(targets_m)):
                raise ValueError("exactly three finite heights in km are required")
            if np.any((targets_m < height_m[0]) | (targets_m > height_m[-1])):
                raise ValueError(
                    "requested heights must lie inside "
                    f"[{height_m[0] / 1000.0:g}, {height_m[-1] / 1000.0:g}] km"
                )
        else:
            fractions_array = np.asarray(fractions, dtype=float)
            if fractions_array.shape != (3,) or not np.all(
                np.isfinite(fractions_array)
            ):
                raise ValueError("exactly three finite height fractions are required")
            if np.any((fractions_array < 0.0) | (fractions_array > 1.0)):
                raise ValueError("height fractions must lie in [0, 1]")
            targets_m = height_m[0] + fractions_array * (height_m[-1] - height_m[0])
        selected = np.asarray(
            [int(np.argmin(np.abs(height_m - target))) for target in targets_m]
        )

    selected = np.sort(selected)
    if np.unique(selected).size != 3:
        raise ValueError(
            "the requested heights map to fewer than three distinct grid points"
        )
    return selected


def thermodynamic_components(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return plotted thermodynamic quantities in display units."""

    return {
        "temperature": np.asarray(fields["temperature"]),
        "log_ne": np.log10(np.asarray(fields["ne"])),
        "log_nhtot": np.log10(np.asarray(fields["nhtot"])),
        "vz": 1.0e-3 * np.asarray(fields["vz"]),
    }


def magnetic_components(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert field strength/inclination/azimuth to Cartesian gauss."""

    strength_g = 1.0e4 * np.asarray(fields["b"])
    inclination = np.asarray(fields["gamma"])
    azimuth = np.asarray(fields["chi"])
    transverse = strength_g * np.sin(inclination)
    return {
        "bx": transverse * np.cos(azimuth),
        "by": transverse * np.sin(azimuth),
        "bz": strength_g * np.cos(inclination),
    }


def _normalization(values: np.ndarray, *, signed: bool) -> Normalize:
    values = np.asarray(values, dtype=float)
    if signed:
        limit = float(np.max(np.abs(values)))
        if limit == 0.0:
            limit = 1.0
        return Normalize(vmin=-limit, vmax=limit)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1.0)
        lower -= padding
        upper += padding
    return Normalize(vmin=lower, vmax=upper)


def shared_component_normalizations(
    truth: dict[str, np.ndarray],
    inferred: dict[str, np.ndarray],
    indices,
    specs: tuple[ComponentSpec, ...],
) -> dict[str, Normalize]:
    """Create input/output-identical limits for each physical component."""

    indices = np.asarray(indices, dtype=int)
    normalizations = {}
    for spec in specs:
        selected = np.concatenate(
            (
                truth[spec.key][:, :, indices].ravel(),
                inferred[spec.key][:, :, indices].ravel(),
            )
        )
        normalizations[spec.key] = _normalization(selected, signed=spec.signed)
    return normalizations


def _horizontal_extent(axis: np.ndarray) -> tuple[float, float]:
    if axis.size > 1:
        return float(axis[0]), float(axis[-1])
    return float(axis[0] - 0.5), float(axis[0] + 0.5)


def plot_component_grid(
    data: InversionSlices,
    components: dict[str, np.ndarray],
    indices,
    specs: tuple[ComponentSpec, ...],
    normalizations: dict[str, Normalize],
    *,
    title: str,
):
    """Create one three-height horizontal-slice figure for the components."""

    indices = np.asarray(indices, dtype=int)
    if indices.shape != (3,):
        raise ValueError("a component grid requires exactly three height indices")
    if not specs:
        raise ValueError("a component grid requires at least one component")
    n_columns = len(specs)
    figure, axes = plt.subplots(
        3,
        n_columns,
        figsize=(4.5 * n_columns, 11.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    extent = (*_horizontal_extent(data.x), *_horizontal_extent(data.y))
    color_images = []
    for row, depth_index in enumerate(indices):
        for column, spec in enumerate(specs):
            axis = axes[row, column]
            image = axis.imshow(
                components[spec.key][:, :, depth_index].T,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
                cmap=spec.cmap,
                norm=normalizations[spec.key],
                rasterized=True,
            )
            if row == 0:
                axis.set_title(spec.title)
                color_images.append(image)
            if row == 2:
                axis.set_xlabel("x (normalized)")
            if column == 0:
                axis.set_ylabel("y (normalized)")
                axis.annotate(
                    f"z = {data.height_m[depth_index] / 1000.0:.1f} km\n"
                    f"index {depth_index}",
                    xy=(-0.43, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    fontweight="bold",
                )
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
    for column, image in enumerate(color_images):
        figure.colorbar(image, ax=axes[:, column], shrink=0.92, pad=0.02)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    return figure


def generate_visualizations(
    result_path: str | Path = DEFAULT_RESULT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    indices: tuple[int, int, int] | list[int] | None = None,
    heights_km: tuple[float, float, float] | list[float] | None = None,
    fractions: tuple[float, float, float] | list[float] | None = None,
    states: tuple[str, ...] = ("truth", "inferred"),
    quantities: tuple[str, ...] = ("thermodynamics", "magnetic"),
    prefix: str = "pinn_3d",
    image_format: str = "png",
    dpi: int = 180,
) -> tuple[list[Path], np.ndarray]:
    """Generate the selected figures and return paths plus depth indices."""

    if not states or any(state not in {"truth", "inferred"} for state in states):
        raise ValueError("states may contain only truth and inferred")
    if not quantities or any(
        quantity not in {"thermodynamics", "magnetic"} for quantity in quantities
    ):
        raise ValueError("quantities may contain only thermodynamics and magnetic")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename prefix")
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("image_format must be png, pdf, or svg")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    data = load_inversion_slices(result_path)
    selected = resolve_height_indices(
        data.height_m,
        indices=indices,
        heights_km=heights_km,
        fractions=fractions,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_components = {
        "truth": {
            "thermodynamics": thermodynamic_components(data.truth),
            "magnetic": magnetic_components(data.truth),
        },
        "inferred": {
            "thermodynamics": thermodynamic_components(data.inferred),
            "magnetic": magnetic_components(data.inferred),
        },
    }
    quantity_specs = {
        "thermodynamics": THERMODYNAMIC_SPECS,
        "magnetic": MAGNETIC_SPECS,
    }
    quantity_titles = {
        "thermodynamics": "thermodynamic atmosphere",
        "magnetic": "Cartesian magnetic field",
    }
    state_titles = {"truth": "Input / truth", "inferred": "PINN inversion"}

    written = []
    for quantity in quantities:
        specs = quantity_specs[quantity]
        normalizations = shared_component_normalizations(
            state_components["truth"][quantity],
            state_components["inferred"][quantity],
            selected,
            specs,
        )
        for state in states:
            figure = plot_component_grid(
                data,
                state_components[state][quantity],
                selected,
                specs,
                normalizations,
                title=f"{state_titles[state]}: {quantity_titles[quantity]}",
            )
            path = output_dir / f"{prefix}_{state}_{quantity}.{image_format}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            written.append(path)
    return written, selected


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot horizontal slices of a PINN inversion's input and output "
            "atmospheres."
        )
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    height_group = parser.add_mutually_exclusive_group()
    height_group.add_argument(
        "--heights-km",
        nargs=3,
        type=float,
        metavar=("LOW", "MIDDLE", "HIGH"),
        help="three target geometric heights in km (nearest grid points are used)",
    )
    height_group.add_argument(
        "--height-indices",
        nargs=3,
        type=int,
        metavar=("LOW", "MIDDLE", "HIGH"),
        help="three exact vertical grid indices",
    )
    height_group.add_argument(
        "--height-fractions",
        nargs=3,
        type=float,
        metavar=("LOW", "MIDDLE", "HIGH"),
        help="three fractions of the geometric height range; default: 0.2 0.5 0.8",
    )
    parser.add_argument(
        "--state",
        choices=("both", "truth", "inferred"),
        default="both",
        help="which atmosphere state to plot",
    )
    parser.add_argument(
        "--quantity",
        choices=("all", "thermodynamics", "magnetic"),
        default="all",
        help="which physical-component figure to plot",
    )
    parser.add_argument("--prefix", default="pinn_3d")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv=None) -> None:
    args = build_argument_parser().parse_args(argv)
    states = ("truth", "inferred") if args.state == "both" else (args.state,)
    quantities = (
        ("thermodynamics", "magnetic") if args.quantity == "all" else (args.quantity,)
    )
    paths, selected = generate_visualizations(
        result_path=args.result,
        output_dir=args.output_dir,
        indices=args.height_indices,
        heights_km=args.heights_km,
        fractions=args.height_fractions,
        states=states,
        quantities=quantities,
        prefix=args.prefix,
        image_format=args.format,
        dpi=args.dpi,
    )
    with np.load(args.result, allow_pickle=False) as archive:
        height_m = np.asarray(archive["height_m"], dtype=float)
    selections = ", ".join(
        f"index {index} = {height_m[index] / 1000.0:.1f} km" for index in selected
    )
    print(f"Selected heights: {selections}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
