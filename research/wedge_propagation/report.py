"""Figures and a readable report for the fixed wedge operator experiment.

This module reports operator measurements. It does not train a model or infer
classification accuracy from an operator residual.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_OPERATOR_NUMERIC = (
    "num_nodes",
    "num_edges",
    "num_wedges",
    "lambda_L",
    "lambda_L2",
    "lambda_Q",
    "num_components",
)
_OPTIONAL_OPERATOR_NUMERIC = (
    "polynomial_residual",
    "normalized_polynomial_residual",
    "degree_sum_std",
    "identity_relative_error",
    "explicit_fast_relative_error",
)
_ACTION_NUMERIC = (
    "raw_L_norm",
    "raw_L2_norm",
    "raw_Q_norm",
    "energy_L",
    "energy_L2",
    "energy_Q",
)
_FIGURES = (
    "polynomial_residual",
    "normalized_action_difference",
    "spectra",
    "raw_action_energy",
)


def _finite_number(value: Any, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite; received {value!r}")


def _validate(
    output_dir: Path,
    operator_rows: list[dict],
    action_rows: list[dict],
    spectra: dict[str, np.ndarray],
) -> tuple[list[str], dict[str, dict]]:
    if not output_dir.is_dir():
        raise ValueError(f"Report output directory must already exist: {output_dir}")
    if not operator_rows:
        raise ValueError("operator_rows must contain at least one measured graph")
    if not action_rows:
        raise ValueError("action_rows must contain at least one measured input")

    by_graph: dict[str, dict] = {}
    families: list[str] = []
    for row_index, row in enumerate(operator_rows):
        graph_id, family = row.get("graph_id"), row.get("family")
        if not isinstance(graph_id, str) or not graph_id:
            raise ValueError(f"operator_rows[{row_index}].graph_id must be nonempty")
        if not isinstance(family, str) or not family:
            raise ValueError(f"operator_rows[{row_index}].family must be nonempty")
        if graph_id in by_graph:
            raise ValueError(f"Duplicate measured graph: {graph_id}")
        for name in _OPERATOR_NUMERIC:
            _finite_number(row.get(name), f"{graph_id}.{name}")
        for name in _OPTIONAL_OPERATOR_NUMERIC:
            if name not in row:
                raise ValueError(f"Missing metric {graph_id}.{name}")
            _finite_number(row[name], f"{graph_id}.{name}", optional=True)
        for name in ("identity_absolute_error", "explicit_fast_absolute_error"):
            if name in row:
                _finite_number(row[name], f"{graph_id}.{name}")
        if row["degree_sum_std"] is None and row["num_edges"] != 0:
            raise ValueError(f"{graph_id}.degree_sum_std is undefined only for an edgeless graph")
        for name in ("num_nodes", "num_edges", "num_wedges", "num_components"):
            value = float(row[name])
            if value < 0 or not value.is_integer():
                raise ValueError(f"{graph_id}.{name} must be a nonnegative integer")
        if row["num_nodes"] < 1:
            raise ValueError(f"{graph_id}.num_nodes must be positive")
        if row["num_components"] < 1 or row["num_components"] > row["num_nodes"]:
            raise ValueError(f"Invalid component count for {graph_id}")
        by_graph[graph_id] = row
        if family not in families:
            families.append(family)
        for operator in ("L", "L2", "Q"):
            key = f"{graph_id}__{operator}"
            if key not in spectra:
                raise ValueError(f"Missing spectrum: {key}")
            values = np.asarray(spectra[key])
            if values.shape != (int(row["num_nodes"]),):
                raise ValueError(f"{key} must contain all num_nodes eigenvalues")
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f"{key} must contain real eigenvalues")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{key} contains nonfinite eigenvalues")

    seen_inputs: set[tuple[str, int]] = set()
    for row_index, row in enumerate(action_rows):
        graph_id = row.get("graph_id")
        if graph_id not in by_graph:
            raise ValueError(f"Unknown action graph: {graph_id}")
        feature_index = row.get("feature_index")
        _finite_number(feature_index, f"action_rows[{row_index}].feature_index")
        if float(feature_index) < 0 or not float(feature_index).is_integer():
            raise ValueError("feature_index must be a nonnegative integer")
        pair = (graph_id, int(feature_index))
        if pair in seen_inputs:
            raise ValueError(f"Duplicate measured graph/feature input: {pair}")
        seen_inputs.add(pair)
        for name in _ACTION_NUMERIC:
            _finite_number(row.get(name), f"{pair}.{name}")
        name = "normalized_action_Q_L2_relative_difference"
        if name not in row:
            raise ValueError(f"Missing metric {pair}.{name}")
        _finite_number(row[name], f"{pair}.{name}", optional=True)

    for graph_id in by_graph:
        if not any(pair[0] == graph_id for pair in seen_inputs):
            raise ValueError(f"No action measurements for {graph_id}")
    destinations = [output_dir / "SUMMARY.md"]
    destinations.extend(
        output_dir / f"{name}.{extension}" for name in _FIGURES for extension in ("png", "pdf")
    )
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite report artifacts: " + ", ".join(existing))
    return families, by_graph


def _save_figure(figure: Any, output_dir: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        # Exclusive creation also protects against an output collision after
        # the initial existence check.
        with (output_dir / f"{name}.{extension}").open("xb") as stream:
            figure.savefig(stream, format=extension, dpi=180, bbox_inches="tight")


def _values(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=float)


def _stats(values: np.ndarray) -> str:
    if values.size == 0:
        return "undefined"
    return f"{np.min(values):.6g} / {np.median(values):.6g} / {np.max(values):.6g}"


def _median(values: np.ndarray) -> str:
    return f"{np.median(values):.6g}" if values.size else "undefined"


def _maximum(values: np.ndarray) -> str:
    return f"{np.max(values):.6g}" if values.size else "undefined"


def _absolute_maximum(rows: list[dict], metric: str) -> str:
    values = np.asarray([row[metric] for row in rows if metric in row], dtype=float)
    return f"{np.max(values):.6g}" if values.size else "unreported"


def _jitter(count: int, width: float = 0.13) -> np.ndarray:
    # Deterministic spacing; plotting never consumes the experiment RNG.
    return np.linspace(-width, width, count) if count > 1 else np.zeros(count)


def _plot_polynomial(plt: Any, output_dir: Path, families: list[str], groups: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    try:
        largest_residual = 0.0
        for index, family in enumerate(families):
            rows = [row for row in groups[family] if row["polynomial_residual"] is not None]
            values = _values(rows, "polynomial_residual")
            if values.size:
                largest_residual = max(largest_residual, float(np.max(values)))
            color = f"C{index % 10}"
            axes[0].scatter(index + _jitter(values.size), values, color=color, s=18, alpha=0.65)
            axes[1].scatter(
                [row["num_nodes"] for row in rows],
                values,
                color=color,
                s=18,
                alpha=0.65,
                label=family,
            )
        axes[0].set_xticks(range(len(families)), families, rotation=25, ha="right")
        axes[0].set_xlabel("Graph family")
        axes[1].set_xlabel("Number of nodes")
        for axis in axes:
            axis.set_ylabel("Relative Frobenius residual")
            axis.set_yscale("symlog", linthresh=1e-12)
            if largest_residual < 1e-12:
                axis.set_ylim(0, 1e-12)
            axis.grid(True, alpha=0.25)
        axes[0].set_title("Best per-graph fit: Q = uL + vL²")
        axes[1].set_title("Polynomial residual versus graph size")
        axes[1].legend(fontsize=8)
        _save_figure(figure, output_dir, "polynomial_residual")
    finally:
        plt.close(figure)


def _plot_difference(plt: Any, output_dir: Path, families: list[str], groups: dict) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    try:
        for index, family in enumerate(families):
            values = _values(groups[family], "normalized_action_Q_L2_relative_difference")
            if values.size:
                axis.boxplot(values, positions=[index], widths=0.55, showfliers=False)
                axis.scatter(
                    index + _jitter(values.size), values, s=6, alpha=0.12, color=f"C{index % 10}"
                )
            else:
                axis.annotate("undefined", (index, 0), rotation=90, ha="center", va="bottom")
        axis.set_xticks(range(len(families)), families, rotation=25, ha="right")
        axis.set_xlabel("Graph family")
        axis.set_ylabel("Relative difference in spectral-normalized actions")
        axis.set_title("Q versus L² on the same features; undefined inputs excluded")
        axis.grid(True, axis="y", alpha=0.25)
        _save_figure(figure, output_dir, "normalized_action_difference")
    finally:
        plt.close(figure)


def _plot_spectra(
    plt: Any,
    output_dir: Path,
    families: list[str],
    groups: dict,
    spectra: dict[str, np.ndarray],
) -> list[str]:
    columns = min(3, len(families))
    rows = math.ceil(len(families) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.1 * columns, 3.3 * rows), squeeze=False, constrained_layout=True
    )
    selected_graphs: list[str] = []
    try:
        for index, family in enumerate(families):
            axis = axes.flat[index]
            # Largest measured size; first graph in the manifest breaks ties.
            row = max(groups[family], key=lambda item: item["num_nodes"])
            graph_id = row["graph_id"]
            selected_graphs.append(graph_id)
            for operator, label in (("L", "L"), ("L2", "L²"), ("Q", "Q")):
                eigenvalues = np.sort(np.asarray(spectra[f"{graph_id}__{operator}"], dtype=float))
                radius = float(np.max(np.abs(eigenvalues)))
                if radius == 0:
                    plotted = eigenvalues
                    label += " (zero operator)"
                else:
                    plotted = eigenvalues / radius
                axis.plot(np.arange(eigenvalues.size), plotted, label=label, linewidth=1.3)
            axis.set_title(f"{family}: {graph_id}", fontsize=9)
            axis.set_xlabel("Sorted eigenvalue index")
            axis.set_ylabel("Eigenvalue / spectral radius")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
        for index in range(len(families), rows * columns):
            axes.flat[index].set_visible(False)
        _save_figure(figure, output_dir, "spectra")
    finally:
        plt.close(figure)
    return selected_graphs


def _plot_raw(plt: Any, output_dir: Path, families: list[str], groups: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    try:
        for index, family in enumerate(families):
            for offset, operator in enumerate(("L", "L2", "Q")):
                position = index + (offset - 1) * 0.22
                color = f"C{offset}"
                for axis, metric in zip(
                    axes, (f"raw_{operator}_norm", f"energy_{operator}"), strict=True
                ):
                    parts = axis.boxplot(
                        _values(groups[family], metric),
                        positions=[position],
                        widths=0.19,
                        showfliers=False,
                        patch_artist=True,
                    )
                    parts["boxes"][0].set_facecolor(color)
                    parts["boxes"][0].set_alpha(0.6)
        for axis in axes:
            axis.set_xticks(range(len(families)), families, rotation=25, ha="right")
            axis.set_xlabel("Graph family")
            axis.set_yscale("symlog", linthresh=1.0)
            axis.grid(True, axis="y", alpha=0.25)
            for index, label in enumerate(("L", "L²", "Q")):
                axis.plot([], [], color=f"C{index}", linewidth=5, label=label)
            axis.legend(fontsize=8)
        axes[0].set_ylabel("Raw action norm ||TX||₂")
        axes[0].set_title("Raw operator actions; feature scale unchanged")
        axes[1].set_ylabel("Raw energy XᵀTX")
        axes[1].set_title("Raw energies; zero values remain visible")
        _save_figure(figure, output_dir, "raw_action_energy")
    finally:
        plt.close(figure)


def _summary(
    families: list[str],
    operator_groups: dict,
    action_groups: dict,
    selected_graphs: list[str],
    contract: dict,
) -> str:
    lines = [
        "# Fixed wedge operator experiment — measured results",
        "",
        "This report compares fixed operators L = BᵀB, L² and Q = AᵀA. "
        "Every path row of A has coefficients (+1, −2, +1), and C₂ = I. "
        "There is no trained model, optimizer, classification metric "
        "or checkpoint in this experiment.",
        "",
        "The graph/feature cases are synthetic operator measurements. "
        "They are not real-dataset classification results.",
        "",
        "## Polynomial fit and implementation checks",
        "",
        "Each residual fits u and v separately using the entire measured graph: "
        "r = min ||Q − uL − vL²||_F / ||Q||_F. "
        "This is an operator diagnostic, not a learned rule or a generalization score. "
        "Independent nonzero scalar normalization of these operators leaves this relative "
        "projection residual unchanged, so the normalized column is the same diagnostic. "
        "Cells give minimum / median / maximum over graphs. "
        "Relative implementation errors are undefined when their reference norm is at most "
        "1e-12; maxima exclude those cases. Absolute errors remain defined. "
        "An absent optional absolute-error field is marked unreported.",
        "",
        "| Family | Graphs | Raw relative residual | Normalized relative residual "
        "| Undefined raw / normalized | Max relative identity error "
        "| Max relative explicit/fast error "
        "| Max absolute identity error | Max absolute explicit/fast error |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in families:
        rows = operator_groups[family]
        residual = _values(rows, "polynomial_residual")
        normalized = _values(rows, "normalized_polynomial_residual")
        lines.append(
            f"| {family} | {len(rows)} | {_stats(residual)} | {_stats(normalized)} "
            f"| {len(rows) - residual.size} / {len(rows) - normalized.size} "
            f"| {_maximum(_values(rows, 'identity_relative_error'))} "
            f"| {_maximum(_values(rows, 'explicit_fast_relative_error'))} "
            f"| {_absolute_maximum(rows, 'identity_absolute_error')} "
            f"| {_absolute_maximum(rows, 'explicit_fast_absolute_error')} |"
        )
    lines.extend(
        [
            "",
            "The exact identity is Q = L² + Bᵀ diag(d_u + d_v − 4) B. "
            "If all edge degree sums equal s, Q = L² + (s − 4)L. "
            "Regular graphs and biregular graphs such as stars can therefore have zero residual. "
            "For cycles, Q = L². Agreement in those cases is an expected positive control.",
            "",
            "A nonzero residual shows that this graph's Q is not captured by a scalar combination "
            "of L and L². It does not establish greater expressive capacity "
            "or improved predictions for a GNN.",
            "",
            "## Actions after matching operator strength",
            "",
            "The following uses each nonzero operator's spectral normalization. "
            "The relative difference is ||(Q/ρQ)X − (L²/ρL²)X|| / ||(Q/ρQ)X||. "
            "All feature draws are retained. Undefined relative differences "
            "are counted and excluded "
            "from numerical summaries; they are not replaced by zero. "
            "Feature draws from one graph are repeated measurements of that graph, "
            "not independent graph samples.",
            "",
            "| Family | Inputs | Q versus L² relative difference: min / median / max "
            "| Undefined inputs |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for family in families:
        rows = action_groups[family]
        values = _values(rows, "normalized_action_Q_L2_relative_difference")
        lines.append(f"| {family} | {len(rows)} | {_stats(values)} | {len(rows) - values.size} |")
    lines.extend(
        [
            "",
            "## Raw operator scales and graph structure",
            "",
            "The raw action and energy figure preserves operator scale. "
            "The table below gives median spectral radii "
            "and median standard deviation of edge degree sums. "
            "An edgeless graph has no edge degree-sum distribution; its standard deviation "
            "is undefined and excluded from the median. The defined/total graph count is shown.",
            "",
            "| Family | Median λmax(L) | Median λmax(L²) | Median λmax(Q) "
            "| Median std(d_u+d_v), defined/total | Nodes: min / max | Edges: min / max "
            "| Wedges: min / max | Components: min / max |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for family in families:
        rows = operator_groups[family]
        medians = " | ".join(
            _median(_values(rows, metric)) for metric in ("lambda_L", "lambda_L2", "lambda_Q")
        )
        degree_std = _values(rows, "degree_sum_std")
        medians += f" | {_median(degree_std)} ({degree_std.size}/{len(rows)})"
        ranges = " | ".join(
            f"{min(row[metric] for row in rows):g} / {max(row[metric] for row in rows):g}"
            for metric in ("num_nodes", "num_edges", "num_wedges", "num_components")
        )
        lines.append(f"| {family} | {medians} | {ranges} |")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "Raw artifacts: [operators.csv](operators.csv), [actions.csv](actions.csv), "
            "[spectra.npz](spectra.npz), and [contract.json](contract.json).",
            "",
            "- [Polynomial residuals](polynomial_residual.png) "
            "([PDF](polynomial_residual.pdf)): all measured graphs, by family and size.",
            "- [Normalized action difference](normalized_action_difference.png) "
            "([PDF](normalized_action_difference.pdf)): all defined graph/feature measurements.",
            "- [Spectra](spectra.png) ([PDF](spectra.pdf)): all eigenvalues of the largest "
            "measured case in each family; ties use the first manifest case.",
            "- [Raw actions and energies](raw_action_energy.png) "
            "([PDF](raw_action_energy.pdf)): operator scales are preserved.",
            "",
            "Spectrum cases: " + ", ".join(f"`{graph_id}`" for graph_id in selected_graphs) + ".",
            "",
            "## Supplied execution contract",
            "",
            "The values below are the supplied run contract, "
            "not defaults inferred by the report generator.",
            "",
            "```json",
            json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False),
            "```",
            "",
            "Implementation checks and fixed-operator measurements are distinct from training, "
            "unseen-graph learned-rule evaluation and real-dataset evaluation. Those later stages "
            "are not completed by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    output_dir: str | Path,
    operator_rows: list[dict],
    action_rows: list[dict],
    spectra: dict[str, np.ndarray],
    contract: dict,
) -> None:
    """Write measured-result PNG/PDF figures and SUMMARY.md without overwriting.

    The caller creates a fresh output directory and writes the raw data. Every
    spectrum is keyed as ``graph_id + '__L'`` (or ``'__L2'`` / ``'__Q'``).
    Undefined relative metrics must be represented by ``None``.
    """
    output_dir = Path(output_dir)
    families, by_graph = _validate(output_dir, operator_rows, action_rows, spectra)
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON-serializable dictionary")
    # Validate the contract before creating any artifact.
    json.dumps(contract, allow_nan=False)
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to generate measured-result figures. "
            "Install it in the experiment environment with: python -m pip install matplotlib"
        ) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    operator_groups: dict[str, list[dict]] = defaultdict(list)
    action_groups: dict[str, list[dict]] = defaultdict(list)
    for row in operator_rows:
        operator_groups[row["family"]].append(row)
    for row in action_rows:
        action_groups[by_graph[row["graph_id"]]["family"]].append(row)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        _plot_polynomial(plt, output_dir, families, operator_groups)
        _plot_difference(plt, output_dir, families, action_groups)
        selected_graphs = _plot_spectra(plt, output_dir, families, operator_groups, spectra)
        _plot_raw(plt, output_dir, families, action_groups)
    with (output_dir / "SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(_summary(families, operator_groups, action_groups, selected_graphs, contract))
