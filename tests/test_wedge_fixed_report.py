"""Debug tests for real report artifacts and explicit invalid-data failures."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from PIL import Image

from research.wedge_propagation.report import write_report


def _measure_graph(graph_id, family, edges, wedges):
    n = max(max(edge) for edge in edges) + 1
    incidence = np.zeros((len(edges), n), dtype=np.float64)
    for row, (source, target) in enumerate(edges):
        incidence[row, source] = -1
        incidence[row, target] = 1
    path = np.zeros((len(wedges), n), dtype=np.float64)
    for row, (first, center, last) in enumerate(wedges):
        path[row, first] = 1
        path[row, center] = -2
        path[row, last] = 1
    laplacian = incidence.T @ incidence
    square = laplacian @ laplacian
    wedge_operator = path.T @ path
    degree = np.diag(laplacian)
    degree_sums = np.asarray([degree[u] + degree[v] for u, v in edges])
    identity = square + incidence.T @ ((degree_sums - 4)[:, None] * incidence)
    coefficients = np.linalg.lstsq(
        np.stack([laplacian.ravel(), square.ravel()], axis=1),
        wedge_operator.ravel(),
        rcond=None,
    )[0]
    residual = np.linalg.norm(
        wedge_operator - coefficients[0] * laplacian - coefficients[1] * square
    ) / np.linalg.norm(wedge_operator)
    eigenvalues = {
        operator: np.linalg.eigvalsh(matrix)
        for operator, matrix in (("L", laplacian), ("L2", square), ("Q", wedge_operator))
    }
    radii = {operator: float(np.max(np.abs(values))) for operator, values in eigenvalues.items()}
    row = {
        "graph_id": graph_id,
        "family": family,
        "num_nodes": n,
        "num_edges": len(edges),
        "num_wedges": len(wedges),
        "polynomial_residual": residual,
        "normalized_polynomial_residual": residual,
        "degree_sum_std": np.std(degree_sums),
        "lambda_L": radii["L"],
        "lambda_L2": radii["L2"],
        "lambda_Q": radii["Q"],
        "identity_relative_error": np.linalg.norm(wedge_operator - identity)
        / np.linalg.norm(wedge_operator),
        "explicit_fast_relative_error": 0.0,
        "num_components": 1,
    }
    action_rows = []
    for index, feature in enumerate(
        (np.arange(n, dtype=float), np.asarray([(-1.0) ** i for i in range(n)]))
    ):
        actions = {
            operator: matrix @ feature
            for operator, matrix in (("L", laplacian), ("L2", square), ("Q", wedge_operator))
        }
        q_normalized = actions["Q"] / radii["Q"]
        denominator = np.linalg.norm(q_normalized)
        action = {
            "graph_id": graph_id,
            "feature_index": index,
            **{
                f"raw_{operator}_norm": np.linalg.norm(value) for operator, value in actions.items()
            },
            **{f"energy_{operator}": feature @ value for operator, value in actions.items()},
            "normalized_action_Q_L2_relative_difference": (
                np.linalg.norm(q_normalized - actions["L2"] / radii["L2"]) / denominator
                if denominator > 0
                else None
            ),
        }
        action_rows.append(action)
    return (
        row,
        action_rows,
        {f"{graph_id}__{operator}": values for operator, values in eigenvalues.items()},
    )


@pytest.fixture
def measured_cases():
    cases = [
        _measure_graph("path-3", "path", [(0, 1), (1, 2)], [(0, 1, 2)]),
        _measure_graph(
            "cycle-4",
            "cycle",
            [(0, 1), (1, 2), (2, 3), (0, 3)],
            [(1, 0, 3), (0, 1, 2), (1, 2, 3), (0, 3, 2)],
        ),
    ]
    operators = [case[0] for case in cases]
    actions = [row for case in cases for row in case[1]]
    spectra = {key: values for case in cases for key, values in case[2].items()}
    return operators, actions, spectra


def test_report_creates_genuine_figures_and_measured_summary(tmp_path, measured_cases):
    operators, actions, spectra = measured_cases
    write_report(tmp_path, operators, actions, spectra, {"profile": "debug", "feature_draws": 2})

    pngs = sorted(tmp_path.glob("*.png"))
    pdfs = sorted(tmp_path.glob("*.pdf"))
    assert len(pngs) == len(pdfs) == 4
    for path in pngs:
        with Image.open(path) as picture:
            assert picture.format == "PNG"
            assert picture.width > 500 and picture.height > 300
            # A real plot must contain visible data/text, not just an empty page.
            pixels = np.asarray(picture.convert("RGB"))
            assert np.std(pixels) > 10
            assert np.count_nonzero(np.min(pixels, axis=-1) < 200) > 1000
    for path in pdfs:
        assert path.read_bytes().startswith(b"%PDF-")
        assert path.stat().st_size > 2000
    summary = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "path-3" in summary and "cycle-4" in summary
    assert "(+1, −2, +1)" in summary
    assert "Undefined" in summary
    assert len(summary.splitlines()) > 35
    assert "not establish greater expressive capacity" in summary
    assert "no trained model" in summary
    assert "||Q − uL − vL²||_F / ||Q||_F" in summary
    assert "operators.csv" in summary and "contract.json" in summary
    assert '"profile": "debug"' in summary


def test_report_refuses_existing_output_before_writing(tmp_path, measured_cases):
    operators, actions, spectra = measured_cases
    existing = tmp_path / "spectra.pdf"
    existing.write_bytes(b"existing result")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_report(tmp_path, operators, actions, spectra, {})
    assert existing.read_bytes() == b"existing result"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "failure",
    [
        "nonfinite_metric",
        "missing_spectrum",
        "wrong_spectrum_length",
        "duplicate_input",
        "unknown_graph",
    ],
)
def test_report_rejects_invalid_measurements_without_artifacts(tmp_path, measured_cases, failure):
    operators, actions, spectra = copy.deepcopy(measured_cases)
    if failure == "nonfinite_metric":
        operators[0]["polynomial_residual"] = float("nan")
    elif failure == "missing_spectrum":
        del spectra["path-3__Q"]
    elif failure == "wrong_spectrum_length":
        spectra["path-3__Q"] = np.asarray([1.0])
    elif failure == "duplicate_input":
        actions.append(actions[0].copy())
    elif failure == "unknown_graph":
        actions[0]["graph_id"] = "unmeasured"
    with pytest.raises(ValueError):
        write_report(tmp_path, operators, actions, spectra, {})
    assert not list(tmp_path.iterdir())


def test_report_requires_precreated_directory(tmp_path, measured_cases):
    operators, actions, spectra = measured_cases
    with pytest.raises(ValueError, match="already exist"):
        write_report(tmp_path / "missing", operators, actions, spectra, {})


def test_zero_operators_keep_relative_measurements_undefined(tmp_path):
    operator = {
        "graph_id": "isolates-2",
        "family": "isolates",
        "num_nodes": 2,
        "num_edges": 0,
        "num_wedges": 0,
        "polynomial_residual": None,
        "normalized_polynomial_residual": None,
        "degree_sum_std": None,
        "lambda_L": 0.0,
        "lambda_L2": 0.0,
        "lambda_Q": 0.0,
        "identity_relative_error": None,
        "explicit_fast_relative_error": None,
        "identity_absolute_error": 0.0,
        "explicit_fast_absolute_error": 0.0,
        "num_components": 2,
    }
    action = {
        "graph_id": "isolates-2",
        "feature_index": 0,
        **{name: 0.0 for name in ("raw_L_norm", "raw_L2_norm", "raw_Q_norm")},
        **{name: 0.0 for name in ("energy_L", "energy_L2", "energy_Q")},
        "normalized_action_Q_L2_relative_difference": None,
    }
    spectra = {f"isolates-2__{name}": np.zeros(2) for name in ("L", "L2", "Q")}
    write_report(tmp_path, [operator], [action], spectra, {"profile": "debug"})
    summary = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "| isolates | 1 | undefined | undefined | 1 / 1 |" in summary
    assert "| isolates | 1 | undefined | 1 |" in summary
    assert "undefined (0/1)" in summary
    assert (
        "| isolates | 1 | undefined | undefined | 1 / 1 | undefined | undefined | 0 | 0 |"
        in summary
    )
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_nullspace_action_has_no_relative_fast_error(tmp_path, measured_cases):
    operators, actions, spectra = measured_cases
    operator = operators[0].copy()
    operator["explicit_fast_relative_error"] = None
    operator["explicit_fast_absolute_error"] = 0.0
    # The first path feature is affine: QX is zero although Q itself is nonzero.
    action = actions[0]
    assert action["raw_Q_norm"] == 0
    assert action["normalized_action_Q_L2_relative_difference"] is None
    write_report(
        tmp_path,
        [operator],
        [action],
        {key: values for key, values in spectra.items() if key.startswith("path-3__")},
        {"profile": "debug"},
    )
    summary = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "| path | 1 | undefined | 1 |" in summary
    assert "| undefined | unreported | 0 |" in summary
