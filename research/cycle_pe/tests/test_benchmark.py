"""Unit fixtures only; the experiment CLI never creates substitute datasets."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chartgat.algebra import incidence_matrix
from chartgat.graphs import spanning_tree_indices
from research.cycle_pe import benchmark
from research.cycle_pe.benchmark_data import (
    CACHE_VERSION,
    DATASETS,
    EXPECTED_SIZES,
    Graph,
    _ready,
    collate,
    cycle_statistics,
    graph_fingerprint,
    prepare_graph,
)
from research.cycle_pe.benchmark_models import MODEL_NAME, CyclePEModel
from research.cycle_pe.features import cycle_set_statistics, static_fundamental_basis
from research.cycle_pe.paper_model import _MessageLayer


def _data(n: int = 4) -> SimpleNamespace:
    undirected = [(i, (i + 1) % n) for i in range(n)]
    edge_index = torch.tensor(undirected + [(v, u) for u, v in undirected]).T.contiguous()
    return SimpleNamespace(
        num_nodes=n,
        x=torch.arange(n).reshape(-1, 1),
        edge_index=edge_index,
        edge_attr=torch.ones((2 * n, 1), dtype=torch.long),
        y=torch.tensor([0.7]),
    )


def _graph(n: int = 4) -> Graph:
    return prepare_graph(_data(n))


def test_defaults_keep_paper_datasets_and_only_our_model() -> None:
    args = benchmark.parser().parse_args([])
    assert tuple(args.datasets) == DATASETS == ("zinc12k", "peptides_struct")
    assert EXPECTED_SIZES["zinc12k"] == (10000, 1000, 1000)
    assert sum(EXPECTED_SIZES["peptides_struct"]) == 15535
    assert MODEL_NAME == "cycle_set"
    assert args.workers == 4
    assert args.prefetch_factor == 2
    assert not hasattr(args, "baselines")
    assert not hasattr(args, "tiny")
    with pytest.raises(SystemExit):
        benchmark.parser().parse_args(["--baselines", "signnet"])


def test_v1_hashes_include_failure_safe_resource_monitor() -> None:
    assert "research/cycle_pe/resource_monitor.py" in benchmark.implementation_hashes()


def test_cpu_actual_benchmark_is_rejected() -> None:
    args = benchmark.parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark._validate(args)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark._train_model("zinc12k", {}, args)


def test_processed_only_cache_does_not_authorize_implicit_pyg_download(tmp_path) -> None:
    processed = tmp_path / "subset" / "processed"
    processed.mkdir(parents=True)
    for name in ("train", "val", "test"):
        (processed / f"{name}.pt").touch()
    assert not _ready(tmp_path, "zinc12k")


def test_fingerprint_hashes_targets_features_and_order() -> None:
    def fingerprint(data):
        digest = hashlib.sha256()
        graph_fingerprint(data, digest)
        return digest.hexdigest()

    original = _data()
    expected = fingerprint(original)
    original.y += 1
    assert fingerprint(original) != expected
    changed = _data()
    changed.x[0, 0] += 1
    assert fingerprint(changed) != expected


def test_preparation_has_only_cycle_pe_and_preserves_targets() -> None:
    data = _data(4)
    graph = prepare_graph(data)
    assert {field.name for field in fields(graph)} == {
        "x",
        "edge_index",
        "edge_attr",
        "y",
        "cycle_set",
    }
    assert CACHE_VERSION == "own-cycle-set-v2"
    assert graph.edge_index.shape == (2, 4)
    assert (graph.edge_index[0] < graph.edge_index[1]).all()
    assert graph.cycle_set.shape == (4, 6)
    torch.testing.assert_close(graph.y, data.y)
    torch.testing.assert_close(graph.x, data.x)


def test_cycle_set_preserves_existing_basis_summary_semantics() -> None:
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    incidence = incidence_matrix(4, edges)
    tree = spanning_tree_indices(4, edges, mode="bfs")
    expected = cycle_set_statistics(static_fundamental_basis(incidence, tree))
    directed = torch.tensor(edges + [(v, u) for u, v in edges]).T
    actual = cycle_statistics(4, directed)
    np.testing.assert_allclose(actual[: len(edges)].numpy(), expected, rtol=1e-6)
    torch.testing.assert_close(actual[: len(edges)], actual[len(edges) :])


def test_our_model_reuses_existing_layers_and_all_parameters_receive_gradients() -> None:
    torch.manual_seed(5)
    model = CyclePEModel(dataset="zinc12k", hidden=12, pe_dim=6, layers=2)
    assert all(isinstance(layer, _MessageLayer) for layer in model.layers)
    graphs = [_graph(4), _graph(5)]
    batch = collate(graphs)
    output = model(batch)
    assert output.shape == (2, 1)
    (output - batch.y).abs().mean().backward()
    benchmark._validate_first_step_gradients(model)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        combined = model(batch)
        separate = torch.cat([model(collate([graph])) for graph in graphs])
    torch.testing.assert_close(combined, separate, atol=3e-6, rtol=3e-6)


def test_observability_reports_official_loaded_graph_and_batch_counts() -> None:
    splits = {
        "train": [_graph(4), _graph(5), _graph(6)],
        "validation": [_graph(4)],
    }
    data = benchmark._data_observability("zinc12k", splits)
    assert data["official_split_counts"] == {
        "train": 10_000,
        "validation": 1_000,
        "test": 1_000,
    }
    assert data["loaded_graph_count"]["value"] == 4
    assert data["loaded_split_counts"]["test"]["value"] is None
    assert data["loaded_split_counts"]["test"]["reason"]
    assert data["graph_statistics"]["nodes_per_graph"]["maximum"]["value"] == 6
    args = benchmark.parser().parse_args(["--batch-size", "2", "--workers", "1"])
    batch = benchmark._batch_observability(args, splits)
    assert batch["maximum_effective_graphs_per_training_batch"] == 2
    assert batch["training_steps_per_epoch"] == 2
    assert batch["effective_batch_size"] == 2
    assert batch["persistent_workers"] is True
    assert batch["batch_candidate_throughput_sweep"]["value"] is None
    assert batch["batch_candidate_throughput_sweep"]["reason"]


def test_first_step_gradient_validation_fails_on_disconnected_parameter() -> None:
    model = torch.nn.Linear(2, 1)
    model.weight.sum().backward()
    with pytest.raises(RuntimeError, match="missing=.*bias"):
        benchmark._validate_first_step_gradients(model)


def test_graph_readout_is_permutation_invariant_given_transported_cycle_chart() -> None:
    torch.manual_seed(4)
    graph = _graph(5)
    model = CyclePEModel(dataset="zinc12k", hidden=12, pe_dim=6, layers=2).eval()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(permutation)
    transformed = replace(graph, x=graph.x[permutation], edge_index=inverse[graph.edge_index])
    torch.testing.assert_close(model(collate([graph])), model(collate([transformed])))
    reverse = replace(graph, edge_index=graph.edge_index.flip(0))
    torch.testing.assert_close(model(collate([graph])), model(collate([reverse])))


def test_cycle_set_amp_aggregation_matches_tensor_dtype() -> None:
    model = CyclePEModel(dataset="zinc12k", hidden=12, pe_dim=6, layers=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(collate([_graph()]))
    assert torch.isfinite(result).all()


def test_peptides_uses_eleven_official_targets_and_stays_within_budget() -> None:
    data = _data()
    data.x = torch.zeros((4, 9), dtype=torch.long)
    data.edge_attr = torch.zeros((8, 3), dtype=torch.long)
    data.y = torch.arange(11).float()
    graph = prepare_graph(data)
    model = CyclePEModel(dataset="peptides_struct")
    assert model(collate([graph])).shape == (1, 11)
    assert sum(p.numel() for p in model.parameters()) <= 500_000


@pytest.mark.parametrize(
    "dataset,atom_width,bond_width,target_width",
    [
        ("zinc12k", 1, 1, 1),
        ("peptides_struct", 9, 3, 11),
    ],
)
def test_edgeless_graph_preparation_and_readout(dataset, atom_width, bond_width, target_width):
    data = SimpleNamespace(
        num_nodes=1,
        x=torch.zeros((1, atom_width), dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, bond_width), dtype=torch.long),
        y=torch.zeros(target_width),
    )
    graph = prepare_graph(data)
    model = CyclePEModel(dataset=dataset, hidden=12, pe_dim=6, layers=2)
    output = model(collate([graph]))
    assert output.shape == (1, target_width)
    assert torch.isfinite(output).all()


def test_prepare_only_reports_prepared_never_passed_training(tmp_path, monkeypatch) -> None:
    graph = _graph()
    monkeypatch.setattr(
        benchmark,
        "load_benchmark",
        lambda *a, **kw: (
            {s: [graph] for s in ("train", "validation", "test")},
            {"official_splits": True, "fixture_only": True},
        ),
    )
    output = tmp_path / "result"
    assert (
        benchmark.main(
            [
                "--datasets",
                "zinc12k",
                "--prepare-only",
                "--device",
                "cpu",
                "--output-dir",
                str(output),
                "--data-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["schema_version"] == 2
    assert metrics["status"] == "prepared"
    assert metrics["datasets"]["zinc12k"]["models"] == {}
    with pytest.raises(FileExistsError):
        benchmark.main(["--datasets", "zinc12k", "--prepare-only", "--output-dir", str(output)])


def test_main_invokes_only_our_model_once_per_dataset(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(benchmark, "_validate", lambda args: None)
    monkeypatch.setattr(benchmark, "load_benchmark", lambda *a, **kw: ({}, {}))

    def fake_train(dataset, splits, args):
        calls.append(dataset)
        return {"test": 0.5, "validation": 0.4}

    monkeypatch.setattr(benchmark, "_train_model", fake_train)
    output = tmp_path / "ours"
    benchmark.main(["--output-dir", str(output), "--data-root", str(tmp_path)])
    assert calls == list(DATASETS)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["schema_version"] == 2
    for dataset in DATASETS:
        assert set(metrics["datasets"][dataset]["models"]) == {"cycle_set"}
        assert "baselines" not in metrics["datasets"][dataset]


def test_selected_test_path_reports_actual_resources_and_throughput() -> None:
    source = inspect.getsource(benchmark._evaluate_test_checkpoint)
    assert "FailureSafeResourceMonitor(" in source
    assert "@resource_failure_boundary" in source
    assert "resource_monitor.start()" in source
    assert "resource_monitor.finish(" in source
    assert '"throughput": throughput' in source
    assert '"evaluation_graphs_per_second"' in source
    assert '"optimizer_created": False' in source
