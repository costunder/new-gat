from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import research.conductance_gat.paper as paper_module
from research.conductance_gat.paper import (
    _normalized_loss,
    _seed_axis_applicability,
    node_message_nnls_metrics,
)
from research.conductance_gat.paper import main as paper_main
from research.conductance_gat.paper_data import (
    _expected_split_counts,
    generate_core,
    make_example,
    prepare_core_cache,
)
from research.conductance_gat.public_data import (
    deduplicate_undirected_edges,
    make_public_fixtures,
    prepare_public_data,
)
from research.conductance_gat.sparse import (
    SparseIncidenceConductanceLayer,
    edge_divergence,
    edge_gradient,
    pack_graph_examples,
)


def _explicit_incidence(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    incidence = torch.zeros(edge_index.shape[1], num_nodes, dtype=torch.float64)
    incidence[torch.arange(edge_index.shape[1]), edge_index[0]] = -1.0
    incidence[torch.arange(edge_index.shape[1]), edge_index[1]] = 1.0
    return incidence


def test_sparse_gather_scatter_matches_dense_algebra_only_in_reference_test() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    state = torch.randn(4, 3, dtype=torch.float64)
    flux = torch.randn(4, 3, dtype=torch.float64)
    incidence = _explicit_incidence(edge_index, 4)

    assert torch.allclose(edge_gradient(edge_index, state), incidence @ state)
    assert torch.allclose(edge_divergence(edge_index, flux, 4), incidence.t() @ flux)


def test_variable_graph_sparse_layer_is_positive_and_orientation_invariant() -> None:
    first = make_example(
        graph_id="first",
        num_nodes=7,
        family="er",
        graph_seed=11,
        excitation_seed=12,
    )
    second = make_example(
        graph_id="second",
        num_nodes=9,
        family="rgg",
        graph_seed=21,
        excitation_seed=22,
    )
    batch = pack_graph_examples([first, second])
    assert torch.allclose(
        first["observed_node_message"],
        edge_divergence(first["edge_index"], first["observed_flux"], 7),
    )
    torch.manual_seed(4)
    model = SparseIncidenceConductanceLayer(2, 3, hidden_channels=12).double()
    batch = batch.to(torch.device("cpu"))
    # Keep the generated float input aligned with the double precision model.
    for name, value in list(batch.__dict__.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(batch, name, value.double())
    output, diagnostics = model(batch, return_diagnostics=True)
    assert torch.all(diagnostics["conductance"] > 0)
    for graph_number in range(batch.num_graphs):
        assert torch.allclose(
            diagnostics["node_message"][batch.node_graph == graph_number].sum(dim=0),
            torch.zeros(2, dtype=torch.float64),
            atol=1e-12,
        )

    flipped = dict(first)
    flipped["edge_index"] = first["edge_index"].flip(0)
    original_batch = pack_graph_examples([first])
    flipped_batch = pack_graph_examples([flipped])
    for packed in (original_batch, flipped_batch):
        for name, value in list(packed.__dict__.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                setattr(packed, name, value.double())
    original, original_diagnostics = model(original_batch, return_diagnostics=True)
    reoriented, flipped_diagnostics = model(flipped_batch, return_diagnostics=True)
    assert torch.allclose(original, reoriented, atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        original_diagnostics["edge_flux"],
        -flipped_diagnostics["edge_flux"],
        atol=1e-11,
        rtol=1e-11,
    )

    gradient_only = SparseIncidenceConductanceLayer(
        2, 3, hidden_channels=12, mode="gradient_only"
    ).double()
    _, gradient_diagnostics = gradient_only(original_batch, return_diagnostics=True)
    _, flipped_gradient_diagnostics = gradient_only(flipped_batch, return_diagnostics=True)
    assert torch.all(gradient_diagnostics["conductance"] > 0)
    assert torch.allclose(
        gradient_diagnostics["conductance"],
        flipped_gradient_diagnostics["conductance"],
        atol=1e-11,
        rtol=1e-11,
    )


def test_training_objectives_keep_headline_independent_of_flux_labels() -> None:
    example = make_example(
        graph_id="objective",
        num_nodes=9,
        family="er",
        graph_seed=31,
        excitation_seed=32,
    )
    batch = pack_graph_examples([example])
    model = SparseIncidenceConductanceLayer(2, 3, hidden_channels=8)
    node_before, node_diagnostics = _normalized_loss(model, batch, objective="node_only")
    assert node_diagnostics["flux_relative_mse"] is None
    flux_before, _ = _normalized_loss(model, batch, objective="flux_only")
    assert batch.observed_flux is not None
    batch.observed_flux = batch.observed_flux + 10.0
    node_after, _ = _normalized_loss(model, batch, objective="node_only")
    flux_after, _ = _normalized_loss(model, batch, objective="flux_only")
    assert torch.equal(node_before, node_after)
    assert not torch.isclose(flux_before, flux_after)

    batch.observed_flux = None
    batch.true_flux = None
    node_without_flux, diagnostics = _normalized_loss(model, batch, objective="node_only")
    assert torch.isfinite(node_without_flux)
    assert diagnostics["flux_relative_mse"] is None
    with pytest.raises(ValueError, match="edge-flux target"):
        _normalized_loss(model, batch, objective="flux_only")


def test_node_message_nnls_recovers_static_conductance_without_flux_labels() -> None:
    suite = generate_core(seed=19, tiny=True)["s1"]
    examples = suite["test"]
    for example in examples:
        example["observed_flux"] = torch.full_like(example["observed_flux"], float("nan"))
    metrics = node_message_nnls_metrics(examples)
    assert metrics["protocol"] == ("transductive_same-evaluation-node-messages_nnls_ceiling")
    assert metrics["graph_macro_node_message_relative_l2"] < 1.0e-5
    assert metrics["graph_macro_log_conductance_rmse"] < 1.0e-5
    assert metrics["graph_macro_conductance_pearson"] == pytest.approx(1.0, abs=1.0e-5)


def test_s1_s4_splits_and_factorial_are_leakage_safe() -> None:
    core = generate_core(seed=7, tiny=True)
    s1 = core["s1"]
    train_ids = {example["graph_id"] for example in s1["train"]}
    validation_ids = {example["graph_id"] for example in s1["validation"]}
    test_ids = {example["graph_id"] for example in s1["test"]}
    assert train_ids.isdisjoint(validation_ids | test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert {example["graph_id"] for example in s1["seen_test"]} == train_ids

    s2 = core["s2"]
    assert {example["metadata"]["family"] for example in s2["train"]} == {"er", "rgg"}
    assert {example["metadata"]["family"] for example in s2["test"]} == {"grid", "barbell"}
    assert min(example["metadata"]["num_nodes"] for example in s2["test"]) > max(
        example["metadata"]["num_nodes"] for example in s2["train"]
    )

    s3 = core["s3"]
    assert s3["horizons"] == [1, 5, 10]
    assert all(trajectory["states"].shape[0] == 11 for trajectory in s3["rollout_test"])

    s4 = core["s4"]
    ids = [
        {example["graph_id"] for example in s4[split]} for split in ("train", "validation", "test")
    ]
    assert ids[0].isdisjoint(ids[1] | ids[2]) and ids[1].isdisjoint(ids[2])
    cells = {
        (
            example["metadata"]["contrast"],
            example["metadata"]["active_node_fraction"],
            example["metadata"]["snr_db"],
        )
        for example in s4["test"]
    }
    assert len(cells) == 18


def test_full_s2_cache_cardinality_matches_graph_and_excitation_protocol() -> None:
    # Check the full cache contract without materializing the 52 full-size
    # graphs and their 184 excitation examples.
    expected = _expected_split_counts(tiny=False)["s2"]
    graph_counts = {"train": 28, "validation": 8, "test": 16}
    excitations_per_graph = {"train": 4, "validation": 3, "test": 3}
    assert expected == {
        split: graph_counts[split] * excitations_per_graph[split] for split in graph_counts
    }
    assert expected == {"train": 112, "validation": 24, "test": 48}


def test_cache_manifest_is_deterministic_and_checksum_verified(tmp_path) -> None:
    first, first_path, first_manifest = prepare_core_cache(tmp_path, seed=9, tiny=True)
    second, second_path, second_manifest = prepare_core_cache(tmp_path, seed=9, tiny=True)
    _, _, independent_manifest = prepare_core_cache(
        tmp_path / "independent-root", seed=9, tiny=True
    )
    assert first_path == second_path
    assert first_manifest == second_manifest
    assert first_manifest["content_sha256"] == second_manifest["content_sha256"]
    assert first_manifest == independent_manifest
    assert first["s1"]["train"][0]["graph_id"] == second["s1"]["train"][0]["graph_id"]


def test_public_fixture_uses_same_reciprocal_edge_adapter_without_network(tmp_path) -> None:
    directed = torch.tensor([[0, 1, 1, 0, 2], [1, 0, 1, 2, 0]], dtype=torch.long)
    attributes = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    edges, features = deduplicate_undirected_edges(directed, attributes, 3)
    assert edges.shape == (2, 2)
    assert features.shape == (2, 2)

    reciprocal = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    continuous = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    _, averaged = deduplicate_undirected_edges(reciprocal, continuous, 2)
    torch.testing.assert_close(averaged, torch.tensor([[2.0, 4.0]]))
    categorical = torch.tensor([[1, 2], [1, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="conflicting categorical reciprocal"):
        deduplicate_undirected_edges(reciprocal, categorical, 2)

    fixture = make_public_fixtures(5)
    assert fixture["fixture"] is True
    labels = [int(graph["y"].item()) for graph in fixture["ogbg_molhiv"]["test"]]
    assert set(labels) == {0, 1}
    _, marker, manifest = prepare_public_data(tmp_path, seed=5, tiny=True)
    assert marker.exists() and manifest["fixture"] is True


def test_public_loss_weight_and_inactive_edge_encoders_match_active_computation() -> None:
    fixture = make_public_fixtures(7)
    node_sample = fixture["pascalvoc_sp"]["train"][0]
    assert paper_module._public_loss_weight(node_sample["y"], "node") == node_sample["y"].numel()
    graph_sample = fixture["ogbg_molhiv"]["train"][0]
    assert paper_module._public_loss_weight(graph_sample["y"], "graph") == 1

    gcn = paper_module.PublicConductanceModel(
        node_sample,
        hidden=8,
        num_classes=3,
        official_molecule=False,
        backbone="gcn",
    )
    gat = paper_module.PublicConductanceModel(
        node_sample,
        hidden=8,
        num_classes=3,
        official_molecule=False,
        backbone="gat",
    )
    assert not gcn.uses_edge_features
    assert not any(parameter.requires_grad for parameter in gcn.edge_encoder.parameters())
    assert gat.uses_edge_features
    assert all(parameter.requires_grad for parameter in gat.edge_encoder.parameters())


def test_cli_refuses_nonempty_output_without_touching_existing_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text('{"status":"previous"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        paper_main(
            [
                "--suite",
                "core",
                "--prepare-only",
                "--tiny",
                "--device",
                "cpu",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output),
            ]
        )
    assert sentinel.read_text(encoding="utf-8") == '{"status":"previous"}\n'
    assert list(output.iterdir()) == [sentinel]
    assert not (tmp_path / "data").exists()


def test_tiny_all_cli_writes_machine_readable_results(tmp_path) -> None:
    output = tmp_path / "output"
    summary = paper_main(
        [
            "--suite",
            "all",
            "--tiny",
            "--device",
            "cpu",
            "--no-amp",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--workers",
            "0",
            "--seed",
            "13",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
        ]
    )
    assert summary["runtime"]["amp"] is False
    assert summary["seed_axes"] == {"data": 13, "split": 13, "chart": 13, "model": 13}
    assert summary["results"]["public"]["pascalvoc_sp"]["fixture"] is True
    expected_public_baselines = {
        "no_message_mlp",
        "gcn",
        "gat",
        "gine",
        "conductance_model",
    }
    for dataset_name in ("pascalvoc_sp", "ogbg_molhiv"):
        public_result = summary["results"]["public"][dataset_name]
        assert set(public_result["baselines"]) == expected_public_baselines
        assert all(result["parameter_count"] > 0 for result in public_result["baselines"].values())
        assert public_result["comparison_protocol"]["backbone_depth"] == 1
    assert summary["results"]["core"]["s3"]["baselines"]["oracle"]["rollout"][
        "horizon_10_relative_l2"
    ] == pytest.approx(0.0)
    for suite_name in ("s1", "s2", "s3", "s4"):
        core_result = summary["results"]["core"][suite_name]
        assert core_result["headline_baseline"] == "full"
        assert core_result["baselines"]["full"]["training_objective"] == "node_only"
        assert core_result["baselines"]["full_flux_supervised"]["training_objective"] == "flux_only"
        assert core_result["baselines"]["full_joint"]["training_objective"] == "joint"
        assert core_result["baselines"]["gradient_only"]["training_objective"] == "node_only"
    assert "node_message_nnls" in summary["results"]["core"]["s1"]["baselines"]
    assert "node_message_nnls" in summary["results"]["core"]["s4"]["baselines"]
    assert (output / "summary.json").exists()
    assert (output / "metrics.csv").exists()
    assert (output / "history.csv").exists()
    assert (output / "models.pt").exists()
    history_header = (output / "history.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "training_objective" in history_header
    parsed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert parsed["scope"] == "independent_sparse_incidence_conductance_attention"


def test_explicit_seed_axes_route_data_and_model_randomness_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, int] = {}

    def fake_run_core(core, **kwargs):
        captured["model_seed"] = kwargs["seed"]
        return {}, [], {}

    monkeypatch.setattr(paper_module, "run_core", fake_run_core)
    summary = paper_module.main(
        [
            "--suite",
            "core",
            "--tiny",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--seed",
            "99",
            "--data-seed",
            "3",
            "--split-seed",
            "4",
            "--chart-seed",
            "5",
            "--model-seed",
            "6",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert "seed" not in summary
    assert summary["seed_axes"] == {"data": 3, "split": 4, "chart": 5, "model": 6}
    assert captured["model_seed"] == 6
    core_manifest = json.loads(
        Path(summary["prepared"]["core"]["manifest"]).read_text(encoding="utf-8")
    )
    assert core_manifest["request"]["seed"] == 3
    assert summary["prepared"]["core"]["data_seed"] == 3
    assert summary["seed_axis_applicability"]["core"]["split"]["applicable"] is False
    assert summary["seed_axis_applicability"]["core"]["chart"]["applicable"] is False


def test_official_public_split_and_chart_seed_axes_are_not_applicable() -> None:
    applicability = _seed_axis_applicability("public", public_fixture=False)["public"]
    assert applicability["data"]["applicable"] is False
    assert applicability["split"]["applicable"] is False
    assert "official" in applicability["split"]["use"]
    assert applicability["chart"]["applicable"] is False
    assert applicability["model"]["applicable"] is True
