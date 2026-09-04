from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import research.conductance_gat.paper as paper_module
import research.conductance_gat.paper_data as core_data_module
import research.conductance_gat.public_data as public_data_module
from chartgat.cache import CacheWrongRequestError
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
    prepare_public_data,
    validate_public_cache,
)
from research.conductance_gat.sparse import (
    SparseIncidenceConductanceLayer,
    edge_divergence,
    edge_gradient,
    pack_graph_examples,
)


@pytest.fixture(scope="module")
def full_core():
    """Generate the real scientific protocol once; never run model training."""

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return generate_core(seed=9)
    finally:
        torch.set_num_threads(previous_threads)


def _unit_public_model_input(task: str):
    """One tensor-level model input, not a public dataset or CLI data source."""

    return {
        "graph_id": "unit-model-input",
        "x": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_features": torch.ones(2, 2),
        "y": torch.tensor([0, 1, 2]) if task == "node" else torch.tensor([1.0]),
        "task": task,
        "categorical": False,
    }


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
    examples = [
        make_example(
            graph_id="unit-nnls",
            num_nodes=7,
            family="er",
            graph_seed=31,
            excitation_seed=40 + excitation,
        )
        for excitation in range(3)
    ]
    for example in examples:
        example["observed_flux"] = torch.full_like(example["observed_flux"], float("nan"))
    metrics = node_message_nnls_metrics(examples)
    assert metrics["protocol"] == ("transductive_same-evaluation-node-messages_nnls_ceiling")
    assert metrics["graph_macro_node_message_relative_l2"] < 1.0e-5
    assert metrics["graph_macro_log_conductance_rmse"] < 1.0e-5
    assert metrics["graph_macro_conductance_pearson"] == pytest.approx(1.0, abs=1.0e-5)


def test_s1_s4_splits_and_factorial_are_leakage_safe(full_core) -> None:
    core = full_core
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
    assert s3["horizons"] == [1, 5, 10, 50]
    assert all(trajectory["states"].shape[0] == 51 for trajectory in s3["rollout_test"])
    assert core_data_module._split_counts(core) == _expected_split_counts()
    assert "tiny" not in core

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
    expected = _expected_split_counts()["s2"]
    graph_counts = {"train": 28, "validation": 8, "test": 16}
    excitations_per_graph = {"train": 4, "validation": 3, "test": 3}
    assert expected == {
        split: graph_counts[split] * excitations_per_graph[split] for split in graph_counts
    }
    assert expected == {"train": 112, "validation": 24, "test": 48}


def test_cache_manifest_is_deterministic_and_checksum_verified(
    tmp_path, monkeypatch: pytest.MonkeyPatch, full_core
) -> None:
    def cached_generation(seed):
        assert seed == 9
        return full_core

    monkeypatch.setattr(core_data_module, "generate_core", cached_generation)
    first, first_path, first_manifest = prepare_core_cache(tmp_path, seed=9)
    second, second_path, second_manifest = prepare_core_cache(tmp_path, seed=9)
    _, _, independent_manifest = prepare_core_cache(tmp_path / "independent-root", seed=9)
    assert first_path == second_path
    assert first_manifest == second_manifest
    assert first_manifest["content_sha256"] == second_manifest["content_sha256"]
    assert first_manifest == independent_manifest
    assert first["s1"]["train"][0]["graph_id"] == second["s1"]["train"][0]["graph_id"]


def test_public_reciprocal_edge_adapter_without_network() -> None:
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


def test_public_loss_weight_and_conductance_edge_encoder() -> None:
    node_sample = _unit_public_model_input("node")
    assert paper_module._public_loss_weight(node_sample["y"], "node") == node_sample["y"].numel()
    graph_sample = _unit_public_model_input("graph")
    assert paper_module._public_loss_weight(graph_sample["y"], "graph") == 1

    model = paper_module.PublicConductanceModel(
        node_sample,
        hidden=8,
        num_classes=3,
        official_molecule=False,
    )
    assert model.uses_edge_features
    assert all(parameter.requires_grad for parameter in model.edge_encoder.parameters())
    assert isinstance(model.layer, SparseIncidenceConductanceLayer)


def test_public_competitor_implementations_and_selector_are_removed() -> None:
    for name in ("NoMessageMLPLayer", "SparseGCNLayer", "SparseGATLayer", "SparseGINELayer"):
        assert not hasattr(paper_module, name)
    with pytest.raises(TypeError, match="backbone"):
        paper_module.PublicConductanceModel(
            _unit_public_model_input("node"),
            hidden=8,
            num_classes=3,
            official_molecule=False,
            backbone="gcn",
        )


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


@pytest.mark.parametrize("suite", ["core", "public", "all"])
def test_paper_cli_rejects_removed_tiny_option_before_writes(tmp_path, suite) -> None:
    with pytest.raises(SystemExit) as caught:
        paper_main(
            [
                "--suite",
                suite,
                "--tiny",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert caught.value.code == 2
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "output").exists()


def test_missing_official_data_fails_without_loader_or_fabricated_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_loader(_root):
        pytest.fail("A missing cache must not invoke the downloader without permission")

    monkeypatch.setattr(public_data_module, "_load_official", forbidden_loader)
    data_root = tmp_path / "data"
    with pytest.raises(RuntimeError, match="Official public data is not marked prepared"):
        prepare_public_data(data_root)
    assert not data_root.exists()
    assert not hasattr(public_data_module, "make_public_fixtures")


def test_legacy_fabricated_public_marker_is_rejected_before_loading(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public_root = tmp_path / "conductance_gat" / "public"
    public_root.mkdir(parents=True)
    marker = public_root / "official-ready.json"
    marker.write_text(
        json.dumps({"schema_version": public_data_module.PUBLIC_SCHEMA_VERSION, "fixture": True}),
        encoding="utf-8",
    )
    before = marker.read_bytes()
    monkeypatch.setattr(
        public_data_module, "_load_official", lambda _root: pytest.fail("must not load")
    )
    with pytest.raises(CacheWrongRequestError, match="only official public data"):
        prepare_public_data(tmp_path)
    assert marker.read_bytes() == before


def test_public_download_failure_propagates_without_generating_substitute(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_download(_root):
        raise OSError("official endpoint unavailable")

    monkeypatch.setattr(public_data_module, "_load_official", failed_download)
    with pytest.raises(OSError, match="official endpoint unavailable"):
        prepare_public_data(tmp_path, allow_download=True)
    assert not list(tmp_path.rglob("*.json"))
    with pytest.raises(FileNotFoundError):
        validate_public_cache(tmp_path)


def test_public_training_rejects_legacy_generated_payload_before_any_model() -> None:
    with pytest.raises(ValueError, match="require official data"):
        paper_module.run_public(
            {"fixture": True},
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=0.001,
            batch_size=2,
            amp=False,
            pin_memory=False,
            num_workers=0,
            seed=7,
        )


def test_public_cli_missing_real_data_never_writes_result_summary(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Official public data is not marked prepared"):
        paper_main(
            [
                "--suite",
                "public",
                "--prepare-only",
                "--device",
                "cpu",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert not (tmp_path / "data").exists()
    assert not list((tmp_path / "output").iterdir())


def test_explicit_seed_axes_route_data_and_model_randomness_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, int] = {}

    def fake_run_core(core, **kwargs):
        captured["model_seed"] = kwargs["seed"]
        return {}, [], {}

    def fake_prepare_core(data_root, *, seed):
        captured["data_seed"] = seed
        return {}, data_root / "unit-dispatch-manifest.json", {
            "cache_key": "unit-dispatch",
            "artifact_sha256": "a" * 64,
            "content_sha256": "b" * 64,
        }

    monkeypatch.setattr(paper_module, "prepare_core_cache", fake_prepare_core)
    monkeypatch.setattr(paper_module, "run_core", fake_run_core)
    summary = paper_module.main(
        [
            "--suite",
            "core",
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
    assert captured["data_seed"] == 3
    assert summary["prepared"]["core"]["data_seed"] == 3
    assert summary["configuration"]["suite_specific_epoch_cap"] is None
    assert summary["runtime"]["resource_observability"]["measurement_scope"]
    assert summary["implementation_integrity"]["valid"] is True
    assert summary["seed_axis_applicability"]["core"]["split"]["applicable"] is False
    assert summary["seed_axis_applicability"]["core"]["chart"]["applicable"] is False


def test_official_public_split_and_chart_seed_axes_are_not_applicable() -> None:
    applicability = _seed_axis_applicability("public")["public"]
    assert applicability["data"]["applicable"] is False
    assert applicability["split"]["applicable"] is False
    assert "official" in applicability["split"]["use"]
    assert applicability["chart"]["applicable"] is False
    assert applicability["model"]["applicable"] is True


def test_public_suite_receives_the_full_requested_epoch_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, int] = {}

    def fake_prepare_public(_data_root, *, allow_download):
        assert allow_download is False
        return {"fixture": False}, tmp_path / "official-ready.json", {
            "fixture": False,
            "processed_sha256": {"unit": "c" * 64},
        }

    def fake_run_public(datasets, **kwargs):
        assert datasets["fixture"] is False
        captured["epochs"] = kwargs["epochs"]
        return {}, [], {}

    monkeypatch.setattr(paper_module, "prepare_public_data", fake_prepare_public)
    monkeypatch.setattr(paper_module, "run_public", fake_run_public)
    summary = paper_main(
        [
            "--suite",
            "public",
            "--device",
            "cpu",
            "--epochs",
            "73",
            "--workers",
            "0",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert captured["epochs"] == 73
    assert summary["configuration"]["epochs_applied_to_public"] == 73
    assert summary["configuration"]["suite_specific_epoch_cap"] is None


def test_oom_is_fail_visible_and_never_changes_the_batch_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_prepare_core(data_root, *, seed):
        assert seed == 17
        return {}, data_root / "manifest.json", {
            "cache_key": "unit-oom",
            "artifact_sha256": "d" * 64,
            "content_sha256": "e" * 64,
        }

    def fail_with_oom(_core, **_kwargs):
        raise torch.cuda.OutOfMemoryError("unit OOM")

    output = tmp_path / "output"
    monkeypatch.setattr(paper_module, "prepare_core_cache", fake_prepare_core)
    monkeypatch.setattr(paper_module, "run_core", fail_with_oom)
    with pytest.raises(RuntimeError, match="No model, dataset, epoch, sampling, or batch"):
        paper_main(
            [
                "--suite",
                "core",
                "--device",
                "cpu",
                "--epochs",
                "100",
                "--batch-size",
                "16",
                "--workers",
                "0",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output),
            ]
        )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["execution_plan"]["optimization"][
        "epochs_applied_without_suite_specific_cap"
    ] == 100
    assert failure["execution_plan"]["batching"]["physical_batch_size_graphs"] == 16
    assert "automatic reduction" in failure["recovery_policy"]
    assert "smaller --batch-size" not in json.dumps(failure)
    assert failure["resources_until_failure"] is not None
    assert failure["resource_observability_unavailable_reason"] is None


def test_monitor_cleanup_failure_preserves_keyboard_interrupt_and_records_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary_error = KeyboardInterrupt("unit training interrupt")

    def fake_prepare_core(data_root, *, seed):
        assert seed == 17
        return {}, data_root / "manifest.json", {
            "cache_key": "unit-interrupt",
            "artifact_sha256": "d" * 64,
            "content_sha256": "e" * 64,
        }

    def interrupt_training(_core, **_kwargs):
        raise primary_error

    class FailingMonitor:
        instances: list[FailingMonitor] = []

        def __init__(self, _device):
            self.finish_calls = 0
            self.instances.append(self)

        def start(self):
            return {"measurement_scope": "unit monitor fixture"}

        def finish(self, **_kwargs):
            self.finish_calls += 1
            raise RuntimeError("unit monitor cleanup failure")

    output = tmp_path / "interrupted-output"
    monkeypatch.setattr(paper_module, "prepare_core_cache", fake_prepare_core)
    monkeypatch.setattr(paper_module, "run_core", interrupt_training)
    monkeypatch.setattr(paper_module, "RuntimeResourceMonitor", FailingMonitor)
    with pytest.raises(KeyboardInterrupt) as captured:
        paper_main(
            [
                "--suite",
                "core",
                "--device",
                "cpu",
                "--workers",
                "0",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output),
            ]
        )
    assert captured.value is primary_error
    assert FailingMonitor.instances[0].finish_calls == 1
    assert any("unit monitor cleanup failure" in note for note in primary_error.__notes__)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "KeyboardInterrupt"
    assert failure["error"] == "unit training interrupt"
    assert failure["resources_until_failure"] is None
    assert "unit monitor cleanup failure" in failure[
        "resource_observability_unavailable_reason"
    ]


def test_actual_training_verifies_gradient_optimizer_and_step_counts() -> None:
    train_examples = [
        make_example(
            graph_id=f"train-{index}",
            num_nodes=7 + index,
            family="er",
            graph_seed=70 + index,
            excitation_seed=80 + index,
        )
        for index in range(2)
    ]
    validation_examples = [
        make_example(
            graph_id="validation",
            num_nodes=8,
            family="rgg",
            graph_seed=91,
            excitation_seed=92,
        )
    ]
    model = SparseIncidenceConductanceLayer(2, 3, hidden_channels=8)
    report: dict[str, object] = {}
    history = paper_module.train_sparse_model(
        model,
        train_examples,
        validation_examples,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=0.003,
        batch_size=2,
        amp=False,
        pin_memory=False,
        num_workers=0,
        seed=5,
        objective="node_only",
        execution_report=report,
    )
    assert len(history) == 1
    assert report["optimizer_integrity"]["verified"] is True
    assert report["optimization"]["optimizer_steps"] == 1
    assert report["optimization"]["expected_optimizer_steps"] == 1
    assert report["first_optimizer_step"]["gradient"]["verified"] is True
    assert report["first_optimizer_step"]["changed_parameter_tensors"]
    assert report["path_integrity"]["input_forward_loss_backward_optimizer"] is True


def test_vectorized_sparse_evaluation_matches_individual_graph_metrics() -> None:
    examples = [
        make_example(
            graph_id=f"evaluation-{index}",
            num_nodes=8 + index,
            family="er" if index == 0 else "rgg",
            graph_seed=110 + index,
            excitation_seed=120 + index,
        )
        for index in range(2)
    ]
    torch.manual_seed(15)
    model = SparseIncidenceConductanceLayer(2, 3, hidden_channels=8)
    options = {
        "device": torch.device("cpu"),
        "amp": False,
        "pin_memory": False,
        "num_workers": 0,
    }
    batched = paper_module.evaluate_sparse_model(
        model, examples, batch_size=2, **options
    )
    individual = [
        paper_module.evaluate_sparse_model(model, [example], batch_size=1, **options)
        for example in examples
    ]
    for name in (
        "graph_macro_flux_relative_l2",
        "graph_macro_node_message_relative_l2",
        "graph_macro_next_state_relative_l2",
        "graph_macro_log_conductance_rmse",
        "graph_macro_conductance_pearson",
        "graph_macro_conductance_spearman",
        "excited_edge_fraction",
        "stability_cap_activation_fraction",
    ):
        assert batched[name] == pytest.approx(
            sum(result[name] for result in individual) / len(individual), abs=1.0e-6
        )
    assert batched["evaluation_batching"].startswith("vectorized")


def test_rollout_batches_independent_trajectories_without_metric_drift() -> None:
    trajectories = [
        core_data_module._make_trajectory(
            graph_id=f"rollout-{index}",
            num_nodes=7 + index,
            family="er" if index == 0 else "rgg",
            graph_seed=130 + index,
            trajectory_seed=140 + index,
            horizon=3,
        )
        for index in range(2)
    ]
    torch.manual_seed(21)
    model = SparseIncidenceConductanceLayer(2, 3, hidden_channels=8)
    batched = paper_module.evaluate_rollout(
        model,
        trajectories,
        [1, 3],
        device=torch.device("cpu"),
        amp=False,
        oracle=False,
    )
    individual = [
        paper_module.evaluate_rollout(
            model,
            [trajectory],
            [1, 3],
            device=torch.device("cpu"),
            amp=False,
            oracle=False,
        )
        for trajectory in trajectories
    ]
    for name in (
        "horizon_1_relative_l2",
        "horizon_3_relative_l2",
        "final_norm_over_initial",
        "dissipation_violation_fraction",
        "stability_cap_activation_fraction",
    ):
        assert batched[name] == pytest.approx(
            sum(result[name] for result in individual) / len(individual), abs=1.0e-6
        )
    assert "all independent trajectories" in batched["evaluation_batching"]


def test_direct_cli_worker_default_matches_parent_paper_runner() -> None:
    assert paper_module.build_parser().parse_args([]).num_workers == 4
