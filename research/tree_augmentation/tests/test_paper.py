"""Offline fixtures for the independent tree-augmentation paper path."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix, validate_spanning_tree
from chartgat.cache import CacheCorruptError, CacheWrongRequestError
from chartgat.seeds import SeedAxes
from research.tree_augmentation import paper as tree_paper
from research.tree_augmentation import paper_model
from research.tree_augmentation.paper import main, run_suite
from research.tree_augmentation.paper_data import (
    GraphRecord,
    OptionalDatasetError,
    _cache_records,
    _load_cached_dataset,
    build_paper_chart,
    prepare_cyclecount_dataset,
    prepare_optional_pyg_dataset,
    simple_cycle_counts,
    traversal_tree_indices,
    validate_prepared_cache,
    wilson_ust_indices,
    zinc_record_from_pyg,
)
from research.tree_augmentation.paper_model import (
    GraphChartView,
    VariableBetaCycleEncoder,
    _validate_first_step_gradients,
    build_chart_views,
    collate_chart_views,
)


def test_wilson_is_deterministic_valid_and_uniform_on_triangle() -> None:
    edges = ((0, 1), (0, 2), (1, 2))
    first = wilson_ust_indices(3, edges, seed=19, root=2)
    second = wilson_ust_indices(3, edges, seed=19, root=2)
    np.testing.assert_array_equal(first, second)
    validate_spanning_tree(incidence_matrix(3, edges), first)

    counts = Counter(
        tuple(int(index) for index in wilson_ust_indices(3, edges, seed=seed))
        for seed in range(1_500)
    )
    assert set(counts) == {(0, 1), (0, 2), (1, 2)}
    frequencies = np.asarray(list(counts.values()), dtype=np.float64) / 1_500
    assert np.max(np.abs(frequencies - 1.0 / 3.0)) < 0.04


def test_random_root_traversals_and_legacy_sampler_stay_separate() -> None:
    edges = ((0, 1), (0, 3), (1, 2), (2, 3), (0, 2))
    root_zero = traversal_tree_indices(4, edges, method="bfs", root=0)
    root_two = traversal_tree_indices(4, edges, method="bfs", root=2)
    assert tuple(root_zero) != tuple(root_two)
    for method in ("bfs", "dfs", "wilson_ust", "random_priority_kruskal"):
        chart = build_paper_chart(4, edges, method=method, seed=11, root=1)
        validate_spanning_tree(incidence_matrix(4, edges), chart.tree_edge_indices)
        assert chart.beta == 2
    assert "wilson_ust" in build_paper_chart(4, edges, method="wilson_ust", seed=11, root=1).name
    assert (
        "random_priority_kruskal"
        in build_paper_chart(4, edges, method="random_priority_kruskal", seed=11, root=1).name
    )


def test_cyclecount_target_is_chart_independent() -> None:
    triangle_and_square = (
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 5),
    )
    assert simple_cycle_counts(6, triangle_and_square) == (1, 1, 0, 0)
    bfs = build_paper_chart(6, triangle_and_square, method="bfs", seed=1, root=0)
    ust = build_paper_chart(6, triangle_and_square, method="wilson_ust", seed=9, root=4)
    assert bfs.beta == ust.beta == 2
    assert simple_cycle_counts(6, triangle_and_square) == (1, 1, 0, 0)


def _view(record: GraphRecord) -> GraphChartView:
    chart = build_paper_chart(record.num_nodes, record.edges, method="bfs", seed=3, root=0)
    return GraphChartView(
        graph_id=record.graph_id,
        graph_family=record.family,
        graph_status="id",
        chart_status="seen",
        num_nodes=record.num_nodes,
        edges=record.edges,
        basis=chart.basis,
        target=record.target,
        chart_name=chart.name,
        tree_key=tuple(int(index) for index in chart.tree_edge_indices),
        x=record.x,
        edge_attr=record.edge_attr,
    )


def test_output_dim_comes_from_declared_target_metadata() -> None:
    dataset = SimpleNamespace(
        suite="csl",
        target_names=("class_0", "class_1", "class_2"),
        records=(SimpleNamespace(split="train", target=(0.0,)),),
    )
    assert tree_paper._output_dim(dataset) == 3
    dataset.target_names = ()
    with pytest.raises(ValueError, match="target_names metadata"):
        tree_paper._output_dim(dataset)


def test_cache_integrity_and_model_split_usage_are_separate() -> None:
    dataset = SimpleNamespace(
        suite="csl",
        records=tuple(
            SimpleNamespace(
                split=split,
                num_nodes=4,
                edges=((0, 1), (1, 2), (2, 3)),
                target=(0.0,),
            )
            for split in ("train", "validation", "test")
        ),
        target_names=("target",),
    )
    assert tree_paper._dataset_cache_integrity(dataset) == {
        "full_cache_loaded": True,
        "all_declared_splits_validated": True,
        "loaded_and_validated_splits": ["test", "train", "validation"],
    }
    assert tree_paper._model_split_usage(
        dataset, evaluation_scope="validation", prepare_only=False
    ) == {
        "fit_splits": ["train"],
        "evaluation_splits": ["validation"],
        "selection_splits": ["validation"],
        "test_evaluated": False,
        "test_used_for_selection": False,
    }
    selected_test = tree_paper._model_split_usage(
        dataset, evaluation_scope="selected_test", prepare_only=False
    )
    assert selected_test["fit_splits"] == []
    assert selected_test["evaluation_splits"] == ["test"]
    assert selected_test["test_evaluated"] is True
    assert selected_test["test_used_for_selection"] is False
    data_observability = tree_paper._data_observability(
        dataset,
        {"train": 1, "validation": 1, "test": 1},
        tree_paper._model_split_usage(
            dataset, evaluation_scope="validation", prepare_only=False
        ),
    )
    assert data_observability["full_dataset_graphs"] == 3
    assert data_observability["model_consumed_graphs"] == 2
    assert data_observability["model_consumed_fraction"] == pytest.approx(2 / 3)
    assert data_observability["subset_or_fast_mode"] is False
    assert data_observability["sampling_ratio"]["value"] == 1.0
    assert data_observability["graph_statistics"]["nodes_per_graph"]["mean"]["value"] == 4
    batch_observability = tree_paper._batch_observability(64, workers=4)
    assert batch_observability["effective_batch_size"] == 64
    assert batch_observability["data_loader"] == {
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "persistent_workers_reason": (
            "each seeded training/evaluation DataLoader is consumed once in full; "
            "persistence does not span separately constructed loader instances"
        ),
    }
    assert tree_paper._optimization_observability(
        training_performed=True, updates=800
    )["total_actual_optimizer_steps"] == 1600


def test_tree_default_workers_and_loader_configuration_are_explicit() -> None:
    args = tree_paper._parser().parse_args([])
    assert args.workers == 4
    assert tree_paper.run_suite.__kwdefaults__["workers"] == 4
    assert paper_model.data_loader_configuration(4) == {
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    assert paper_model.data_loader_configuration(0) == {
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
    }
    with pytest.raises(ValueError, match="non-negative"):
        paper_model.data_loader_configuration(-1)


def test_tree_runtime_records_exact_loader_parallelism() -> None:
    runtime = tree_paper._runtime_metadata(
        device=torch.device("cpu"),
        amp_requested=False,
        pin_memory=False,
        non_blocking=False,
        batch_size=16,
        workers=4,
        elapsed_seconds=1.0,
    )
    assert runtime["workers"] == 4
    assert runtime["persistent_workers"] is False
    assert runtime["prefetch_factor"] == 2
    assert runtime["data_loader"]["num_workers"] == 4
    assert "one complete pass" in runtime["data_loader"][
        "persistent_workers_reason"
    ]


def test_keyboard_interrupt_preserves_original_when_failure_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = []

    def write_then_fail(path, payload):
        writes.append((path, payload["status"]))
        if len(writes) == 2:
            raise OSError("unit failure recorder")

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("unit original interrupt")

    monkeypatch.setattr(tree_paper, "_write_json", write_then_fail)
    monkeypatch.setattr(tree_paper, "_prepare_dataset", interrupt)
    with pytest.raises(KeyboardInterrupt, match="unit original interrupt") as caught:
        run_suite(
            "core",
            data_root=tmp_path / "data",
            output_dir=tmp_path / "result",
            requested_device="cpu",
            seed=0,
            prepare_only=True,
            amp_override=False,
            batch_size_override=None,
            pin_memory_override=False,
            non_blocking_override=False,
            workers=0,
        )
    assert writes == [
        (tmp_path / "result" / "manifest.json", "preparing"),
        (tmp_path / "result" / "manifest.json", "failed"),
    ]
    notes = getattr(caught.value, "__notes__", [])
    assert any("without replacing the original error" in note for note in notes)


def test_variable_beta_batch_masks_tree_cycle_and_multicycle() -> None:
    records = (
        GraphRecord("tree", "fixture", "train", 4, ((0, 1), (1, 2), (2, 3)), (0.0,)),
        GraphRecord(
            "cycle",
            "fixture",
            "train",
            4,
            ((0, 1), (0, 3), (1, 2), (2, 3)),
            (1.0,),
        ),
        GraphRecord(
            "multi",
            "fixture",
            "train",
            5,
            ((0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)),
            (2.0,),
        ),
    )
    batch = collate_chart_views([_view(record) for record in records])
    assert batch.basis.shape == (3, 6, 2)
    assert batch.cycle_mask.sum(dim=1).tolist() == [0, 1, 2]
    assert batch.edge_mask.sum(dim=1).tolist() == [3, 4, 6]
    assert torch.all(batch.node_categories[batch.node_mask] == 28)
    assert torch.all(batch.edge_categories[batch.edge_mask] == 4)
    output = VariableBetaCycleEncoder(hidden_dim=8, output_dim=2)(batch)
    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()
    stats = tree_paper._view_stats([_view(record) for record in records])
    assert stats["nodes_per_graph"] == {
        "minimum": 4,
        "mean": pytest.approx(13 / 3),
        "median": 4.0,
        "maximum": 5,
    }
    assert stats["edges_per_graph"]["maximum"] == 6
    assert stats["cycle_rank_per_graph"] == {
        "minimum": 0,
        "mean": 1.0,
        "median": 1.0,
        "maximum": 2,
    }
    assert stats["collated_input_shape_contract"]["padding_is_excluded_by_masks"] is True


def test_tree_model_connects_every_trainable_parameter_to_task_loss_and_optimizer() -> None:
    records = (
        GraphRecord(
            "cycle-a",
            "fixture",
            "train",
            4,
            ((0, 1), (0, 3), (1, 2), (2, 3)),
            (1.0, -0.5),
        ),
        GraphRecord(
            "cycle-b",
            "fixture",
            "train",
            5,
            ((0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)),
            (-0.25, 0.75),
        ),
    )
    batch = collate_chart_views([_view(record) for record in records])
    model = VariableBetaCycleEncoder(hidden_dim=8, output_dim=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert optimizer_parameter_ids == {id(parameter) for parameter in trainable.values()}

    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    loss = torch.nn.functional.mse_loss(model(batch), batch.targets)
    loss.backward()
    _validate_first_step_gradients(model)
    assert all(parameter.grad is not None for parameter in trainable.values())
    optimizer.step()
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in trainable.items()
    )


def _gauge_fixture_view() -> GraphChartView:
    record = GraphRecord(
        "gauge-fixture",
        "fixture",
        "train",
        5,
        ((0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)),
        (2.0,),
        x=(1, 2, 3, 4, 5),
        edge_attr=(0, 1, 2, 3, 1, 0),
    )
    return _view(record)


def _gauge_predictions(views: list[GraphChartView]) -> torch.Tensor:
    torch.manual_seed(109)
    model = VariableBetaCycleEncoder(hidden_dim=12, output_dim=2).eval()
    with torch.no_grad():
        return model(collate_chart_views(views))


def test_encoder_ignores_legal_edge_orientation_and_cycle_column_gauges() -> None:
    original = _gauge_fixture_view()
    orientation_signs = np.asarray((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    reoriented_edges = tuple(
        (v, u) if orientation_signs[index] < 0 else (u, v)
        for index, (u, v) in enumerate(original.edges)
    )
    reoriented_basis, chords = fundamental_cycle_basis(
        incidence_matrix(original.num_nodes, reoriented_edges),
        original.tree_key,
        return_chords=True,
    )
    expected_basis = (
        orientation_signs[:, None] * original.basis * orientation_signs[chords][None, :]
    )
    np.testing.assert_allclose(reoriented_basis, expected_basis, atol=1e-12)
    reoriented = replace(original, edges=reoriented_edges, basis=reoriented_basis)

    column_order = np.asarray((1, 0))
    column_signs = np.asarray((-1.0, 1.0))
    signed_column_permutation = replace(
        original,
        basis=original.basis[:, column_order] * column_signs[None, :],
    )
    predictions = _gauge_predictions([original, reoriented, signed_column_permutation])
    torch.testing.assert_close(predictions[1], predictions[0], atol=1e-7, rtol=0.0)
    torch.testing.assert_close(predictions[2], predictions[0], atol=1e-7, rtol=0.0)


def test_encoder_ignores_aligned_edge_order_permutations() -> None:
    original = _gauge_fixture_view()
    edge_order = np.asarray((4, 2, 0, 5, 1, 3))
    old_to_new = {int(old): new for new, old in enumerate(edge_order)}
    reordered = replace(
        original,
        edges=tuple(original.edges[index] for index in edge_order),
        basis=original.basis[edge_order],
        edge_attr=tuple(original.edge_attr[index] for index in edge_order),
        tree_key=tuple(sorted(old_to_new[index] for index in original.tree_key)),
    )
    predictions = _gauge_predictions([original, reordered])
    torch.testing.assert_close(predictions[1], predictions[0], atol=1e-7, rtol=0.0)


def test_encoder_ignores_same_tree_node_relabeling_with_mapped_chemistry() -> None:
    original = _gauge_fixture_view()
    old_to_new_node = (4, 1, 3, 0, 2)
    mapped_edges = []
    for old_edge_index, (u, v) in enumerate(original.edges):
        mapped_u, mapped_v = old_to_new_node[u], old_to_new_node[v]
        mapped_edges.append(((min(mapped_u, mapped_v), max(mapped_u, mapped_v)), old_edge_index))
    mapped_edges.sort()
    relabeled_edges = tuple(edge for edge, _ in mapped_edges)
    old_to_new_edge = {
        old_edge_index: new_edge_index
        for new_edge_index, (_, old_edge_index) in enumerate(mapped_edges)
    }
    relabeled_tree = tuple(
        sorted(old_to_new_edge[old_edge_index] for old_edge_index in original.tree_key)
    )
    relabeled_basis = fundamental_cycle_basis(
        incidence_matrix(original.num_nodes, relabeled_edges), relabeled_tree
    )
    relabeled_x = [0] * original.num_nodes
    for old_node, new_node in enumerate(old_to_new_node):
        relabeled_x[new_node] = original.x[old_node]
    relabeled = replace(
        original,
        edges=relabeled_edges,
        basis=relabeled_basis,
        tree_key=relabeled_tree,
        x=tuple(relabeled_x),
        edge_attr=tuple(original.edge_attr[old_index] for _, old_index in mapped_edges),
    )
    predictions = _gauge_predictions([original, relabeled])
    torch.testing.assert_close(predictions[1], predictions[0], atol=1e-7, rtol=0.0)


def test_core_cache_is_deterministic_and_graph_splits_are_disjoint(tmp_path: Path) -> None:
    first = prepare_cyclecount_dataset(tmp_path, seed=31)
    second = prepare_cyclecount_dataset(tmp_path, seed=31)
    assert first.data_sha256 == second.data_sha256
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    split_sets = [set(ids) for ids in manifest["split_graph_ids"].values()]
    for index, left in enumerate(split_sets):
        for right in split_sets[index + 1 :]:
            assert left.isdisjoint(right)
    assert manifest["graph_split_before_chart_sampling"] is True
    assert {name: len(ids) for name, ids in manifest["split_graph_ids"].items()} == {
        "train": 128,
        "validation": 24,
        "id_test": 40,
        "ood_test": 40,
    }
    assert manifest["profile"] == "full"
    assert "tiny" not in manifest

    # Old full-cache manifests remain valid without rewriting the cached data.
    manifest.pop("profile")
    manifest["tiny"] = False
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_prepared_cache("core", tmp_path, seed=31).data_sha256 == first.data_sha256
    manifest["tiny"] = True
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CacheWrongRequestError, match="seed/profile mismatch"):
        prepare_cyclecount_dataset(tmp_path, seed=31)


def test_optional_pyg_adapter_has_actionable_dependency_error(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch_geometric") is not None:
        pytest.skip("PyG is installed; download behavior is environment-specific")
    with pytest.raises(OptionalDatasetError, match="torch-geometric"):
        prepare_optional_pyg_dataset("csl", tmp_path, seed=1, allow_download=True)


@pytest.mark.parametrize("suite", ["csl", "zinc"])
def test_optional_pyg_adapter_requires_explicit_download_permission(
    tmp_path: Path,
    suite: str,
) -> None:
    with pytest.raises(OptionalDatasetError, match="--allow-download"):
        prepare_optional_pyg_dataset(suite, tmp_path, seed=1)
    assert not list(tmp_path.rglob("*.json"))


def test_dataset_seed_axes_route_to_their_declared_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    sentinel = object()

    def prepare_core(data_root: Path, *, seed: int) -> object:
        calls.append(("core", seed))
        return sentinel

    def prepare_public(
        suite: str,
        data_root: Path,
        *,
        seed: int,
        allow_download: bool,
    ) -> object:
        calls.append((suite, seed))
        return sentinel

    monkeypatch.setattr(tree_paper, "prepare_cyclecount_dataset", prepare_core)
    monkeypatch.setattr(tree_paper, "prepare_optional_pyg_dataset", prepare_public)
    axes = SeedAxes(data=11, split=13, chart=17, model=19)
    for suite in ("core", "csl", "zinc"):
        assert (
            tree_paper._prepare_dataset(
                suite,
                tmp_path,
                seed_axes=axes,
                allow_download=False,
            )
            is sentinel
        )
    assert calls == [("core", 11), ("csl", 13), ("zinc", 11)]


def _pyg_like_zinc_fixture() -> SimpleNamespace:
    # Directed arcs are deliberately not in canonical undirected order.
    arcs = (
        (2, 3, 3),
        (1, 3, 2),
        (0, 2, 1),
        (0, 1, 0),
        (3, 2, 3),
        (3, 1, 2),
        (2, 0, 1),
        (1, 0, 0),
        (1, 2, 1),
        (2, 1, 1),
    )
    return SimpleNamespace(
        num_nodes=4,
        x=torch.tensor([[3], [7], [2], [11]], dtype=torch.long),
        edge_index=torch.tensor(
            [[u for u, _, _ in arcs], [v for _, v, _ in arcs]], dtype=torch.long
        ),
        edge_attr=torch.tensor([[kind] for _, _, kind in arcs], dtype=torch.long),
        y=torch.tensor([0.375], dtype=torch.float32),
    )


def test_zinc_pyg_fixture_chemistry_is_lossless_and_cache_roundtrips(
    tmp_path: Path,
) -> None:
    record = zinc_record_from_pyg(
        _pyg_like_zinc_fixture(), graph_id="zinc-test-00000", split="test"
    )
    assert record.x == (3, 7, 2, 11)
    assert record.edges == ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
    assert record.edge_attr == (0, 1, 1, 2, 3)

    # Exercise serialization directly; the public loader must reject a one-record dataset.
    prepared = _cache_records(
        suite="zinc",
        records=(record,),
        data_path=tmp_path / "unit-record.json",
        manifest_path=tmp_path / "unit-record.manifest.json",
        target_names=("constrained_logP",),
        task_type="regression",
        source="unit-test-only",
        seed=9,
    )
    loaded = _load_cached_dataset(
        suite="zinc", data_path=prepared.data_path, manifest_path=prepared.manifest_path
    )
    assert loaded.records == prepared.records == (record,)
    payload = json.loads(prepared.data_path.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == 2
    assert payload["records"][0]["x"] == [3, 7, 2, 11]
    assert payload["records"][0]["edge_attr"] == [0, 1, 1, 2, 3]
    assert "canonical undirected edge" in manifest["categorical_feature_schema"]["edge_attr"]


def test_public_loader_rejects_reduced_records_without_creating_paper_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = zinc_record_from_pyg(_pyg_like_zinc_fixture(), graph_id="unit-zinc", split="train")
    monkeypatch.setattr(
        "research.tree_augmentation.paper_data._prepare_zinc_records", lambda _root: (record,)
    )
    with pytest.raises(CacheCorruptError, match="split cardinalities"):
        prepare_optional_pyg_dataset("zinc", tmp_path, seed=9, allow_download=True)
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("suite", ["csl", "zinc"])
def test_public_loader_rejects_reduced_cache_even_with_valid_checksum(
    tmp_path: Path, suite: str
) -> None:
    record = zinc_record_from_pyg(_pyg_like_zinc_fixture(), graph_id="unit-record", split="train")
    source = "PyG:ZINC(subset=True)"
    if suite == "csl":
        record = replace(record, family="CSL", task_type="classification", target=(0.0,))
        source = "PyG:GNNBenchmarkDataset/CSL"
    cache = tmp_path / f"{suite}_pyg_v2"
    _cache_records(
        suite=suite,
        records=(record,),
        data_path=cache / "seed-9-full.json",
        manifest_path=cache / "seed-9-full.manifest.json",
        target_names=("unit-target",),
        task_type=record.task_type,
        source=source,
        seed=9,
    )
    with pytest.raises(CacheCorruptError, match="split cardinalities"):
        prepare_optional_pyg_dataset(suite, tmp_path, seed=9)


@pytest.mark.parametrize("suite", ["csl", "zinc"])
def test_public_download_failure_does_not_generate_substitute_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: str
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OptionalDatasetError("public download unavailable")

    monkeypatch.setattr(
        f"research.tree_augmentation.paper_data._prepare_{suite}_records", unavailable
    )
    with pytest.raises(OptionalDatasetError, match="public download unavailable"):
        prepare_optional_pyg_dataset(suite, tmp_path, seed=9, allow_download=True)
    assert not list(tmp_path.rglob("*.json"))


def test_cli_rejects_tiny_and_keeps_full_reference_settings() -> None:
    with pytest.raises(SystemExit) as caught:
        tree_paper._parser().parse_args(["--tiny"])
    assert caught.value.code == 2
    settings, _ = tree_paper._load_settings()
    assert settings["hidden_dim"] == 128
    assert settings["message_layers"] == 8
    assert settings["optimizer_updates"] == 800
    assert settings["batch_size"] == 16
    assert settings["train_charts_per_graph"] == settings["eval_charts_per_graph"] == 8
    assert "tiny" not in settings


def test_chemistry_is_chart_invariant_and_changes_model_input_and_prediction() -> None:
    record = zinc_record_from_pyg(
        _pyg_like_zinc_fixture(), graph_id="zinc-test-00000", split="train"
    )
    chart_views = build_chart_views(
        [record],
        chart_status="seen",
        count=2,
        methods=("bfs", "dfs"),
        roots=(0, 3),
        seed=13,
        require_distinct=True,
    )
    batch = collate_chart_views(chart_views)
    assert chart_views[0].tree_key != chart_views[1].tree_key
    assert not torch.equal(batch.basis[0], batch.basis[1])
    assert torch.equal(batch.edge_index[0], batch.edge_index[1])
    assert torch.equal(batch.node_categories[0], batch.node_categories[1])
    assert torch.equal(batch.edge_categories[0], batch.edge_categories[1])

    changed = replace(
        record,
        x=(4, *record.x[1:]),
        edge_attr=(*record.edge_attr[:-1], 0),
    )
    original_view = build_chart_views(
        [record],
        chart_status="seen",
        count=1,
        methods=("bfs",),
        roots=(0,),
        seed=17,
    )[0]
    changed_view = build_chart_views(
        [changed],
        chart_status="seen",
        count=1,
        methods=("bfs",),
        roots=(0,),
        seed=17,
    )[0]
    chemistry_batch = collate_chart_views([original_view, changed_view])
    assert torch.equal(chemistry_batch.basis[0], chemistry_batch.basis[1])
    assert not torch.equal(chemistry_batch.node_categories[0], chemistry_batch.node_categories[1])
    assert not torch.equal(chemistry_batch.edge_categories[0], chemistry_batch.edge_categories[1])
    torch.manual_seed(101)
    model = VariableBetaCycleEncoder(hidden_dim=12, output_dim=1).eval()
    prediction = model(chemistry_batch)
    assert not torch.allclose(prediction[0], prediction[1])


def test_core_orchestration_with_unit_test_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = tuple(
        GraphRecord(
            f"unit-{split}-{index}",
            "unit-test-only",
            split,
            4,
            ((0, 1), (0, 3), (1, 2), (2, 3)),
            (0.0, 1.0, 0.0, 0.0),
        )
        for split in ("train", "validation", "id_test", "ood_test")
        for index in range(2)
    )
    dataset = _cache_records(
        suite="core",
        records=records,
        data_path=tmp_path / "unit-records.json",
        manifest_path=tmp_path / "unit-records.manifest.json",
        target_names=("cycles_len_3", "cycles_len_4", "cycles_len_5", "cycles_len_6"),
        task_type="regression",
        source="unit-test-only",
        seed=43,
    )
    settings, config_path = tree_paper._load_settings()
    settings.update(
        hidden_dim=8, optimizer_updates=2, train_charts_per_graph=3, eval_charts_per_graph=2
    )
    monkeypatch.setattr(tree_paper, "_load_settings", lambda: (settings, config_path))
    monkeypatch.setattr(tree_paper, "_prepare_dataset", lambda *_args, **_kwargs: dataset)
    summary = run_suite(
        "core",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        requested_device="cpu",
        seed=43,
        prepare_only=False,
        amp_override=False,
        batch_size_override=4,
        pin_memory_override=False,
        non_blocking_override=False,
        workers=0,
        allow_download=False,
    )
    assert summary["runtime"]["workers"] == 0
    assert summary["seed_axes"] == {"data": 43, "split": 43, "chart": 43, "model": 43}
    assert "seed" not in summary
    assert summary["protocol"] == "cyclecount_graph_x_fresh_chart_family_2x2_v2"
    assert "tiny" not in summary
    assert summary["comparison"]["projector_target_used"] is False
    expected = {
        "id_graph_fresh_chart_seen_family",
        "id_graph_fresh_chart_unseen_family",
        "ood_graph_fresh_chart_seen_family",
        "ood_graph_fresh_chart_unseen_family",
    }
    for model_name in ("fixed_bfs", "multi_chart"):
        quadrants = summary["models"][model_name]["quadrants"]
        assert set(quadrants) == expected
        assert all(
            np.isfinite(value) for metrics in quadrants.values() for value in metrics.values()
        )
    manifest = json.loads((tmp_path / "results" / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["seed_axes"] == summary["seed_axes"]
    assert "seed" not in manifest
    assert manifest["protocol"] == summary["protocol"]
    assert manifest["runtime"]["device"] == "cpu"
    assert manifest["runtime"]["amp_effective"] is False
    assert manifest["sampler_protocol"]["train_multi"] == [
        "bfs_random_root",
        "dfs_random_root",
    ]
    assert manifest["sampler_protocol"]["fresh_chart_unseen_family"] == ["wilson_ust"]
    assert manifest["sampler_protocol"]["exact_tree_overlap_between_families_allowed"] is True
    assert manifest["sampler_protocol"]["wilson_draws_conditioned_on_bfs_outputs"] is False
    assert summary["sampler_protocol"] == manifest["sampler_protocol"]
    assert set(summary["view_counts"]["fixed_train"]["sampler_counts"]) == {"bfs"}
    assert set(summary["view_counts"]["fixed_train"]["chart_status_counts"]) == {
        "train_fixed_bfs_family"
    }
    assert set(summary["view_counts"]["multi_train"]["sampler_counts"]) == {"bfs", "dfs"}
    assert set(summary["view_counts"]["multi_train"]["chart_status_counts"]) == {
        "train_multi_bfs_dfs_families"
    }
    for axis, stats in summary["view_counts"]["evaluation"].items():
        expected_sampler = "wilson_ust" if "unseen_family" in axis else "bfs"
        expected_status = (
            "fresh_chart_unseen_family" if "unseen_family" in axis else "fresh_chart_seen_family"
        )
        assert set(stats["sampler_counts"]) == {expected_sampler}
        assert set(stats["chart_status_counts"]) == {expected_status}
    assert set(summary["fresh_axis_exact_tree_overlap"]) == {"id_graph", "ood_graph"}

    repeated = run_suite(
        "core",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "repeated-results",
        requested_device="cpu",
        seed=999,
        data_seed=43,
        split_seed=43,
        chart_seed=43,
        model_seed=43,
        prepare_only=False,
        amp_override=False,
        batch_size_override=4,
        pin_memory_override=False,
        non_blocking_override=False,
        workers=0,
        allow_download=False,
    )
    assert repeated["models"] == summary["models"]
    assert repeated["comparison"] == summary["comparison"]


def test_prepare_only_all_attempts_every_suite_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "all"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper",
            "--suite",
            "all",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
            "--seed",
            "999",
            "--data-seed",
            "5",
            "--split-seed",
            "7",
            "--chart-seed",
            "11",
            "--model-seed",
            "13",
            "--prepare-only",
            "--workers",
            "0",
        ],
    )
    assert main() == 2
    aggregate = json.loads((output / "manifest.json").read_text("utf-8"))
    assert aggregate["status"] == "failed"
    assert aggregate["seed_axes"] == {"data": 5, "split": 7, "chart": 11, "model": 13}
    assert "seed" not in aggregate
    assert aggregate["suites"]["core"]["status"] == "prepared"
    assert aggregate["suites"]["csl"]["status"] == "failed"
    assert aggregate["suites"]["zinc"]["status"] == "failed"
    assert "--allow-download" in aggregate["suites"]["csl"]["error"]
    assert (output / "zinc" / "manifest.json").is_file()
