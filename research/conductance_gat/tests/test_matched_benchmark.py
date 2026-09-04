"""Unit fixtures only: no public downloads and no CPU/GPU benchmark training."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from chartgat.cache import CacheCorruptError, CacheIncompleteError
from research.conductance_gat import benchmark, benchmark_data


@pytest.fixture
def payload(monkeypatch):
    # Reduced dimensions exist only in this test fixture, never a production dataset path.
    monkeypatch.setitem(
        benchmark_data.EXPECTED,
        "cora",
        {
            "nodes": 8,
            "features": 3,
            "classes": 2,
            "splits": [3, 2, 2],
        },
    )
    arcs, incidence = benchmark_data.canonical_edges(
        torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]]), 8
    )
    masks = {}
    for name, indices in (("train", [0, 1, 2]), ("validation", [3, 4]), ("test", [5, 6])):
        mask = torch.zeros(8, dtype=torch.bool)
        mask[indices] = True
        masks[name] = mask
    return {
        "dataset": "cora",
        "classes": 2,
        "graphs": [
            {
                "x": torch.arange(24).float().reshape(8, 3),
                "y": torch.arange(8) % 2,
                "edge_index": arcs,
                "incidence_edge_index": incidence,
            }
        ],
        "splits": masks,
    }


def _mock_download(monkeypatch, tmp_path, payload):
    def download(name, root):
        source = root / "sources" / "fixture.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("test fixture; not a production dataset", encoding="utf-8")
        return copy.deepcopy(payload), [source]

    monkeypatch.setattr(benchmark_data, "_download_official", download)


def test_default_dataset_and_own_model_only_contract():
    args = benchmark.build_parser().parse_args([])
    assert args.datasets == ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
    assert not hasattr(args, "baselines")
    assert not hasattr(args, "heads")
    assert args.device == "cuda" and not args.amp
    assert args.workers is None
    benchmark.resolve_worker_arguments(args)
    assert args.workers == 4
    assert args.worker_configuration_source == "dataset_default"
    assert args.workers_by_dataset == {
        "cora": 0,
        "citeseer": 0,
        "pubmed": 0,
        "ppi": 4,
        "ogbn-arxiv": 0,
    }
    with pytest.raises(SystemExit):
        benchmark.build_parser().parse_args(["--tiny"])
    with pytest.raises(SystemExit):
        benchmark.build_parser().parse_args(["--baselines", "gat"])


def test_direct_benchmark_worker_resolution_is_dataset_specific():
    transductive = benchmark.build_parser().parse_args(["--datasets", "cora"])
    benchmark.resolve_worker_arguments(transductive)
    assert transductive.workers == 0
    assert transductive.workers_by_dataset == {"cora": 0}

    explicit = benchmark.build_parser().parse_args(
        ["--datasets", "cora", "ppi", "--workers", "0"]
    )
    benchmark.resolve_worker_arguments(explicit)
    assert explicit.workers == 0
    assert explicit.worker_configuration_source == "explicit_cli"
    assert explicit.workers_by_dataset == {"cora": 0, "ppi": 0}


def test_canonical_incidence_and_adjacency_have_same_edges():
    arcs, incidence = benchmark_data.canonical_edges(
        torch.tensor([[2, 1, 0, 1, 1, 0], [1, 2, 1, 0, 1, 1]]), 3
    )
    assert torch.equal(incidence, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(arcs, torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))


def test_split_validator_accepts_official_mask_semantics(payload):
    benchmark_data.validate_payload("cora", payload)
    assert sum(int(mask.sum()) for mask in payload["splits"].values()) == 7
    # Transductive public protocols deliberately leave some nodes unlabeled.


def test_split_validator_rejects_overlap(payload):
    payload["splits"]["validation"][0] = True
    payload["splits"]["validation"][3] = False
    with pytest.raises(ValueError, match="overlap"):
        benchmark_data.validate_payload("cora", payload)


def test_split_validator_rejects_wrong_official_size(payload):
    payload["splits"]["train"][0] = False
    with pytest.raises(ValueError, match="official protocol"):
        benchmark_data.validate_payload("cora", payload)


def test_same_graph_required_for_incidence_and_adjacency(payload):
    payload["graphs"][0]["incidence_edge_index"] = payload["graphs"][0]["incidence_edge_index"][
        :, :-1
    ]
    with pytest.raises(ValueError, match="different graphs"):
        benchmark_data.validate_payload("cora", payload)


def test_offline_missing_cache_never_calls_downloader(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline preparation must never call a downloader")

    monkeypatch.setattr(benchmark_data, "_download_official", forbidden)
    with pytest.raises(FileNotFoundError, match="No synthetic substitute"):
        benchmark_data.load_dataset("cora", tmp_path, allow_download=False)
    assert not list(tmp_path.iterdir())


def test_real_cache_contract_roundtrip_and_checksum(monkeypatch, tmp_path, payload):
    _mock_download(monkeypatch, tmp_path, payload)
    _, manifest = benchmark_data.load_dataset("cora", tmp_path, allow_download=True)
    assert len(manifest["data_sha256"]) == 64
    assert set(manifest["split_sha256"]) == {"train", "validation", "test"}
    loaded, reloaded_manifest = benchmark_data.load_dataset("cora", tmp_path, allow_download=False)
    assert torch.equal(loaded["graphs"][0]["x"], payload["graphs"][0]["x"])
    assert reloaded_manifest == manifest
    tensor_path = tmp_path / "conductance_gat/matched_benchmark_v1/cora/data.pt"
    with tensor_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CacheCorruptError, match="checksum"):
        benchmark_data.load_dataset("cora", tmp_path, allow_download=False)


def test_partial_cache_fails_even_when_download_allowed(tmp_path):
    folder = tmp_path / "conductance_gat/matched_benchmark_v1/cora"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CacheIncompleteError):
        benchmark_data.load_dataset("cora", tmp_path, allow_download=True)


def test_manifest_split_hash_corruption_fails(monkeypatch, tmp_path, payload):
    _mock_download(monkeypatch, tmp_path, payload)
    benchmark_data.load_dataset("cora", tmp_path, allow_download=True)
    path = tmp_path / "conductance_gat/matched_benchmark_v1/cora/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["split_sha256"]["train"] = "bad"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CacheCorruptError, match="split fingerprint"):
        benchmark_data.load_dataset("cora", tmp_path, allow_download=False)


def test_ppi_micro_f1_counts_node_labels_globally():
    logits = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    truth = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert benchmark.micro_f1(logits, truth) == pytest.approx(2 / 3)
    assert benchmark.micro_f1(torch.zeros(1, 2), torch.zeros(1, 2)) == 0


def test_incidence_operator_orientation_invariance_and_autograd():
    torch.manual_seed(4)
    model = benchmark.ConductanceConv(4)
    state = torch.randn(4, 4, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
    groups = torch.zeros(4, dtype=torch.long)
    output = model(state, edges, groups)
    assert torch.allclose(output, model(state, edges.flip(0), groups), atol=1e-6)
    assert torch.allclose(output.mean(0), state.mean(0), atol=1e-6)
    output.square().sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_conductance_classifier_forward_only(payload):
    graph = SimpleNamespace(**payload["graphs"][0])
    model = benchmark.ConductanceNodeClassifier(3, 2, hidden_channels=8, layers=2, dropout=0.0)
    assert model(graph).shape == (8, 2)


def test_cpu_training_is_rejected_before_any_dataset_action(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        benchmark.main(["--device", "cpu", "--output-dir", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_preparation_saves_protocol_without_training(monkeypatch, tmp_path, payload):
    _mock_download(monkeypatch, tmp_path, payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only must never train")

    monkeypatch.setattr(benchmark, "train_model", forbidden)
    output = tmp_path / "output"
    assert (
        benchmark.main(
            [
                "--prepare-only",
                "--allow-download",
                "--device",
                "cpu",
                "--datasets",
                "cora",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert result["status"] == "prepared"
    assert result["schema_version"] == 2
    assert result["datasets"]["cora"]["models"] == {}
    assert "baselines" not in result["datasets"]["cora"]
    assert not list(output.rglob("best.pt"))


def test_selection_rejects_duplicates_unknown_and_empty():
    assert benchmark._selection(["cora,citeseer", "pubmed"], benchmark_data.DATASETS) == [
        "cora",
        "citeseer",
        "pubmed",
    ]
    for values in (["cora", "cora"], ["toy"], []):
        with pytest.raises(ValueError):
            benchmark._selection(values, benchmark_data.DATASETS)


def test_optional_pyg_batch_offsets_and_conductance_forward(payload):
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch, Data

    graph = payload["graphs"][0]
    batch = Batch.from_data_list([Data(**graph), Data(**graph)])
    edge_count = graph["incidence_edge_index"].shape[1]
    assert torch.equal(
        batch.incidence_edge_index[:, edge_count:], graph["incidence_edge_index"] + 8
    )
    model = benchmark.ConductanceNodeClassifier(3, 2, hidden_channels=8, layers=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        result = model(batch)
    assert result.shape == (16, 2)
    assert torch.isfinite(result).all()
