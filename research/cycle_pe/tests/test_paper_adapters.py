from __future__ import annotations

import builtins
import io
import zipfile

import networkx as nx
import numpy as np
import pytest
import torch

from research.cycle_pe import paper_adapters
from research.cycle_pe.paper_adapters import (
    BREC_OFFICIAL_RECORD_COUNT,
    BRECAdapter,
    download_brec_v3,
    find_brec_v3,
    load_brec_v3,
    validate_brec_v3,
    write_tiny_brec_fixture,
)


def test_tiny_brec_fixture_matches_lazy_rpc_layout(tmp_path) -> None:
    path = write_tiny_brec_fixture(
        tmp_path / "BREC" / "Data" / "raw" / "brec_v3.npy", num_relabel=2
    )
    assert find_brec_v3(tmp_path) == path
    adapter = BRECAdapter(path, num_relabel=2)
    assert adapter.pair_count == 2
    pair = adapter.load_pair(0)
    assert len(pair.train_test) == 4
    assert len(pair.reliability) == 4
    left = nx.Graph(pair.train_test[0].edges)
    right = nx.Graph(pair.train_test[1].edges)
    assert not nx.is_isomorphic(left, right)
    assert nx.is_isomorphic(
        nx.Graph(pair.reliability[0].edges), nx.Graph(pair.reliability[1].edges)
    )
    assert adapter.metadata["sha256"]
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_official_brec_validation_enforces_400_pair_record_layout(tmp_path) -> None:
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.asarray([b"A_"] * 128, dtype=object), allow_pickle=True)
    with pytest.raises(RuntimeError, match="51,200 records"):
        validate_brec_v3(wrong, protocol="official")

    official_shape = tmp_path / "official-shape.npy"
    np.save(
        official_shape,
        np.asarray([b"A_"] * BREC_OFFICIAL_RECORD_COUNT, dtype=object),
        allow_pickle=True,
    )
    metadata = validate_brec_v3(official_shape, protocol="official")
    assert metadata["records"] == 51_200
    assert metadata["pair_count"] == 400
    assert metadata["official_shape_validated"] is True
    assert metadata["official_source_hash_pinned"] is False


def test_missing_brec_artifact_error_is_actionable(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="GraphPKU/BREC"):
        find_brec_v3(tmp_path)


def test_brec_full_load_is_fail_closed_without_opt_in(monkeypatch, tmp_path) -> None:
    def unexpected_network(*args, **kwargs):
        raise AssertionError("network must not be touched without --allow-download")

    monkeypatch.setattr(paper_adapters.urllib.request, "urlopen", unexpected_network)
    with pytest.raises(FileNotFoundError, match="--allow-download"):
        load_brec_v3(tmp_path, allow_download=False)


def test_brec_tiny_fixture_is_always_offline(monkeypatch, tmp_path) -> None:
    def unexpected_network(*args, **kwargs):
        raise AssertionError("tiny BREC fixture must never use the network")

    monkeypatch.setattr(paper_adapters.urllib.request, "urlopen", unexpected_network)
    adapter = load_brec_v3(
        tmp_path,
        num_relabel=2,
        allow_download=True,
        tiny=True,
    )
    assert adapter.pair_count == 2
    assert adapter.metadata["offline_fixture"] is True


class _FakeHTTPResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def geturl(self) -> str:
        return paper_adapters.BREC_RAW_URL

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _zip_payload(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def test_explicit_brec_download_extracts_only_valid_npy(monkeypatch, tmp_path) -> None:
    fixture = write_tiny_brec_fixture(tmp_path / "source.npy", num_relabel=2)
    payload = _zip_payload({"BREC/Data/raw/brec_v3.npy": fixture.read_bytes()})
    monkeypatch.setattr(
        paper_adapters.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    target = download_brec_v3(tmp_path / "data")
    assert target == (tmp_path / "data" / "BREC" / "Data" / "raw" / "brec_v3.npy")
    assert target.read_bytes().startswith(b"\x93NUMPY")
    metadata = target.with_name("brec_v3.download.json").read_text(encoding="utf-8")
    assert "archive_sha256" in metadata and "brec_v3_sha256" in metadata
    records = np.load(target, allow_pickle=True)
    assert records.shape == (16,)


def test_brec_download_rejects_archive_path_traversal(monkeypatch, tmp_path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray([b"A_"], dtype=object), allow_pickle=True)
    payload = _zip_payload(
        {
            "BREC/Data/raw/brec_v3.npy": buffer.getvalue(),
            "../escaped.txt": b"unsafe",
        }
    )
    monkeypatch.setattr(
        paper_adapters.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )
    data_root = tmp_path / "data"
    with pytest.raises(RuntimeError, match="unsafe path"):
        download_brec_v3(data_root)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (data_root / "BREC" / "Data" / "raw" / "brec_v3.npy").exists()


def test_missing_pyg_error_has_cuda_install_guidance(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("torch_geometric"):
            raise ImportError("fixture blocks optional PyG")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="CUDA runtime") as error:
        paper_adapters._require_pyg_zinc()
    assert "torch-geometric" in str(error.value)
    assert "installation.html" in str(error.value)


def test_zinc_download_requires_explicit_opt_in(monkeypatch, tmp_path) -> None:
    class UnexpectedZincConstruction:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("adapter must reject before PyG starts a download")

    monkeypatch.setattr(paper_adapters, "_require_pyg_zinc", lambda: UnexpectedZincConstruction)
    with pytest.raises(FileNotFoundError, match="--allow-download"):
        paper_adapters.load_zinc12k(tmp_path, tiny=True, allow_download=False)


def test_zinc_adapter_uses_official_split_names_without_network(monkeypatch, tmp_path) -> None:
    processed = tmp_path / "ZINC12K" / "subset" / "processed"
    processed.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (processed / f"{split}.pt").touch()

    class FakeData:
        num_nodes = 3
        x = torch.tensor([[0], [2], [4]])
        edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
        edge_attr = torch.tensor([1, 1, 2, 2, 3, 3])
        y = torch.tensor([0.75])

    calls: list[tuple[bool, str]] = []

    class FakeZinc:
        def __init__(self, *, root, subset, split) -> None:
            calls.append((subset, split))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index):
            assert index == 0
            return FakeData()

    monkeypatch.setattr(paper_adapters, "_require_pyg_zinc", lambda: FakeZinc)
    bundle = paper_adapters.load_zinc12k(tmp_path, tiny=True, allow_download=False)
    assert calls == [(True, "train"), (True, "val"), (True, "test")]
    assert {name: len(graphs) for name, graphs in bundle.splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    graph = bundle.splits["train"][0]
    assert graph.edges == ((0, 1), (0, 2), (1, 2))
    assert graph.node_features is not None and graph.node_features.shape == (3, 28)
    assert graph.edge_features is not None and graph.edge_features.shape == (3, 4)
    assert graph.graph_targets is not None and graph.graph_targets.tolist() == [0.75]
    assert set(bundle.metadata["cache_sha256"]) == {
        "subset/processed/train.pt",
        "subset/processed/val.pt",
        "subset/processed/test.pt",
    }
