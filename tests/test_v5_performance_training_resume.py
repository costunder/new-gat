"""Archived-source contracts with synthetic CPU state-handoff integration.

Both source identities use actual 51da819 -> 8da06ca Git blobs. Current CPU
training control flow is exercised with explicitly injected historical source
maps; it is NOT identified as archived code, an authorized live-source resume,
or reproduction of past GPU kernels/results. Small tensors exercise state
handoff only. The unchanged production registry/helper are never mocked.
Separate tests require the actual current source inventory to be rejected
without changing checkpoint/history files or making an optimizer update.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import subprocess
from pathlib import Path

import pytest
import torch
from test_v5_training_repair import _repair_fixture
from test_v5_transition_state import _assert_nested_equal
from test_v5_transition_training import (
    DebugCrash,
    _args,
    _debug_data,
    _install_cpu_debug_environment,
    _run,
)

from chartgat import resume_compat as compat
from research.conductance_gat.v5 import train

ROOT = Path(__file__).resolve().parents[1]
REVISION = "51da8191274e8e8ff89b6878964a6e5af091c696"
LIVE_SOURCE_FUNCTION = train.implementation_source_hashes


def _git(*arguments, input_bytes=None):
    # Invocation-only exception for the exact shared test checkout; no global
    # configuration changes, git writes, revision checkout, or index mutation.
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments],
        cwd=ROOT,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def _revision_sources(revision):
    """Read a revision's own V5 source inventory and exact Git-blob digests."""
    archived_train = _git("show", f"{revision}:research/conductance_gat/v5/train.py")
    assignment = next(
        item
        for item in ast.parse(archived_train).body
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_SHARED_IMPLEMENTATION_SOURCES"
            for target in item.targets
        )
    )
    shared = []
    for item in assignment.value.elts:
        if isinstance(item, ast.Constant):
            shared.append(item.value)
        else:
            assert isinstance(item, ast.Starred)
            assert (
                isinstance(item.value, ast.Name) and item.value.id == "COMPATIBILITY_SOURCE_FILES"
            )
            shared.extend(compat.COMPATIBILITY_SOURCE_FILES)
    archived_files = _git("ls-tree", "-r", "--name-only", revision).decode("utf-8").splitlines()
    paths = sorted(
        set(shared)
        | {
            name
            for name in archived_files
            if name.startswith("research/conductance_gat/v5/")
            and name.count("/") == 3
            and name.endswith(".py")
        }
    )
    request = "".join(f"{revision}:{name}\n" for name in paths).encode()
    blobs = io.BytesIO(_git("cat-file", "--batch", input_bytes=request))
    result = {}
    for name in paths:
        header = blobs.readline().decode().strip().split()
        assert len(header) == 3 and header[1] == "blob"
        raw = blobs.read(int(header[2]))
        assert blobs.read(1) == b"\n"
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def _server_live_sources():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in LIVE_SOURCE_FUNCTION()
    }


@pytest.fixture(scope="module")
def source_maps():
    previous = _revision_sources(REVISION)
    historical_target = _revision_sources("8da06cacec59515d84c08d892315e4c8ecfd5b5b")
    # The real production helper/registry must still match the registered
    # historical target. Their authorization is not extended to today's code.
    for name in compat.COMPATIBILITY_SOURCE_FILES:
        assert historical_target[name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    evidence = compat.require_source_compatibility(previous, historical_target)
    assert evidence["base_commit"] == REVISION
    assert evidence["patch_id"] == compat.PERFORMANCE_PATCH_ID
    assert evidence["bitwise_numerical_identity"] is False
    return previous, historical_target


def _fixture(directory, monkeypatch, sources, *, fixed=False):
    graph, indices, payload, protocol = _debug_data()
    _install_cpu_debug_environment(monkeypatch, graph, indices)
    # Replace the older generic fixture's invented source identifiers with
    # actual archived/server-LF source inputs. Compatibility itself stays real.
    monkeypatch.setattr(train, "implementation_source_hashes", lambda: dict(sources))
    return _args(directory, fixed=fixed), payload, protocol


def _boundary(directory, monkeypatch, previous, *, fixed=False, ppi=False):
    if ppi:
        args, payload, protocol = _repair_fixture(directory, monkeypatch)
        monkeypatch.setattr(train, "implementation_source_hashes", lambda: dict(previous))
    else:
        args, payload, protocol = _fixture(directory, monkeypatch, previous, fixed=fixed)
    save = train._save

    def stop_at_epoch_two(path, value):
        save(path, value)
        if path.name == "last.pt" and value["epoch"] == 2:
            raise DebugCrash("synthetic CPU checkpoint boundary, not a server interruption")

    with monkeypatch.context() as crash:
        crash.setattr(train, "_save", stop_at_epoch_two)
        with pytest.raises(DebugCrash, match="checkpoint boundary"):
            _run(args, payload, protocol)
    saved = train.load_checkpoint_on_cpu(directory / "last.pt")
    assert saved["epoch"] == 2
    assert saved["optimizer_steps"] == sum(row["train_batches"] for row in saved["history"])
    assert saved["resume_identity"]["source_sha256"] == previous
    assert not saved["complete"]
    return args, payload, protocol, saved


@pytest.mark.parametrize("fixed", [False, True])
def test_actual_51_source_resume_retains_state_at_first_batch_and_advances(
    tmp_path,
    monkeypatch,
    source_maps,
    fixed,
):
    previous, current = source_maps
    control_args, payload, protocol = _fixture(
        tmp_path / "current-control",
        monkeypatch,
        current,
        fixed=fixed,
    )
    _run(control_args, payload, protocol)
    control = train.load_checkpoint_on_cpu(control_args.output_dir / "last.pt")
    args, payload, protocol, boundary = _boundary(
        tmp_path / "historical-identity-debug",
        monkeypatch,
        previous,
        fixed=fixed,
    )
    historical_history = copy.deepcopy(boundary["history"])
    monkeypatch.setattr(train, "implementation_source_hashes", lambda: dict(current))
    make_optimizer, batches = train.make_optimizer, train._training_batches
    captured, visited = {}, []

    def capture_optimizer(model):
        optimizer = make_optimizer(model)
        captured.update(model=model, optimizer=optimizer)
        return optimizer

    def inspect_first_batch(data, indices, sampler, epoch, device, seed, run_args, **kwargs):
        if not visited:
            assert epoch == boundary["epoch"] + 1 == 3
            _assert_nested_equal(captured["model"].state_dict(), boundary["model_state"])
            _assert_nested_equal(captured["optimizer"].state_dict(), boundary["optimizer_state"])
            assert torch.equal(torch.get_rng_state(), boundary["cpu_rng_state"])
        visited.append(epoch)
        yield from batches(data, indices, sampler, epoch, device, seed, run_args, **kwargs)

    monkeypatch.setattr(train, "make_optimizer", capture_optimizer)
    monkeypatch.setattr(train, "_training_batches", inspect_first_batch)
    result = _run(args, payload, protocol)
    resumed = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    assert visited == [3, 4]
    assert resumed["epoch"] == resumed["optimizer_steps"] == 4
    assert resumed["history"][:2] == historical_history
    assert resumed["resume_identity"]["source_sha256"] == current
    provenance = resumed["resume_source_compatibility"]
    assert len(provenance) == 1
    assert provenance[0]["patch_id"] == compat.PERFORMANCE_PATCH_ID
    assert provenance[0]["previous_source_sha256"] == previous
    assert provenance[0]["current_source_sha256"] == current
    assert provenance[0]["bitwise_numerical_identity"] is False
    for name in ("model_state", "optimizer_state", "cpu_rng_state", "cuda_rng_state"):
        # Both control and synthetic source tensor generation use current CPU
        # kernels. This checks resume/RNG handling, not old/new CUDA numerics.
        _assert_nested_equal(resumed[name], control[name])
    assert result["optimizer_steps"] == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("beta_initial", 0.5),
        ("solver_steps", 9),
        ("training_schedule", "staged"),
        ("batch_size", 2),
        ("epochs", 5),
    ],
)
def test_real_registry_never_waives_recipe_changes_or_overwrites_checkpoint(
    tmp_path,
    monkeypatch,
    source_maps,
    field,
    value,
):
    previous, current = source_maps
    args, payload, protocol, _ = _boundary(
        tmp_path / "preserved", monkeypatch, previous, ppi=field == "batch_size"
    )
    original = {
        path.name: path.read_bytes() for path in args.output_dir.iterdir() if path.is_file()
    }
    monkeypatch.setattr(train, "implementation_source_hashes", lambda: dict(current))
    setattr(args, field, value)

    def forbidden_update(*_args, **_kwargs):
        raise AssertionError("a rejected recipe must not perform an optimizer update")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(args, payload, protocol)
    after = {path.name: path.read_bytes() for path in args.output_dir.iterdir() if path.is_file()}
    assert after == original


@pytest.mark.parametrize("origin", ["before_performance_repair", "historical_target"])
def test_actual_current_sources_cannot_inherit_historical_resume_authority(
    tmp_path, monkeypatch, source_maps, origin
):
    previous, historical_target = source_maps
    source = previous if origin == "before_performance_repair" else historical_target
    args, payload, protocol, _ = _boundary(
        tmp_path / "preserved-live-rejection", monkeypatch, source
    )
    original = {
        path.name: path.read_bytes() for path in args.output_dir.iterdir() if path.is_file()
    }
    live = _server_live_sources()
    assert live != historical_target
    assert not compat.snapshots_match(source, live)
    with pytest.raises(ValueError):
        compat.require_source_compatibility(source, live)
    monkeypatch.setattr(train, "implementation_source_hashes", lambda: dict(live))

    def forbidden_update(*_args, **_kwargs):
        raise AssertionError("unregistered live sources must not perform an optimizer update")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(args, payload, protocol)
    after = {path.name: path.read_bytes() for path in args.output_dir.iterdir() if path.is_file()}
    assert after == original
