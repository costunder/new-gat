"""Explicit CPU debug integration of V5 transfer and atomic interruption/resume.

Only CUDA/resource measurements and loading the small synthetic input graph are
stubbed. Model forward, validation, loss, backward, clipping, AdamW, checkpoint
publication, migration verification and resume are the real implementations.
These tests provide no GPU utilization, throughput or final-dataset evidence.
"""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from test_v5_transition_state import LEGACY_SOURCE, _assert_nested_equal

from research.conductance_gat.v5 import train, transition_initialization, transition_training
from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.transition import C_NAMESPACE


class DebugCrash(RuntimeError):
    pass


class DebugGraph:
    def __init__(self, **values):
        self.__dict__.update(values)

    def clone(self):
        return DebugGraph(
            **{
                name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                for name, value in vars(self).items()
            }
        )

    def to(self, device, **_kwargs):
        return DebugGraph(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


class CpuDebugMonitor:
    def __init__(self, device):
        assert device.type == "cpu"

    def start(self):
        return {"classification": "explicit_cpu_debug_hardware_stub"}

    def finish(self, **_kwargs):
        return {
            "classification": "explicit_cpu_debug_hardware_stub",
            "summary": {"classification": "CPU debug; GPU metrics unavailable"},
            "interval_series": {
                "gpu_sm_utilization_percent": {
                    "value": None,
                    "reason": "CPU debug test has no GPU measurements",
                }
            },
        }


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def _args(output, *, fixed=False):
    args = train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "fixed_c" if fixed else "shared_dynamic_c",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--hidden-channels",
            "8",
            "--layers",
            "2",
            "--heads",
            "2",
            "--ffn-multiplier",
            "2",
            "--epochs",
            "4",
            "--patience",
            "20",
            "--dropout",
            "0.2",
            "--no-activation-checkpoint",
            "--edge-chunk-size",
            "3",
            "--conductance-backend",
            "mlp" if fixed else "optimization",
            "--training-schedule",
            "staged" if fixed else "joint",
        ]
    )
    train.validate_args(args)
    return args


def _model(args):
    return GraphConditionedConductanceNodeClassifier(
        5,
        3,
        **train.architecture_configuration(args),
        conductance_mode=train.CONDITIONS[args.condition]["conductance_mode"],
        max_log_conductance=train.COMMON["max_log_conductance"],
        edge_chunk_size=args.edge_chunk_size,
    )


def _debug_data():
    generator = torch.Generator().manual_seed(108)
    graph = DebugGraph(
        x=torch.randn(9, 5, generator=generator),
        y=torch.arange(9) % 3,
        incidence_edge_index=torch.tensor(
            [[0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7], [1, 2, 3, 2, 3, 4, 4, 5, 6, 7, 7, 8]]
        ),
    )
    indices = {"train": torch.arange(6), "validation": torch.arange(6, 9)}
    payload = {
        "dataset": "cora",
        "classes": 3,
        "graphs": [vars(graph)],
        "classification": "synthetic_cpu_debug_fixture",
    }
    protocol = {
        "data_sha256": "a" * 64,
        "dataset": "cora",
        "classification": "synthetic_cpu_debug_fixture",
    }
    return graph, indices, payload, protocol


def _install_cpu_debug_environment(monkeypatch, graph, indices):
    monkeypatch.setattr(train, "_require_cuda", lambda device: None)
    monkeypatch.setattr(
        train,
        "validate_hardware_runtime",
        lambda args, device: {
            "classification": "explicit_cpu_debug_hardware_stub",
            "total_memory_bytes": 1,
        },
    )
    monkeypatch.setattr(train, "RuntimeResourceMonitor", CpuDebugMonitor)
    monkeypatch.setattr(train, "configure_compute", lambda args: None)
    monkeypatch.setattr(train, "_prepare_data", lambda *_args: (graph.clone(), indices, None))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_args: "CPU debug; no CUDA device")
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *_args: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *_args: None)
    # The production certificate validator is tested independently. No GPU
    # measurement is invented or used as a scientific result in this fixture.
    monkeypatch.setattr(
        transition_training,
        "_read_certificate",
        lambda *_args: (
            {"classification": "explicit_cpu_debug_certificate_stub"},
            {},
        ),
    )
    target_sources = copy.deepcopy(LEGACY_SOURCE)
    target_sources["research/conductance_gat/v5/model.py"] = "d" * 64
    target_sources["research/conductance_gat/v5/train.py"] = "e" * 64
    target_sources["research/conductance_gat/v5/optimization.py"] = "f" * 64
    monkeypatch.setattr(
        train, "implementation_source_hashes", lambda: copy.deepcopy(target_sources)
    )


def _source(tmp_path, graph, indices, protocol, *, fixed=False):
    directory = tmp_path / ("old-fixed" if fixed else "old-dynamic")
    directory.mkdir()
    args = _args(directory, fixed=fixed)
    args.conductance_backend, args.training_schedule = "mlp", "staged"
    train._seed(args.model_seed)
    model, optimizer = _model(args), None
    optimizer = train.make_optimizer(model)
    schedule = train.phase_schedule(4, list(args.phase_fractions), "staged")
    initial = train.state_sha256(model)
    counts = dict.fromkeys(("backbone", "spatial_w", "beta", "conductance"), 0)
    history = []
    first_gradient = None
    for epoch in (1, 2):
        phase, local = train.phase_at(schedule, epoch)
        phase_state = train.configure_phase(model, phase, local)
        optimizer.zero_grad(set_to_none=True)
        loss, count = train.training_loss(model(graph), graph, indices["train"])
        loss.backward()
        train.validate_active_gradient_connectivity(model, phase_state["active_parameter_groups"])
        if not fixed and epoch == 2:
            first_gradient = train.require_first_step_conductance_gradient(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.COMMON["gradient_clip_norm"])
        optimizer.step()
        counts = train.count_effective_group_step(
            counts, optimizer, phase_state["active_parameter_groups"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "train_label_count": count,
                "train_batches": 1,
                "maximum_preclip_gradient_norm": 1.0,
                "validation": 0.99,
                "classification": "historical_selection_sentinel_cpu_debug_only",
            }
        )
    identity = train.build_resume_identity(
        args,
        protocol,
        schedule,
        initial_state_sha256=initial,
        source_sha256=copy.deepcopy(LEGACY_SOURCE),
        runtime_versions=train._versions(),
    )
    for name in (
        "conductance_backend",
        "training_schedule",
        "solver_steps",
        "solver_step_size",
        "solver_entropy",
        "solver_degree_barrier",
    ):
        identity["configuration"].pop(name, None)
    identity_hash = train._canonical_sha256(identity)
    selected = {
        "model_state": copy.deepcopy(model.state_dict()),
        "resume_identity": identity,
        "resume_identity_sha256": identity_hash,
        "epoch": 2,
        "validation": 0.99,
        "selection_role": "primary",
        "phase": "conductance_calibration",
        "configuration": copy.deepcopy(identity["configuration"]),
        "schedule": schedule,
    }
    source_best = directory / "best.pt"
    torch.save(selected, source_best)
    saved = {
        "schema_version": 3,
        "complete": False,
        "epoch": 2,
        "history": history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "resume_identity": identity,
        "resume_identity_sha256": identity_hash,
        "best_epoch": 2,
        "best_metric": 0.99,
        "best_checkpoint_sha256": train.sha256_file(source_best),
        "global_best_epoch": 2,
        "global_best_metric": 0.99,
        "joint_best_epoch": 0,
        "joint_best_metric": -math.inf,
        "first_c_gradient": first_gradient,
        "optimizer_steps": 2,
        "effective_optimizer_steps_by_group": counts,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.get_rng_state(),
        "elapsed_seconds": 1.0,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
        "resume_source_compatibility": [],
    }
    source = directory / "last.pt"
    torch.save(saved, source)
    return source, saved


def _transition_args(source, output, *, fixed=False):
    args = _args(output, fixed=fixed)
    args.transition_from_checkpoint = source
    args.transition_source_sha256 = train.sha256_file(source)
    args.transition_mode = "continue_fixed" if fixed else "replace_c"
    args.transition_extra_epochs = 0
    args.transition_resource_certificate = source.parent / "not-a-real-gpu-certificate.json"
    args.transition_resource_sha256 = "b" * 64
    return args


def _run(args, payload, protocol):
    args.output_dir.mkdir(exist_ok=True)
    return train._train_model_impl(
        payload, protocol, args, torch.device("cpu"), args.output_dir, resource_state={}
    )


def _crash_after_save(monkeypatch, epoch):
    real = train._save

    def save_then_crash(path, payload):
        real(path, payload)
        if path.name == "last.pt" and payload.get("epoch") == epoch:
            raise DebugCrash(f"CPU debug interruption after atomic epoch {epoch}")

    monkeypatch.setattr(train, "_save", save_then_crash)
    return real


def test_real_training_reuses_shared_state_and_only_selects_new_c_epochs(tmp_path, monkeypatch):
    graph, indices, payload, protocol = _debug_data()
    _install_cpu_debug_environment(monkeypatch, graph, indices)
    source, saved = _source(tmp_path, graph, indices, protocol)
    source_bytes = source.read_bytes()
    args = _transition_args(source, tmp_path / "new-dynamic")
    result = _run(args, payload, protocol)
    last = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    assert last["schema_version"] == 4
    assert last["epoch"] == 4 and last["epoch_offset"] == 2
    assert [row["epoch"] for row in last["history"]] == [3, 4]
    assert last["optimizer_steps"] == 4
    assert last["best_epoch"] in (3, 4) and last["global_best_epoch"] in (3, 4)
    assert last["joint_best_epoch"] in (3, 4)
    assert result["transition_provenance"]["source_selection_state"]["best_metric"] == 0.99
    assert result["post_transition_epochs_completed"] == 2
    assert result["post_transition_optimizer_steps"] == 2
    assert result["comparison_design"]["fresh_paired_initialization"] is False
    old_steps = {
        name: float(saved["optimizer_state"]["state"][identifier]["step"])
        for group in saved["optimizer_state"]["param_groups"]
        for name, identifier in zip(group["parameter_names"], group["params"], strict=True)
    }
    for group in last["optimizer_state"]["param_groups"]:
        for name, identifier in zip(group["parameter_names"], group["params"], strict=True):
            expected = 2 if C_NAMESPACE.match(name) else old_steps[name] + 2
            assert float(last["optimizer_state"]["state"][identifier]["step"]) == expected
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("crash_epoch", [2, 3])
def test_boundary_and_epoch3_crash_resume_match_uninterrupted_cpu_exactly(
    tmp_path,
    monkeypatch,
    crash_epoch,
):
    graph, indices, payload, protocol = _debug_data()
    _install_cpu_debug_environment(monkeypatch, graph, indices)
    source, _ = _source(tmp_path, graph, indices, protocol)
    control_args = _transition_args(source, tmp_path / "control")
    _run(control_args, payload, protocol)
    expected = train.load_checkpoint_on_cpu(control_args.output_dir / "last.pt")
    resumed_args = _transition_args(source, tmp_path / "interrupted")
    real_save = _crash_after_save(monkeypatch, crash_epoch)
    with pytest.raises(DebugCrash, match=f"epoch {crash_epoch}"):
        _run(resumed_args, payload, protocol)
    boundary = train.load_checkpoint_on_cpu(resumed_args.output_dir / "last.pt")
    assert boundary["epoch"] == crash_epoch
    assert boundary["epoch_offset"] == 2
    if crash_epoch == 2:
        assert boundary["history"] == []
        assert boundary["best_epoch"] == 0
        assert boundary["optimizer_steps"] == 2
    else:
        assert [row["epoch"] for row in boundary["history"]] == [3]
        assert boundary["best_epoch"] == 3
    monkeypatch.setattr(train, "_save", real_save)
    _run(resumed_args, payload, protocol)
    actual = train.load_checkpoint_on_cpu(resumed_args.output_dir / "last.pt")
    for field in ("model_state", "optimizer_state", "cpu_rng_state", "cuda_rng_state"):
        _assert_nested_equal(actual[field], expected[field])
    assert [row["epoch"] for row in actual["history"]] == [3, 4]
    assert actual["optimizer_steps"] == expected["optimizer_steps"] == 4


def test_fixed_boundary_copies_best_with_new_identity_without_changing_source(
    tmp_path,
    monkeypatch,
):
    graph, indices, payload, protocol = _debug_data()
    _install_cpu_debug_environment(monkeypatch, graph, indices)
    source, saved = _source(tmp_path, graph, indices, protocol, fixed=True)
    source_bytes = source.read_bytes()
    old_best = (source.parent / "best.pt").read_bytes()
    args = _transition_args(source, tmp_path / "fixed-continuation", fixed=True)
    _crash_after_save(monkeypatch, 2)
    with pytest.raises(DebugCrash):
        _run(args, payload, protocol)
    boundary = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    selected = train.load_checkpoint_on_cpu(args.output_dir / "best.pt")
    _assert_nested_equal(boundary["model_state"], saved["model_state"])
    _assert_nested_equal(boundary["optimizer_state"], saved["optimizer_state"])
    assert boundary["history"] == saved["history"]
    assert boundary["epoch_offset"] == 0 and boundary["best_metric"] == 0.99
    assert selected["resume_identity"] == boundary["resume_identity"]
    assert selected["resume_identity"] != saved["resume_identity"]
    assert selected["source_selected_checkpoint_sha256"] == saved["best_checkpoint_sha256"]
    assert train.sha256_file(args.output_dir / "best.pt") == boundary["best_checkpoint_sha256"]
    assert source.read_bytes() == source_bytes
    assert (source.parent / "best.pt").read_bytes() == old_best


def test_completed_fixed_c_is_rejected_by_training_transition(tmp_path, monkeypatch):
    graph, indices, payload, protocol = _debug_data()
    _install_cpu_debug_environment(monkeypatch, graph, indices)
    source, saved = _source(tmp_path, graph, indices, protocol, fixed=True)
    saved["complete"] = True
    torch.save(saved, source)
    args = _transition_args(source, tmp_path / "must-not-retrain", fixed=True)
    with pytest.raises(ValueError, match="historical reference"):
        _run(args, payload, protocol)
    assert not (args.output_dir / "last.pt").exists()


def test_transition_request_rejects_no_resume(tmp_path):
    source = tmp_path / "source-last.pt"
    source.write_bytes(b"explicit CPU debug placeholder; validation does not load this file")
    args = _transition_args(source, tmp_path / "new-output")
    args.resume = False
    with pytest.raises(ValueError, match="resume"):
        transition_training.validate_arguments(args)


def _main_argv(args):
    """Serialize the explicit CPU debug profile; production defaults are untouched."""
    fields = (
        "dataset",
        "condition",
        "output_dir",
        "device",
        "hidden_channels",
        "layers",
        "heads",
        "ffn_multiplier",
        "epochs",
        "patience",
        "dropout",
        "edge_chunk_size",
        "conductance_backend",
        "training_schedule",
        "transition_from_checkpoint",
        "transition_source_sha256",
        "transition_mode",
        "transition_extra_epochs",
        "transition_resource_certificate",
        "transition_resource_sha256",
    )
    argv = [
        part for key in fields for part in ("--" + key.replace("_", "-"), str(getattr(args, key)))
    ]
    return argv + ["--no-activation-checkpoint"]


def _main_fixture(tmp_path, monkeypatch, *, fixed=False):
    graph, indices, payload, protocol = _debug_data()
    _install_cpu_debug_environment(monkeypatch, graph, indices)
    monkeypatch.setattr(train, "load_dataset", lambda *_args, **_kwargs: (payload, protocol))
    source, saved = _source(tmp_path, graph, indices, protocol, fixed=fixed)
    return source, saved


def _inject_initialization_crash(monkeypatch, point):
    if point in {"marker", "metrics", "source_history"}:
        owner, filename = {
            "marker": (transition_initialization, transition_initialization.MARKER_FILENAME),
            "metrics": (train, "metrics.json"),
            "source_history": (transition_training, "source-history.json"),
        }[point]
        real = owner.atomic_write_json

        def write_then_crash(path, payload):
            real(path, payload)
            if path.name == filename:
                raise DebugCrash(f"explicit CPU debug crash after {point}")

        monkeypatch.setattr(owner, "atomic_write_json", write_then_crash)
    else:
        real = train._save

        def checkpoint_crash(path, payload):
            if point == "before_last" and path.name == "last.pt":
                raise DebugCrash("explicit CPU debug crash before_last")
            real(path, payload)
            if point == "fixed_best" and path.name == "best.pt":
                raise DebugCrash("explicit CPU debug crash after fixed_best")

        monkeypatch.setattr(train, "_save", checkpoint_crash)


@pytest.mark.parametrize(
    ("point", "fixed"),
    [
        ("marker", False),
        ("metrics", False),
        ("source_history", False),
        ("fixed_best", True),
        ("before_last", True),
    ],
)
def test_main_retries_each_owned_initialization_publication_exactly(
    tmp_path,
    monkeypatch,
    point,
    fixed,
):
    source, _ = _main_fixture(tmp_path, monkeypatch, fixed=fixed)
    source_bytes = source.read_bytes()
    source_best_bytes = (source.parent / "best.pt").read_bytes()
    control = _transition_args(source, tmp_path / "control-main", fixed=fixed)
    assert train.main(_main_argv(control)) == 0
    expected = train.load_checkpoint_on_cpu(control.output_dir / "last.pt")
    args = _transition_args(source, tmp_path / "retry-main", fixed=fixed)
    with monkeypatch.context() as interruption:
        _inject_initialization_crash(interruption, point)
        with pytest.raises(DebugCrash, match=point):
            train.main(_main_argv(args))
    assert not (args.output_dir / "last.pt").exists()
    marker = args.output_dir / transition_initialization.MARKER_FILENAME
    marker_bytes = marker.read_bytes()
    retained = {
        name: (args.output_dir / name).read_bytes()
        for name in ("best.pt", "source-history.json")
        if (args.output_dir / name).exists()
    }
    assert train.main(_main_argv(args)) == 0
    actual = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    for field in ("model_state", "optimizer_state", "cpu_rng_state", "cuda_rng_state"):
        _assert_nested_equal(actual[field], expected[field])
    assert actual["complete"] and actual["epoch"] == 4
    assert actual["epoch_offset"] == (0 if fixed else 2)
    assert actual["optimizer_steps"] == expected["optimizer_steps"] == 4
    assert marker.read_bytes() == marker_bytes
    for name, original in retained.items():
        assert (args.output_dir / name).read_bytes() == original
    if fixed:
        assert not (args.output_dir / "best.previous.pt").exists()
    assert source.read_bytes() == source_bytes
    assert (source.parent / "best.pt").read_bytes() == source_best_bytes


def _ensure_marker(args):
    return transition_initialization.ensure_transition_initialization(
        args,
        args.output_dir,
        configuration=train.configuration(args),
        source_sha256=train.implementation_source_hashes(),
        runtime_versions=train._versions(),
    )


def test_initialization_never_adopts_unmarked_nonempty_output(tmp_path, monkeypatch):
    source, _ = _main_fixture(tmp_path, monkeypatch)
    args = _transition_args(source, tmp_path / "unowned")
    args.output_dir.mkdir()
    unrelated = args.output_dir / "user-results.txt"
    unrelated.write_text("existing results must remain unchanged", encoding="utf-8")
    before = unrelated.read_bytes()
    with pytest.raises(FileExistsError, match="no matching initialization marker"):
        train.main(_main_argv(args))
    assert unrelated.read_bytes() == before
    assert list(args.output_dir.iterdir()) == [unrelated]


@pytest.mark.parametrize("field", ["dataset", "condition", "epochs", "transition_resource_sha256"])
def test_marker_only_retry_binds_request_dataset_condition_and_configuration(
    tmp_path,
    monkeypatch,
    field,
):
    source, _ = _main_fixture(tmp_path, monkeypatch)
    args = _transition_args(source, tmp_path / "claimed")
    assert _ensure_marker(args)
    marker = args.output_dir / transition_initialization.MARKER_FILENAME
    original = marker.read_bytes()
    changed = copy.deepcopy(args)
    setattr(
        changed,
        field,
        {
            "dataset": "citeseer",
            "condition": "fixed_c",
            "epochs": 5,
            "transition_resource_sha256": "c" * 64,
        }[field],
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _ensure_marker(changed)
    assert marker.read_bytes() == original


@pytest.mark.parametrize("artifact", ["history.json", "best.previous.pt", "unrelated.json"])
def test_marker_does_not_authorize_trained_or_unrelated_artifacts(
    tmp_path,
    monkeypatch,
    artifact,
):
    source, _ = _main_fixture(tmp_path, monkeypatch)
    args = _transition_args(source, tmp_path / "claimed")
    assert _ensure_marker(args)
    other = args.output_dir / artifact
    other.write_text("unowned debug artifact", encoding="utf-8")
    before = other.read_bytes()
    with pytest.raises(FileExistsError, match="unrelated or trained artifact"):
        _ensure_marker(args)
    assert other.read_bytes() == before


def test_marker_rejects_completed_metrics_without_last_checkpoint(tmp_path, monkeypatch):
    source, _ = _main_fixture(tmp_path, monkeypatch)
    args = _transition_args(source, tmp_path / "claimed")
    assert _ensure_marker(args)
    metrics = args.output_dir / "metrics.json"
    train.atomic_write_json(
        metrics,
        {
            "schema_version": 1,
            "research_suite": train.SUITE,
            "dataset": args.dataset,
            "condition": args.condition,
            "configuration": train.configuration(args),
            "test_evaluated": False,
            "status": "passed",
        },
    )
    before = metrics.read_bytes()
    with pytest.raises(ValueError, match="completed, trained or mismatched"):
        _ensure_marker(args)
    assert metrics.read_bytes() == before


@pytest.mark.parametrize("artifact", ["source-history.json", "best.pt"])
def test_main_refuses_tampered_partial_publications_without_overwriting(
    tmp_path,
    monkeypatch,
    artifact,
):
    source, _ = _main_fixture(tmp_path, monkeypatch, fixed=True)
    source_bytes = source.read_bytes()
    args = _transition_args(source, tmp_path / "partial-tamper", fixed=True)
    with monkeypatch.context() as interruption:
        _inject_initialization_crash(interruption, "before_last")
        with pytest.raises(DebugCrash, match="before_last"):
            train.main(_main_argv(args))
    path = args.output_dir / artifact
    if artifact == "source-history.json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows[0]["train_loss"] += 1.0
        train.atomic_write_json(path, rows)
    else:
        selected = train.load_checkpoint_on_cpu(path)
        next(iter(selected["model_state"].values())).add_(1.0)
        torch.save(selected, path)
    tampered_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="source history differs|best checkpoint does not match"):
        train.main(_main_argv(args))
    assert path.read_bytes() == tampered_bytes
    assert source.read_bytes() == source_bytes
    assert not (args.output_dir / "last.pt").exists()
    assert not (args.output_dir / "best.previous.pt").exists()
