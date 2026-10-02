"""Recovery control checks and explicit synthetic CUDA smoke, not benchmark results."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import recover_output_init_study as recovery


@pytest.fixture
def source_run(tmp_path):
    repo, run = tmp_path / "repo", tmp_path / "original"
    repo.mkdir()
    run.mkdir()
    files = {
        "experiments/output_init_ablation/train.py": b"# original training\n",
        "research/conductance_gat/v5/train.py": b"# original data\n",
    }
    with zipfile.ZipFile(run / "execution_sources.zip", "w") as archive:
        for name, content in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            archive.writestr(name, content)
    recovery.write(
        run / "study_contract.json",
        {"source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}},
    )
    return repo, run


def test_independent_wedge_additions_are_reported_without_changing_original(source_run):
    repo, run = source_run
    before = {p.name: recovery.digest(p) for p in run.iterdir()}
    path = repo / "research/wedge_propagation/learned/model.py"
    path.parent.mkdir(parents=True)
    path.write_text("# separate study\n")
    report = recovery.source_audit(run, repo)
    assert not report["blocked"]
    assert [row["path"] for row in report["changes"]] == [path.relative_to(repo).as_posix()]
    assert before == {p.name: recovery.digest(p) for p in run.iterdir()}


@pytest.mark.parametrize("change", ["edit", "delete", "new_model"])
def test_scientific_changes_block_recovery(source_run, change):
    repo, run = source_run
    path = repo / "experiments/output_init_ablation/train.py"
    if change == "delete":
        path.unlink()  # A disposable test fixture only.
    elif change == "edit":
        path.write_text("# different training\n")
    else:
        path = path.with_name("new_model.py")
        path.write_text("# new scientific source\n")
    report = recovery.source_audit(run, repo)
    assert report["blocked"] == [path.relative_to(repo).as_posix()]
    with pytest.raises(ValueError, match="unverified source"):
        recovery.launch(run, run.parent / "recovered", report)
    assert not (run.parent / "recovered").exists()


def test_archive_tampering_is_not_accepted(source_run):
    repo, run = source_run
    manifest = recovery.read(run / "study_contract.json")
    with zipfile.ZipFile(run / "execution_sources.zip", "w") as archive:
        for name in manifest["source_sha256"]:
            archive.writestr(name, "wrong bytes")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        recovery.source_audit(run, repo)


@pytest.mark.parametrize("name", ["../model.py", "/model.py", "C:/model.py", "a\\b.py"])
def test_archive_paths_cannot_escape(name):
    with pytest.raises(ValueError, match="unsafe archive"):
        recovery.safe_name(name)


@pytest.fixture
def completed(tmp_path):
    # Fabricated JSON/checkpoint bytes exercise file integrity only, never a model claim.
    folder = tmp_path / "file-integrity-fixture"
    folder.mkdir()
    checkpoint = folder / "learned-best-0001.pt"
    checkpoint.write_bytes(b"explicit file-integrity fixture; not a torch checkpoint")
    history = [
        {
            "debug": True,
            "epoch": i,
            "supervised_nodes": 12,
            "optimization_steps": 3,
            "validation": {"correct": 10 if i == 1 else 9},
        }
        for i in (1, 2)
    ]
    record = {
        "debug": True,
        "initial_common_sha256": "fixture",
        "history": history,
        "best": {
            "path": str(checkpoint),
            "file_sha256": recovery.digest(checkpoint),
            "epoch": 1,
            "validation": history[0]["validation"],
        },
    }
    recovery.write(folder / "learned-trained.json", record)
    recovery.write(
        folder / "learned-initial.json",
        {
            "debug": True,
            "epochs": 2,
            "train_coverage": 1.0,
            "common_sha256": "fixture",
            "train_nodes_used_per_epoch": 12,
            "optimization_steps": 6,
        },
    )
    for row in history:
        recovery.write(folder / f"learned-epoch-{row['epoch']:04d}.json", row)
        recovery.write(
            folder / f"learned-inspection-{row['epoch']:04d}.json",
            {"debug": True, "epoch": row["epoch"]},
        )
    return folder, record


def test_completed_evidence_copy_preserves_original(completed, tmp_path):
    folder, record = completed
    original = {p.name: recovery.digest(p) for p in folder.iterdir()}
    assert recovery.completed_condition(folder, "learned", 2, synthetic_debug=True) == record
    target = tmp_path / "copy"
    target.mkdir()
    copied = recovery.copy_condition(folder, target, "learned", record)
    assert Path(copied["best"]["path"]).parent == target
    assert recovery.completed_condition(target, "learned", 2, synthetic_debug=True) == copied
    assert original == {p.name: recovery.digest(p) for p in folder.iterdir()}
    with pytest.raises(FileExistsError):
        recovery.copy_condition(folder, target, "learned", record)


def test_production_rejects_debug_evidence(completed):
    folder, _ = completed
    with pytest.raises(ValueError, match="debug"):
        recovery.completed_condition(folder, "learned", 2)


def test_partial_condition_is_not_resumed_from_best_weights(completed):
    folder, _ = completed
    (folder / "learned-trained.json").unlink()
    with pytest.raises(ValueError, match="optimizer/RNG"):
        recovery.completed_condition(folder, "learned", 2, synthetic_debug=True)
    assert recovery.completed_condition(folder, "fixed", 2) is None


@pytest.mark.parametrize(
    "defect", ["checkpoint", "missing_epoch", "changed_epoch", "steps", "best"]
)
def test_invalid_completed_artifacts_rejected(completed, defect):
    folder, record = completed
    if defect == "checkpoint":
        Path(record["best"]["path"]).write_bytes(b"corrupt")
    elif defect == "missing_epoch":
        record["history"].pop()
    elif defect == "changed_epoch":
        (folder / "learned-epoch-0001.json").write_text("{}")
    elif defect == "steps":
        initial = recovery.read(folder / "learned-initial.json")
        initial["optimization_steps"] = 7
        (folder / "learned-initial.json").write_text(json.dumps(initial))
    else:
        record["best"]["epoch"] = 2
    (folder / "learned-trained.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        recovery.completed_condition(folder, "learned", 2, synthetic_debug=True)


def test_reused_pairing_checked_before_further_training(tmp_path):
    selected = {}
    for initialization in recovery.INITIALIZATIONS:
        folder = tmp_path / f"train-{initialization}"
        folder.mkdir()
        recovery.write(folder / "learned-initial.json", {"initial_non_output_sha256": "same"})
        selected[initialization] = {"learned": {"history": [{"sampling_evidence": "same"}]}}
    recovery.verify_pairing(selected, tmp_path)
    selected["kaiming_relu"]["learned"]["history"][0]["sampling_evidence"] = "different"
    with pytest.raises(ValueError, match="sampled-context pairing"):
        recovery.verify_pairing(selected, tmp_path)


def test_actual_archived_imports_reproduce_original_source_inventory(tmp_path):
    # Actual repository source/archive IO, no data download or training.
    from experiments.output_init_ablation.study import command
    from experiments.output_init_ablation.train import parser, research_contract, validate

    repository = recovery.ROOT
    args = parser().parse_args(command("baseline", "calibrate", tmp_path / "unused", tmp_path)[4:])
    validate(args)
    hashes = research_contract(args, {})["source_sha256"]
    original = {
        name: sha
        for name, sha in hashes.items()
        if not name.startswith("research/wedge_propagation/") and name != recovery.SELF
    }
    run, snapshot = tmp_path / "source-fixture", tmp_path / "snapshot"
    run.mkdir()
    snapshot.mkdir()
    recovery.write(run / "study_contract.json", {"source_sha256": original})
    with zipfile.ZipFile(run / "execution_sources.zip", "w") as archive:
        for name in original:
            archive.write(repository / name, name)
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((repository / name).read_bytes())
    report = recovery.source_audit(run, repository)
    assert not report["blocked"]
    assert all(row["unrelated_addition"] for row in report["changes"])
    program = """
import json, sys
from pathlib import Path
import chartgat
from experiments.output_init_ablation import study, train
root = Path.cwd()
for module in (chartgat, study, train):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
args = train.parser().parse_args(study.command("baseline", "calibrate", root / "unused", root)[4:])
train.validate(args)
expected = json.loads(Path(sys.argv[1]).read_text())["source_sha256"]
assert train.research_contract(args, {})["source_sha256"] == expected
print("archived imports and source inventory match")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(snapshot), str(snapshot / "src")))
    child = subprocess.run(
        [sys.executable, "-B", "-c", program, str(run / "study_contract.json")],
        cwd=snapshot,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_cuda_reuses_learned_and_trains_only_missing_fixed(tmp_path, monkeypatch):
    import torch

    from experiments.c_learning_bracket.test_debug import debug_payload
    from experiments.output_init_ablation import train
    from experiments.output_init_ablation.test_debug import arguments

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for recovery smoke")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        args = arguments("kaiming_relu")
        args.conductance_evaluation = "log_row"
        args.epochs = 2  # Explicit synthetic smoke only; production still requires >=200.
        args.sample_seed_batch_size = 64
        args.sample_context_seed_batch_size = 32
        args.eval_context_seeds = [16, 32]
        args.pin_memory = False
        args.sample_prefetch = False
        payload = debug_payload()
        device = torch.device("cuda")
        source, target = tmp_path / "original-debug", tmp_path / "recovered-debug"
        source.mkdir()
        target.mkdir()
        learned = train.train_one(payload, args, "learned", source, device)
        original = {p.name: recovery.digest(p) for p in source.iterdir()}
        record = recovery.completed_condition(source, "learned", 2, synthetic_debug=True)
        assert record == learned
        verified = recovery.verify_reused(train, payload, args, source, "learned", record, device)
        assert verified["correct"] == record["best"]["validation"]["correct"]
        bad = copy.deepcopy(record)
        bad["best"]["validation"]["correct"] += 1
        with pytest.raises(ValueError, match="not reproduced"):
            recovery.verify_reused(train, payload, args, source, "learned", bad, device)
        selected = {"learned": recovery.copy_condition(source, target, "learned", record)}
        args.output_dir = target
        called, checks = [], []
        original_train = train.train_one

        def track(*positional, **keyword):
            called.append(positional[2])
            return original_train(*positional, **keyword)

        monkeypatch.setattr(train, "train_one", track)
        recovery.complete_conditions(
            train, payload, args, selected, device, lambda: checks.append(True)
        )
        assert called == ["fixed"] and len(checks) == 4
        assert (args.layers, args.hidden_channels, args.heads) == (8, 256, 8)
        train.evaluate_selected(payload, args, selected, target, device)
        result = recovery.read(target / "evaluation.json")
        assert result["run_completed"] and not result["test_set_evaluated"]
        assert original == {p.name: recovery.digest(p) for p in source.iterdir()}
    finally:
        torch.set_num_threads(previous_threads)
