"""Audit a stopped output-init study; reuse complete conditions under archived sources.

Default is read-only. --output-dir creates a separate recovery and executes CUDA
validation plus only untouched conditions. Partial conditions are never resumed
from a best-weight checkpoint (which has no optimizer/RNG state).
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/recover_output_init_study.py"
INITIALIZATIONS = ("baseline", "kaiming_relu")
CONDITIONS = ("learned", "fixed")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, default=str))


def safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or ":" in name
        or "\\" in name
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe archive path: {name}")
    return path


def source_audit(run, repository):
    """Allow only the independently added wedge package and this external driver.

    Never allow a change to an old training/model/data source. This deliberately
    narrow exception addresses this incident, not arbitrary source drift.
    """
    expected = read(run / "study_contract.json")["source_sha256"]
    if not expected:
        raise ValueError("empty source manifest")
    with zipfile.ZipFile(run / "execution_sources.zip") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        for name in names:
            safe_name(name)
            if name not in expected and not (
                name.startswith("experiments/output_init_ablation/") and name.endswith(".md")
            ):
                raise ValueError(f"unbound archive entry: {name}")
        for name, sha in expected.items():
            safe_name(name)
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError(f"archive hash mismatch: {name}")
        # The accepted original sources must not refer to the unrelated package.
        for name in expected:
            if name.endswith(".py") and not name.startswith("research/wedge_propagation/"):
                if b"wedge_propagation" in archive.read(name):
                    raise ValueError(f"wedge independence not established: {name}")
    current = set(expected)
    for directory in ("research", "src/chartgat", "scripts"):
        current.update(
            p.relative_to(repository).as_posix() for p in (repository / directory).rglob("*.py")
        )
    current.update(
        p.relative_to(repository).as_posix() for p in (repository / "research").rglob("*.yaml")
    )
    for directory in (
        "aggregation_comparison",
        "incidence_ablation",
        "c_learning_only",
        "c_learning_bracket",
        "output_init_ablation",
    ):
        current.update(
            p.relative_to(repository).as_posix()
            for p in (repository / "experiments" / directory).glob("*.py")
        )
    changes, blocked = [], []
    for name in sorted(current):
        path = repository / name
        actual = digest(path) if path.is_file() else None
        previous = expected.get(name)
        if actual != previous:
            # Only additions are accepted. Edits/deletions of original files block.
            allowed = previous is None and (
                name.startswith("research/wedge_propagation/") or name == SELF
            )
            changes.append(
                {"path": name, "before": previous, "now": actual, "unrelated_addition": allowed}
            )
            if not allowed:
                blocked.append(name)
    return {
        "source_sha256": expected,
        "changes": changes,
        "blocked": blocked,
        "archive_sha256": digest(run / "execution_sources.zip"),
        "limitation": (
            "compares saved start sources with current disk; no historical filesystem trace"
        ),
    }


def completed_condition(folder, condition, epochs, *, synthetic_debug=False):
    path = folder / f"{condition}-trained.json"
    if not path.exists():
        if list(folder.glob(f"{condition}-*")):
            raise ValueError(f"partial {folder.name}/{condition}: no optimizer/RNG resume state")
        return None
    record = read(path)
    initial = read(folder / f"{condition}-initial.json")
    history = record["history"]
    if (
        record["debug"] != synthetic_debug
        or initial["debug"] != synthetic_debug
        or initial["epochs"] != epochs
    ):
        raise ValueError("debug or different epoch budget cannot be reused")
    if [row["epoch"] for row in history] != list(range(1, epochs + 1)):
        raise ValueError("completed condition must contain all consecutive epochs")
    if initial["train_coverage"] != 1.0:
        raise ValueError("incomplete training coverage")
    if initial["common_sha256"] != record["initial_common_sha256"]:
        raise ValueError("initial parameter evidence differs")
    for row in history:
        epoch = row["epoch"]
        if read(folder / f"{condition}-epoch-{epoch:04d}.json") != row:
            raise ValueError(f"epoch evidence mismatch: {condition}/{epoch}")
        inspection = read(folder / f"{condition}-inspection-{epoch:04d}.json")
        if inspection["debug"] != synthetic_debug or inspection["epoch"] != epoch:
            raise ValueError("inspection evidence mismatch")
        if (
            row["debug"] != synthetic_debug
            or row["supervised_nodes"] != initial["train_nodes_used_per_epoch"]
        ):
            raise ValueError("incomplete supervised epoch")
    if sum(row["optimization_steps"] for row in history) != initial["optimization_steps"]:
        raise ValueError("optimization step count differs")
    best = record["best"]
    selected = max(history, key=lambda row: row["validation"]["correct"])
    if best["epoch"] != selected["epoch"] or best["validation"] != selected["validation"]:
        raise ValueError("best checkpoint selection differs from history")
    checkpoint = folder / f"{condition}-best-{best['epoch']:04d}.pt"
    if Path(best["path"]).name != checkpoint.name or digest(checkpoint) != best["file_sha256"]:
        raise ValueError("selected checkpoint hash/path mismatch")
    return record


def audit(run, repository):
    report = source_audit(run, repository)
    study = read(run / "study_contract.json")
    epochs = study["epochs_per_condition"]
    if epochs < 200 or study["conditions"] != list(CONDITIONS):
        raise ValueError("unexpected production study contract")
    report["conditions"] = {}
    for initialization in INITIALIZATIONS:
        folder = run / f"train-{initialization}"
        # This recovery handles the reported late training failure, not absent calibration.
        contract = read(folder / "contract.json")
        calibration = read(run / f"calibrate-{initialization}" / "calibration.json")
        if contract["source_sha256"] != report["source_sha256"]:
            raise ValueError("child and study source manifests differ")
        if any(contract.get(key) != value for key, value in calibration["contract"].items()):
            raise ValueError("saved calibration/scientific contract differs")
        if contract["configuration"]["epochs"] != epochs:
            raise ValueError("study and child epoch budgets differ")
        report["conditions"][initialization] = {
            condition: completed_condition(folder, condition, epochs) is not None
            for condition in CONDITIONS
        }
    return report


def verify_reused(train, payload, args, folder, condition, record, device):
    """On CUDA, recheck dataset identity, initial pairing, checkpoint and validation."""
    import torch

    inputs = train.StudyInputs(payload, args)
    initial = read(folder / f"{condition}-initial.json")
    if inputs.metadata()["source_sha256"] != initial["inputs"]["source_sha256"]:
        raise ValueError("graph, features, labels or splits changed")
    model = train.make_model(payload, args, device, condition)
    if (
        train.common_digest(model) != record["initial_common_sha256"]
        or train.non_output_digest(model) != initial["initial_non_output_sha256"]
    ):
        raise ValueError("recreated initialization differs")
    checkpoint = folder / Path(record["best"]["path"]).name
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if (
        saved["suite"] != train.SUITE
        or saved["debug"] != bool(payload.get("explicit_synthetic_debug", False))
        or saved["condition"] != condition
        or saved["epoch"] != record["best"]["epoch"]
        or saved["model_contract"] != model.contract()
    ):
        raise ValueError("checkpoint model contract mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    validation = train.evaluate(
        model, inputs.validation_batches(device), expected_seed_ids=inputs.indices["validation"]
    )
    expected = record["best"]["validation"]
    for key in ("correct", "total", "parameter_sha256", "seed_coverage"):
        if validation[key] != expected[key]:
            raise ValueError(f"saved best validation not reproduced: {key}")
    return validation


def copy_condition(source, target, condition, record):
    """Copy completed evidence to a fresh output; rebind only the checkpoint path."""
    copied = copy.deepcopy(record)
    checkpoint = Path(record["best"]["path"]).name
    names = [f"{condition}-initial.json", checkpoint]
    for row in record["history"]:
        names.extend(
            f"{condition}-{kind}-{row['epoch']:04d}.json" for kind in ("epoch", "inspection")
        )
    evidence = {}
    for name in names:
        destination = target / name
        with (source / name).open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        evidence[name] = digest(destination)
        if evidence[name] != digest(source / name):
            raise ValueError(f"source artifact changed while copying: {name}")
    copied["best"]["path"] = str(target / checkpoint)
    write(target / f"{condition}-trained.json", copied)
    write(
        target / f"{condition}-reused.json",
        {
            "source": str(source),
            "sha256": evidence,
            "source_trained_sha256": digest(source / f"{condition}-trained.json"),
        },
    )
    return copied


def verify_pairing(selected, output):
    reference = None
    non_output = {}
    for initialization, conditions in selected.items():
        for condition, record in conditions.items():
            initial = read(output / f"train-{initialization}" / f"{condition}-initial.json")
            previous = non_output.setdefault(condition, initial["initial_non_output_sha256"])
            if previous != initial["initial_non_output_sha256"]:
                raise ValueError("reused non-output initialization pairing differs")
            evidence = [row["sampling_evidence"] for row in record["history"]]
            if reference is None:
                reference = evidence
            elif evidence != reference:
                raise ValueError("reused sampled-context pairing differs")
        if set(conditions) == set(CONDITIONS):
            if (
                conditions["learned"]["initial_common_sha256"]
                != conditions["fixed"]["initial_common_sha256"]
            ):
                raise ValueError("reused learned/fixed initialization differs")


def complete_conditions(train, payload, args, selected, device, check_contract):
    for condition in CONDITIONS:
        check_contract()
        if condition not in selected:
            print(
                f"[recovery] TRAIN missing {args.output_initialization}/{condition}: "
                f"{args.epochs} epochs",
                flush=True,
            )
            selected[condition] = train.train_one(
                payload,
                args,
                condition,
                args.output_dir,
                device,
                selected.get("learned") if condition == "fixed" else None,
            )
            gc.collect()
        check_contract()
    return selected


def recover_worker(run, output, snapshot):
    # Import only after putting the archived tree before any editable installation.
    sys.path[:0] = [str(snapshot), str(snapshot / "src")]
    import torch

    import chartgat
    from experiments.output_init_ablation import study, train

    for module in (chartgat, study, train):
        if not Path(module.__file__).resolve().is_relative_to(snapshot):
            raise RuntimeError(f"module escaped archived sources: {module.__name__}")
    expected_sources = read(run / "study_contract.json")["source_sha256"]
    selected, recipes, contracts, payloads = {}, {}, {}, {}
    device = torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("recovery requires exactly one visible CUDA allocation")
    # Validate every reused arm BEFORE starting any additional 200-epoch condition.
    for initialization in INITIALIZATIONS:
        label = f"train-{initialization}"
        source = run / label
        command = read(run / f"{label}-started.json")["command"]
        if command[1:4] != ["-u", "-m", "experiments.output_init_ablation.train"]:
            raise ValueError("unexpected original training command")
        args = train.parser().parse_args(command[4:])
        train.validate(args)
        if args.action != "train" or args.output_initialization != initialization:
            raise ValueError("training command and phase differ")
        if args.output_dir.resolve() != source:
            raise ValueError("original output path differs")
        if (
            args.calibration_report.resolve()
            != run / f"calibrate-{initialization}/calibration.json"
        ):
            raise ValueError("calibration path is outside original study")
        train.engine.base.configure_compute(args)
        train.engine.base.validate_hardware_runtime(args, device)
        payload, protocol = train.engine.load_dataset(args)
        train.require_approval(protocol)
        frozen = normalized(train.research_contract(args, protocol))
        saved = read(source / "contract.json")
        if frozen["source_sha256"] != expected_sources or any(
            saved.get(k) != v for k, v in frozen.items()
        ):
            raise ValueError("archived source/science/data contract cannot be reproduced")
        train.validate_calibration(args, protocol, device)
        target = output / label
        target.mkdir()
        args.output_dir = target
        write(
            target / "contract.json",
            {
                **saved,
                "arguments": vars(args),
                "recovery_resources": train.resources(device),
                "recovered_from": str(source),
            },
        )
        selected[initialization] = {}
        for condition in CONDITIONS:
            record = completed_condition(source, condition, args.epochs)
            if record is not None:
                print(
                    f"[recovery] CUDA verify {initialization}/{condition}; no retraining",
                    flush=True,
                )
                validation = verify_reused(train, payload, args, source, condition, record, device)
                selected[initialization][condition] = copy_condition(
                    source, target, condition, record
                )
                write(target / f"{condition}-reproduction.json", validation)
                gc.collect()
        recipes[initialization], contracts[initialization] = args, frozen
        payloads[initialization] = (payload, protocol)
    verify_pairing(selected, output)
    for initialization in INITIALIZATIONS:
        args = recipes[initialization]
        payload, protocol = payloads[initialization]

        def check_contract(args=args, protocol=protocol, initialization=initialization):
            if normalized(train.research_contract(args, protocol)) != contracts[initialization]:
                raise RuntimeError("archived source/science changed during recovery")

        complete_conditions(train, payload, args, selected[initialization], device, check_contract)
        train.evaluate_selected(payload, args, selected[initialization], args.output_dir, device)
    comparison = study.compare_results(output)
    write(output / "signal_trajectory.json", study.signal_trajectory(output))
    for initialization in INITIALIZATIONS:
        _, protocol = payloads[initialization]
        if (
            normalized(train.research_contract(recipes[initialization], protocol))
            != contracts[initialization]
        ):
            raise RuntimeError("archived source/science changed during evaluation")
    write(output / "comparison.json", comparison)
    write(
        output / "recovery-completed.json",
        {"complete": True, "original_run": str(run), "test_set_evaluated": False},
    )


def launch(run, output, report):
    if report["blocked"]:
        raise ValueError(f"unverified source changes; recovery blocked: {report['blocked']}")
    if output.exists() or output.is_relative_to(run):
        raise ValueError("choose a fresh output outside the original run")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "recovery-audit.json", report)
    snapshot = output / "execution_sources"
    snapshot.mkdir()
    with zipfile.ZipFile(run / "execution_sources.zip") as archive:
        for name in archive.namelist():
            safe_name(name)
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(archive.read(name))
    # The driver lives outside the archived source inventory and is itself frozen.
    driver = output / "recovery_driver.py"
    with driver.open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    write(output / "recovery-driver.json", {"sha256": digest(driver)})
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(snapshot), str(snapshot / "src")))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    argv = [
        sys.executable,
        "-u",
        str(driver),
        "--run-dir",
        str(run),
        "--output-dir",
        str(output),
        "--worker-snapshot",
        str(snapshot),
    ]
    # Import the archived console-only progress helper, without training imports.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recovery_progress", snapshot / "experiments/output_init_ablation/progress.py"
    )
    progress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(progress)
    log_path = output / "recovery.log"
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            argv, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        write(output / "recovery-started.json", {"pid": child.pid, "command": argv})
        code = progress.wait_with_progress(child, log_path, "recovery")
    if code:
        raise RuntimeError(f"recovery failed with code {code}; evidence preserved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.worker_snapshot is not None:
        try:
            recover_worker(run, args.output_dir.resolve(), args.worker_snapshot.resolve())
        except Exception as error:
            write(
                args.output_dir / "recovery-failure.json",
                {"error": str(error), "traceback": traceback.format_exc()},
            )
            raise
        return
    report = audit(run, args.repository.resolve())
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "source_sha256"}, indent=2, ensure_ascii=False
        ),
        flush=True,
    )
    if args.output_dir is not None:
        launch(run, args.output_dir.resolve(), report)


if __name__ == "__main__":
    main()
