"""Run both calibrations, four full training conditions, and a paired comparison."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

from experiments.aggregation_comparison import engine

from .model import INITIALIZATIONS
from .progress import announce, wait_with_progress
from .train import parser as run_parser
from .train import research_contract, validate, write_json


def command(initialization, action, output, data_root, calibration=None):
    argv = [
        sys.executable,
        "-u",
        "-m",
        "experiments.output_init_ablation.train",
        "--output-initialization",
        initialization,
        "--conductance-evaluation",
        "log_row",
        "--action",
        action,
        "--output-dir",
        str(output),
        "--data-root",
        str(data_root),
        "--sample-seed-batch-size",
        "2048",
        "--sample-context-seed-batch-size",
        "2048",
        "--sample-context-workers",
        "4",
        "--edge-chunk-size",
        "16384",
        "--eval-context-seeds",
        "2048",
        "4096",
    ]
    if action == "calibrate":
        argv.extend(
            [
                "--physical-seed-candidates",
                "2048",
                "4096",
                "--context-worker-candidates",
                "2",
                "4",
            ]
        )
    else:
        argv.extend(["--calibration-report", str(calibration)])
    return argv


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signal_trajectory(root, *, expected_epochs=200):
    rows = []
    for initialization in INITIALIZATIONS:
        folder = root / ("train-" + initialization)
        for condition in ("learned", "fixed"):
            for epoch in range(1, expected_epochs + 1):
                inspection = read(folder / f"{condition}-inspection-{epoch:04d}.json")
                layers = []
                for layer in inspection["layers"]:
                    layers.append(
                        {
                            "layer": layer["layer"],
                            "stage_rms": {
                                key: value["rms"]
                                for key, value in layer["signal_stages_before_update"].items()
                            },
                            "conductance_coordinate": "log_c" if "log_c" in layer else "c",
                            "conductance_std_by_head": layer.get("log_c", layer.get("c"))["before"][
                                "std"
                            ],
                            "conductance_update_max_by_head": layer.get("log_c", layer.get("c"))[
                                "change"
                            ]["max_abs"],
                            "alpha_update_max_by_head": layer["alpha"]["change"]["max_abs"],
                            "alpha_resolved_by_head": layer[
                                "alpha_change_resolved_above_observed_replay_by_head"
                            ],
                            "live_conductance_gradient": layer.get(
                                "live_log_c_gradient_norm_and_max_by_head",
                                layer.get("live_c_gradient_norm_and_max_by_head"),
                            ),
                        }
                    )
                rows.append(
                    {
                        "initialization": initialization,
                        "condition": condition,
                        "epoch": epoch,
                        "layers": layers,
                    }
                )
    return {
        "scope": "first actual training batch of each epoch; all layers and heads",
        "interpretation": "numerical C change and prediction usefulness are separate",
        "observations": rows,
    }


def compare_results(root, *, expected_epochs=200):
    """Only complete matched runs may produce a final comparison."""
    results = {}
    trained = {}
    initial = {}
    for initialization in INITIALIZATIONS:
        folder = root / ("train-" + initialization)
        evaluation = read(folder / "evaluation.json")
        if not evaluation["run_completed"]:
            raise ValueError("cannot compare incomplete runs")
        results[initialization] = evaluation["results"]
        trained[initialization] = {}
        initial[initialization] = {}
        for condition in ("learned", "fixed"):
            record = read(folder / f"{condition}-trained.json")
            history = record["history"]
            if len(history) != expected_epochs:
                raise ValueError("comparison requires every declared epoch")
            if [x["epoch"] for x in history] != list(range(1, expected_epochs + 1)):
                raise ValueError("epoch sequence differs")
            trained[initialization][condition] = record
            initial[initialization][condition] = read(folder / f"{condition}-initial.json")
    for condition in ("learned", "fixed"):
        before, after = (initial[k][condition] for k in INITIALIZATIONS)
        if before["initial_non_output_sha256"] != after["initial_non_output_sha256"]:
            raise RuntimeError("non-output initial parameters differ between initialization arms")
    reference = trained["baseline"]["learned"]["history"]
    for cases in trained.values():
        for record in cases.values():
            for a, b in zip(reference, record["history"], strict=True):
                if a["sampling_evidence"] != b["sampling_evidence"]:
                    raise RuntimeError("sampled contexts differ between study conditions")
    metrics = {}
    for initialization in INITIALIZATIONS:
        metrics[initialization] = {}
        for condition in ("learned", "fixed"):
            final = results[initialization][condition]
            full = final["full_validation"]
            metrics[initialization][condition] = {
                "selected_epoch": trained[initialization][condition]["best"]["epoch"],
                "full_validation": full,
                "new_sampling_contexts": final["new_sampling_contexts"],
                "total_training_seconds": sum(
                    x["seconds_including_inspection_validation"]
                    for x in trained[initialization][condition]["history"]
                ),
            }
    benefits = {
        initialization: results[initialization]["learned"]["full_validation"]["accuracy"]
        - results[initialization]["fixed"]["full_validation"]["accuracy"]
        for initialization in INITIALIZATIONS
    }
    return {
        "complete": True,
        "epochs_per_condition": expected_epochs,
        "conditions": metrics,
        "learned_minus_fixed_validation_accuracy": benefits,
        "change_in_learned_minus_fixed_benefit": benefits["kaiming_relu"] - benefits["baseline"],
        "pairing_verified": True,
        "scope": (
            "one model seed; validation-selected checkpoints; "
            "no official test or significance claim"
        ),
        "c_benefit_confirmed": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data_root = args.data_root.resolve()
    probe = run_parser().parse_args(command("baseline", "calibrate", root / "probe", data_root)[4:])
    validate(probe)
    sources = research_contract(probe, {})["source_sha256"]
    write_json(
        root / "study_contract.json",
        {
            "source_sha256": sources,
            "initializations": INITIALIZATIONS,
            "conditions": ["learned", "fixed"],
            "epochs_per_condition": 200,
            "model_seed": 0,
            "physical_seed_batch": 2048,
            "effective_seed_batch": 2048,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "layers": 8,
            "hidden_channels": 256,
            "heads": 8,
            "conductance_evaluation": "log_row",
            "calibration_physical_candidates": [2048, 4096],
            "calibration_worker_candidates": [2, 4],
            "data_root": str(data_root),
            "single_factor": "output projection initial weights multiplied by sqrt(6)",
            "all_other_initial_weights_and_sample_sequences_must_match": True,
            "parent_pid": os.getpid(),
            "started_at_unix": time.time(),
        },
    )
    with zipfile.ZipFile(root / "execution_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(engine.ROOT / name, name)
        for path in Path(__file__).parent.glob("*.md"):
            archive.write(path, path.relative_to(engine.ROOT))

    def verify_sources():
        for name, expected in sources.items():
            if hashlib.sha256((engine.ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"source changed during study: {name}")

    completed = []

    def run(initialization, action):
        verify_sources()
        label = action + "-" + initialization
        output = root / label
        calibration = root / ("calibrate-" + initialization) / "calibration.json"
        argv = command(initialization, action, output, data_root, calibration)
        announce(f"START {label} | results: {output}")
        with (root / (label + ".log")).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(argv, cwd=engine.ROOT, stdout=log, stderr=subprocess.STDOUT)
            write_json(
                root / (label + "-started.json"),
                {
                    "command": argv,
                    "pid": child.pid,
                    "started_at_unix": time.time(),
                },
            )
            # Owned status file only; no user files or experiment results are replaced.
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "phase": label,
                        "child_pid": child.pid,
                        "completed": completed,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return_code = wait_with_progress(child, root / (label + ".log"), label)
        if return_code:
            raise RuntimeError(
                f"{label} failed with code {return_code}; log: {root / (label + '.log')}"
            )
        verify_sources()
        completed.append(label)
        write_json(root / (label + "-completed.json"), {"completed_at_unix": time.time()})
        announce(f"DONE {label}")

    try:
        # Inspect the candidate first; both arms are calibrated before any full training.
        for initialization in ("kaiming_relu", "baseline"):
            run(initialization, "calibrate")
        for initialization in INITIALIZATIONS:
            run(initialization, "train")
        verify_sources()
        comparison = compare_results(root)
        write_json(root / "signal_trajectory.json", signal_trajectory(root))
        write_json(root / "comparison.json", comparison)
        announce(f"STUDY COMPLETE | comparison: {root / 'comparison.json'}")
        (root / "status.json").write_text(
            json.dumps(
                {
                    "state": "complete",
                    "completed": completed,
                    "comparison": "comparison.json",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as error:
        write_json(
            root / "study_failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "completed": completed,
                "fallback_used": False,
            },
        )
        (root / "status.json").write_text(
            json.dumps(
                {
                    "state": "failed",
                    "completed": completed,
                    "error": str(error),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
