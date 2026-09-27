"""Arxiv fixed/learned C x full/sampled study, with a global test barrier."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from chartgat.cache import atomic_write_json
from scripts.calibration_lock import calibration_lock

from . import engine, final_test, provenance, runner

ARMS = ("incidence_fixed", "incidence")
MODES = ("full", "sampled")


def parser():
    result = runner.parser()
    result.description = __doc__
    result.set_defaults(
        arms=list(ARMS),
        learning_budget_policy="epochs",
        complete_supervised_passes=True,
        sampling="cluster_disjoint",
    )
    result.add_argument("--include-sampled-baselines", action="store_true")
    return result


def validate(args):
    runner.validate_args(args)
    if args.sampling != "cluster_disjoint":
        raise ValueError("core sampled cells require cluster_disjoint contexts")
    if set(args.arms) != set(ARMS) or args.datasets != ["ogbn-arxiv"]:
        raise ValueError("core study requires all four arxiv fixed/dynamic C cells")
    if not args.complete_supervised_passes or args.learning_budget_policy != "epochs":
        raise ValueError("core study requires complete exposure-matched supervised passes")
    if args.sample_context_seed_batch_size is None:
        raise ValueError("declare --sample-context-seed-batch-size; context is a research variable")
    if len(args.run_id) > 100:
        raise ValueError("core run ID must leave room for independent child identifiers")


def child_argv(args, root, mode):
    argv = [
        "--run-id",
        args.run_id + "-" + mode,
        "--datasets",
        "ogbn-arxiv",
        "--profiles",
        *args.profiles,
        "--model-seeds",
        *map(str, args.model_seeds),
        "--arms",
        *(ARMS + (("gcn", "graphsage") if args.include_sampled_baselines else ())),
        "--results-root",
        str(root / "children"),
        "--learning-budget-policy",
        "epochs",
        "--complete-supervised-passes",
        "--sampling",
        "full" if mode == "full" else "cluster_disjoint",
    ]
    argv += [
        "--visibility-protocol",
        args.visibility_protocol,
        "--gram-implementation",
        args.gram_implementation,
    ]
    if args.include_sampled_baselines:
        argv.append("--sampled-local-baselines")
    for name in (
        "data_root",
        "device",
        "hardware_profile",
        "cuda_allocator_limit_gib",
        "epochs",
        "patience",
        "workers",
        "learning_rate",
        "edge_chunk_size",
        "min_free_gb",
        "repeat_evaluations",
    ):
        value = getattr(args, name)
        if value is not None:
            argv.extend(("--" + name.replace("_", "-"), str(value)))
    argv.append(
        "--activation-checkpoint" if args.activation_checkpoint else "--no-activation-checkpoint"
    )
    if mode == "sampled":
        argv += [
            "--sample-context-seed-batch-size",
            str(args.sample_context_seed_batch_size),
            "--sample-context-workers",
            str(runner.sampling_context_configuration(args)["sample_context_workers"]),
            "--num-neighbors",
            *map(str, args.num_neighbors),
        ]
        if args.sample_seed_batch_size is not None:
            argv += ["--sample-seed-batch-size", str(args.sample_seed_batch_size)]
    if args.calibration_only:
        argv.append("--calibration-only")
    return argv


def child_root(args, root, mode):
    return root / "children" / "aggregation_comparison" / (args.run_id + "-" + mode)


def read_groups(args, root):
    groups = {}
    for mode in MODES:
        path = child_root(args, root, mode) / "manifest.json"
        if path.is_symlink():
            raise ValueError("indirect core child manifest")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "passed":
            raise ValueError("every core child must finish training and audit before test")
        child = runner.parser().parse_args(child_argv(args, root, mode))
        runner._resume(
            path,
            child,
            runner.make_jobs(child, path.parent),
            provenance.source_snapshot(),
            manifest["dependencies"],
        )
        groups[mode] = {"args": child, "manifest": manifest, "path": path}
    return groups


def verify_and_report(groups):
    """Reject unequal exposure/sample pairing before reporting any causal contrast."""
    rows = {}
    comparators = []
    for mode, group in groups.items():
        for job in group["manifest"]["jobs"]:
            metrics = engine.inspect_completed(job["output_dir"])
            if job["variant_id"] not in ARMS:
                comparators.append(
                    {
                        "mode": mode,
                        "arm": job["variant_id"],
                        "profile": job["profile"],
                        "seed": job["model_seed"],
                        "validation_accuracy": metrics["validation"],
                        "parameters": metrics["model_contract"]["total_parameters"],
                        "optimizer_steps": metrics["optimizer_steps"],
                        "resources": metrics["resource_observability"],
                    }
                )
                continue
            config = metrics["configuration"]
            if (
                not config["complete_supervised_passes"]
                or config["learning_budget_policy"] != "epochs"
            ):
                raise ValueError("core results lack the declared complete-pass budget")
            if config["sampling"] != ("full" if mode == "full" else "cluster_disjoint"):
                raise ValueError("core cell has the wrong sampling law")
            history = json.loads(
                (Path(job["output_dir"]) / "history.json").read_text(encoding="utf-8")
            )
            label = ("F" if mode == "full" else "S") + (
                "0" if job["variant_id"] == ARMS[0] else "1"
            )
            key = (job["profile"], job["model_seed"], label)
            if key in rows:
                raise ValueError("duplicate core cell")
            rows[key] = {
                "profile": key[0],
                "seed": key[1],
                "cell": label,
                "validation_accuracy": metrics["validation"],
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "history_sha256": metrics["history_sha256"],
                "source_sha256": metrics["source_sha256"],
                "data_protocol": metrics["protocol"],
                "common_initialization": metrics["shared_initial_state_sha256"],
                "common_incidence_initialization": metrics["resume_identity"][
                    "common_incidence_initial_sha256"
                ],
                "epochs": metrics["epochs_run"],
                "validation_observations": len(history),
                "optimizer_steps": metrics["optimizer_steps"],
                "supervised_exposures": sum(r["train_labels"] for r in history),
                "pass_evidence": [r["supervised_pass_evidence"] for r in history],
                "history": history,
                "total_parameters": metrics["model_contract"]["total_parameters"],
                "resource_observability": metrics["resource_observability"],
                "training_child_wall_seconds_including_IO": job.get("elapsed_seconds"),
                "sampling_configuration": {
                    name: config.get(name)
                    for name in (
                        "sampling",
                        "sample_seed_batch_size",
                        "sample_context_seed_batch_size",
                        "sample_context_workers",
                        "num_neighbors",
                    )
                },
            }
    contrasts = []
    if not rows or set(groups) != set(MODES):
        raise ValueError("core study requires full and sampled result groups")
    for profile, seed in sorted({(key[0], key[1]) for key in rows}):
        if any((profile, seed, name) not in rows for name in ("F0", "F1", "S0", "S1")):
            raise ValueError("core study requires every F0/F1/S0/S1 cell")
        cells = {name: rows[(profile, seed, name)] for name in ("F0", "F1", "S0", "S1")}
        for field in (
            "epochs",
            "validation_observations",
            "supervised_exposures",
            "data_protocol",
            "common_initialization",
            "common_incidence_initialization",
            "source_sha256",
        ):
            if any(row[field] != cells["F0"][field] for row in cells.values()):
                raise ValueError(f"core cells do not share {field}")
        seeds = [
            [proof["seed_counts_sha256"] for proof in row["pass_evidence"]]
            for row in cells.values()
        ]
        if any(value != seeds[0] for value in seeds):
            raise ValueError("core cells supervised different original seed IDs")
        if cells["S0"]["pass_evidence"] != cells["S1"]["pass_evidence"]:
            raise ValueError("sampled fixed/learned C saw different context sequences")
        f0, f1, s0, s1 = (cells[name]["validation_accuracy"] for name in ("F0", "F1", "S0", "S1"))
        contrasts.append(
            {
                "profile": profile,
                "seed": seed,
                "selection_split": "validation",
                "C_full": f1 - f0,
                "C_sampled": s1 - s0,
                "sampling_dynamic": s1 - f1,
                "interaction": s1 - s0 - f1 + f0,
            }
        )
    return {
        "cells": list(rows.values()),
        "comparators": comparators,
        "contrasts": contrasts,
        "interpretation": "exposure-matched training strategies; optimizer steps differ",
        "independent_graph_generalization_claimed": False,
        "parameter_matched": False,
    }


def freeze_and_test(args, groups, master, persist):
    # Verify EVERY group's actual training and audit evidence before ANY test.
    for group in groups.values():
        final_test.freeze_matrix(group["args"], group["manifest"], lambda: None)
    report = verify_and_report(groups)
    lock = {
        mode + "/" + job["job_id"]: job["result"]["checkpoint_sha256"]
        for mode, group in groups.items()
        for job in group["manifest"]["jobs"]
    }
    if master.get("checkpoint_lock") not in (None, lock):
        raise ValueError("core study checkpoint matrix changed after freeze")
    master.update(checkpoint_lock=lock, validation_report=report)
    persist()  # ALL four cells frozen before ANY official test access.
    if args.evaluate_test:
        for group in groups.values():
            final_test.evaluate_matrix(
                group["args"],
                group["manifest"],
                lambda group=group: atomic_write_json(group["path"], group["manifest"]),
            )
            runner._summary(group["path"].parent, group["manifest"])
        master["official_test"] = {
            mode: group["manifest"]["official_test"] for mode, group in groups.items()
        }
        contrasts = []
        for row in report["contrasts"]:
            prefix = f"{row['profile']}/ogbn-arxiv/model-seed-{row['seed']}/"
            scores = [
                master["official_test"][mode]["results"][prefix + arm]["evaluation"]["metric"]
                for mode, arm in (
                    ("full", ARMS[0]),
                    ("full", ARMS[1]),
                    ("sampled", ARMS[0]),
                    ("sampled", ARMS[1]),
                )
            ]
            f0, f1, s0, s1 = scores
            contrasts.append(
                {
                    "profile": row["profile"],
                    "seed": row["seed"],
                    "C_full": f1 - f0,
                    "C_sampled": s1 - s0,
                    "sampling_dynamic": s1 - f1,
                    "interaction": s1 - s0 - f1 + f0,
                    "selection_split": "validation",
                    "evaluation_protocol": args.visibility_protocol,
                }
            )
        report["heldout_contrasts"] = contrasts
    return report


def execute(args):
    validate(args)
    root = args.results_root.resolve() / "core_conductance" / args.run_id
    if root.is_relative_to(args.data_root.resolve()) or args.data_root.resolve().is_relative_to(
        root
    ):
        raise ValueError("core outputs must be outside the dataset cache")
    if args.dry_run:
        for mode in MODES:
            child = runner.parser().parse_args(child_argv(args, root, mode))
            jobs = runner.make_jobs(child, child_root(args, root, mode))
            print(json.dumps({"mode": mode, "jobs": jobs}, default=str))
        return 0
    with calibration_lock(root):
        path = root / "manifest.json"
        identity = {"config": runner._config(args), "source_sha256": provenance.source_snapshot()}
        if path.exists():
            master = json.loads(path.read_text(encoding="utf-8"))
            if master["identity"] != identity:
                raise ValueError("core identity changed; use a fresh run and preserve evidence")
        else:
            master = {"schema_version": 1, "identity": identity, "status": "running"}
        root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        def persist():
            atomic_write_json(path, master)

        persist()
        try:
            for mode in MODES:
                status = runner.main(child_argv(args, root, mode))
                if status:
                    raise RuntimeError(f"core {mode} child failed with status {status}")
            if args.calibration_only:
                master["status"] = "calibrated"
            else:
                groups = read_groups(args, root)
                report = freeze_and_test(args, groups, master, persist)
                atomic_write_json(root / "comparison.json", report)
                master["status"] = "passed"
            master["last_invocation_wall_seconds_including_calibration_audit_IO"] = (
                time.perf_counter() - started
            )
            persist()
            print(f"Core study {master['status']}: {root}")
            return 0
        except (Exception, KeyboardInterrupt) as error:
            master.update(status="failed", error=f"{type(error).__name__}: {error}")
            persist()
            raise


def main(argv=None):
    try:
        return execute(parser().parse_args(argv))
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Core study stopped safely: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
