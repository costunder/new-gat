"""CPU analysis of completed Experiment 4.1 CSVs; no model or optimizer execution."""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from ..classification.common import digest, file_sha256, write_csv, write_json
from .core import analyze
from .report import write_report
from .source import assert_unchanged, load_run


class _Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def source_manifest():
    folder = Path(__file__).parent
    hashes = {path.name: file_sha256(path) for path in sorted(folder.glob("*.py"))}
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={folder.parents[2].as_posix()}", "rev-parse", "HEAD"],
            cwd=folder,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return {"sha256": hashes, "code_digest": digest(hashes), "git_commit": commit}


def cpu_resources(output):
    process, memory = psutil.Process(), psutil.virtual_memory()
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
    disk = shutil.disk_usage(output)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu": os.cpu_count(),
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "process_rss_bytes": process.memory_info().rss,
        "cpu_seconds": sum(process.cpu_times()[:2]),
        "output_storage_free_bytes": disk.free,
        "selected_device": "cpu",
        "gpu_count_used": 0,
        "physical_batch_size": "N/A_saved_scalar_tables",
        "data_loader_workers": "N/A_single_read_of_saved_tables",
        "precision": "float64_scalar_analysis_of_saved_float32_measurements",
        "parallelism": "seed_manifest_groups_from_all_saved_rows",
    }


def run(args):
    source_dir, output = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    if output.exists() or output.is_relative_to(source_dir):
        raise ValueError("output must be a NEW directory outside the preserved source run")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "terminal.log").open("x", encoding="utf-8") as logfile:
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, logfile)),
            contextlib.redirect_stderr(_Tee(sys.stderr, logfile)),
        ):
            try:
                started = time.perf_counter()
                code = source_manifest()
                write_json(output / "source.json", code)
                print(
                    f"[start] source={source_dir} device=cpu csv_only=true "
                    "model_forwards=0 optimizer_updates=0 all_saved_seeds_and_manifests",
                    flush=True,
                )
                hardware = cpu_resources(output)
                print(f"[hardware] {hardware}", flush=True)
                source = load_run(source_dir)
                config = source.config
                write_json(output / "config.json", config)
                print(
                    f"[source] verified profile={config['profile']} "
                    f"datasets={config['data']['datasets']} final_seeds={config['final_seeds']} "
                    f"manifests={config['manifests_per_dataset']} "
                    f"rows={source.completion['coverage']} sampling=1.0 subset=false",
                    flush=True,
                )
                print(
                    "[analysis] deriving each seed's beta/kappa and branch norm factors", flush=True
                )
                analysis = analyze(
                    config,
                    source.tables["baseline_rows"],
                    source.tables["treatment_rows"],
                    source.tables["diagnostic_rows"],
                    source.tables["fixed_z_rows"],
                )
                for name in (
                    "baseline_strength",
                    "strength_estimates",
                    "layer_changes",
                    "fixed_estimates",
                ):
                    write_csv(output / f"{name}.csv", analysis[name])
                write_json(output / "coverage.json", analysis["coverage"])
                print(
                    f"[analysis] per_seed_layers={len(analysis['baseline_strength'])} "
                    f"strength_estimates={len(analysis['strength_estimates'])} "
                    f"all_split_layer_changes={len(analysis['layer_changes'])} "
                    f"fixed_estimates={len(analysis['fixed_estimates'])}",
                    flush=True,
                )
                assert_unchanged(source)
                if source_manifest()["code_digest"] != code["code_digest"]:
                    raise RuntimeError("analysis implementation changed during execution")
                provenance = {
                    "source_dir": str(source.directory),
                    "source_config": source.contract["source_config"],
                    "source_completion": source.completion,
                    "source_hashes": source.hashes,
                    "source_code_digest": source.contract["diagnostic_source"]["code_digest"],
                    "analysis_source": code,
                    "source_files_preserved": True,
                    "optimizer_updates": 0,
                    "model_forwards": 0,
                    "csv_only": True,
                    "hardware": hardware,
                }
                write_json(output / "contract.json", provenance)
                print("[report] writing saved-message factor and layer comparisons", flush=True)
                names = write_report(output, config, analysis, provenance)
                summary = (output / names["summary"]).read_text(encoding="utf-8")
                assert_unchanged(source)
                if source_manifest()["code_digest"] != code["code_digest"]:
                    raise RuntimeError("analysis implementation changed during reporting")
                final_resources = cpu_resources(output)
                seconds = time.perf_counter() - started
                write_json(output / "resources.json", {"start": hardware, "end": final_resources})
                write_json(
                    output / "completion.json",
                    {
                        "completed": True,
                        "profile": config["profile"],
                        "scope": "saved_branch_strength_analysis",
                        "actual_data": source.completion["actual_data"],
                        "source_coverage": source.completion["coverage"],
                        "analysis_rows": {
                            key: len(analysis[key])
                            for key in (
                                "baseline_strength",
                                "strength_estimates",
                                "layer_changes",
                                "fixed_estimates",
                            )
                        },
                        "optimizer_updates": 0,
                        "model_forwards": 0,
                        "source_files_preserved": True,
                        "seconds": seconds,
                    },
                )
                print(summary, flush=True)
                print(
                    f"[complete] results={output} seconds={seconds:.3f} "
                    "model_forwards=0 optimizer_updates=0 source_preserved=true",
                    flush=True,
                )
            except Exception as error:
                traceback.print_exc()
                write_json(
                    output / "failure.json",
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "source_and_partial_results_preserved": True,
                    },
                )
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
