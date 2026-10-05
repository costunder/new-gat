"""CUDA component validation on complete cached citation inputs, without models.

This is resource/chunk/packing verification, not frozen trained-checkpoint results.
No new data download, training, graph/feature subset or synthetic prediction.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from research.wedge_propagation.classification.common import file_sha256, write_json
from research.wedge_propagation.classification.data import EXPECTED, SCHEMA, load_graph
from research.wedge_propagation.study import available_cpus, runtime_resources

from .energy import build_topology, measure
from .study import ROOT, calibrate, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/wedge-citation")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one allocated visible CUDA GPU is required")
    device = torch.device("cuda:0")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    implementation = sources()
    hardware = runtime_resources(device)
    workers = available_cpus(hardware)
    write_json(output / "hardware.json", hardware)
    write_json(output / "source.json", implementation)
    rows, fingerprints = [], {}
    for name in EXPECTED:
        path = args.data_root / name / "processed" / f"{SCHEMA}.npz"
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        fingerprints[str(path.resolve())] = manifest["sha256"]
        graph = load_graph(path, manifest["sha256"])
        expected = EXPECTED[name]
        if (graph.num_nodes, graph.num_features, graph.num_classes) != (
            expected["nodes"],
            expected["features"],
            expected["classes"],
        ):
            raise ValueError("complete official graph dimensions differ")
        start = time.perf_counter()
        cpu_top = build_topology(graph.num_nodes, graph.edges, workers=workers)
        construction = time.perf_counter() - start
        top, h = cpu_top.to(device), graph.x.to(device)
        chunk, calibration = calibrate(top, h, device, [32, 128, 512], repeats=2)
        a = measure(top, h, feature_chunk=chunk)
        b = measure(top, h, feature_chunk=32 if chunk != 32 else 128)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=1e-10, rtol=1e-10)
        del b
        # Max source stage group is 15. Duplicate full original inputs to verify
        # this packed axis; these are explicitly not distinct model/seed samples.
        packed = h[None].expand(15, -1, -1)
        packed_chunk, packed_calibration = calibrate(top, packed, device, [8, 32], repeats=2)
        packed_fields = measure(top, packed, feature_chunk=packed_chunk)
        for key in a:
            torch.testing.assert_close(
                packed_fields[key],
                a[key][None].expand_as(packed_fields[key]),
                atol=1e-10,
                rtol=1e-10,
            )
        row = {
            "dataset": name,
            "actual_citation_data": True,
            "nodes": graph.num_nodes,
            "physical_edges": graph.edges.shape[1],
            "all_features": graph.num_features,
            "local_nodes": top.num_local_nodes,
            "local_edges": top.num_local_edges,
            "directed_pairs": top.num_pairs,
            "topology_seconds": construction,
            "workers": workers,
            "chosen_chunk": chunk,
            "calibration": calibration,
            "packed_input_copies": 15,
            "packed_chunk": packed_chunk,
            "packed_calibration": packed_calibration,
            "chunk_and_packing_parity": True,
            "global_input_raw_energy": float(a["global_E"]),
            "seconds": time.perf_counter() - start,
            "resources": runtime_resources(device),
        }
        write_json(output / f"{name}.json", row)
        rows.append(row)
        print(
            f"[full input verified] {name} nodes={graph.num_nodes} "
            f"edges={graph.edges.shape[1]} features={graph.num_features} subset=False",
            flush=True,
        )
        del packed, packed_fields, a, top, cpu_top, h, graph
        gc.collect()
        torch.cuda.empty_cache()
    if sources() != implementation or any(
        file_sha256(path) != sha for path, sha in fingerprints.items()
    ):
        raise RuntimeError("component implementation or complete original input changed")
    write_json(
        output / "completion.json",
        {
            "completed": True,
            "scope": "component_validation_not_checkpoint_analysis",
            "actual_citation_data": True,
            "new_training_runs": 0,
            "trained_checkpoints_analyzed": 0,
            "all_nodes_edges_features": True,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
