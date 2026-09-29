"""Explicit real-data input equivalence check, not training or a model-quality result."""

import copy
import time

import torch

from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.study_inputs import StudyInputs as PreviousInputs

from .inputs import BracketInputs
from .model import make_model
from .train import parser, validate, write_json


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    engine.base.configure_compute(args)
    payload, protocol = engine.load_dataset(args)
    output = args.output_dir
    if output.exists():
        raise FileExistsError("use a new input-equivalence output directory")
    output.mkdir(parents=True)
    # This CPU comparison isolates graph preparation, without pinning/transfer.
    # Production pinning is measured in calibration and CUDA regression tests.
    selected = copy.deepcopy(args)
    selected.pin_memory, selected.sample_prefetch = False, False
    started = time.perf_counter()
    previous = PreviousInputs(payload, selected)
    previous_init = time.perf_counter() - started
    started = time.perf_counter()
    current = BracketInputs(payload, selected)
    current_init = time.perf_counter() - started
    old_batches = previous.training_batches(0, torch.device("cpu"))
    new_batches = current.training_batches(0, torch.device("cpu"))
    measurements = []
    fields = (
        "x",
        "y",
        "incidence_edge_index",
        "full_degree",
        "graph_structure",
        "sampling_correction",
        "batch",
        "global_node_id",
    )
    for index in range(args.calibration_repeats):
        started = time.perf_counter()
        old = next(old_batches)
        old_seconds = time.perf_counter() - started
        started = time.perf_counter()
        new = next(new_batches)
        new_seconds = time.perf_counter() - started
        for name in fields:
            if not torch.equal(getattr(old.graph, name), getattr(new.graph, name)):
                raise AssertionError(f"sample {index}: changed {name}")
        if not torch.equal(old.selected_indices, new.selected_indices):
            raise AssertionError("changed supervised seed selection")
        if old.graph.study_batch_identity != new.graph.study_batch_identity:
            raise AssertionError("changed sampling hashes")
        measurements.append(
            {
                "batch_index": index,
                "previous_cpu_preparation_seconds": old_seconds,
                "current_cpu_preparation_seconds": new_seconds,
                "identity": new.graph.study_batch_identity,
                "all_compared_fields_equal": True,
                "nodes": new.graph.x.shape[0],
                "edges": new.graph.incidence_edge_index.shape[1],
                "supervised_seeds": new.selected_indices.numel(),
            }
        )
    model = make_model(payload, args, torch.device("cuda")).eval()
    with torch.no_grad():
        before = model(old.to("cuda").graph)
        after = model(new.to("cuda").graph)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-7)
        change = float((before - after).abs().max())
    report = {
        "scope": "real-data input equivalence; four batches, no optimizer updates",
        "dataset_protocol": protocol,
        "previous_initialization_seconds": previous_init,
        "current_initialization_seconds": current_init,
        "batches": measurements,
        "forward_max_abs_difference_last_batch": change,
        "model": model.contract(),
        "note": "CPU preparation timing excludes pinning and GPU copy; same full sampling recipe",
    }
    write_json(output / "comparison.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
