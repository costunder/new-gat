"""Validation-only interventions on an existing conductance checkpoint.

This module performs no training, downloads, cache writes, or test-label queries.
The learned reference uses the original Conv, including its original degree cap.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from types import MethodType, SimpleNamespace


def _shuffle_generator(seed, graph_index, layer_index):
    import torch

    key = f"conductance-intervention:{seed}:{graph_index}:{layer_index}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(value)


def _check_inputs(state, edges, node_graph):
    import torch

    if state.ndim != 2 or len(state) == 0 or not torch.isfinite(state).all():
        raise ValueError("Interventions require finite, nonempty node states")
    if (
        node_graph.shape != (len(state),)
        or torch.any(node_graph != 0)
        or edges.ndim != 2
        or edges.shape[0] != 2
        or edges.dtype != torch.long
    ):
        raise ValueError("Interventions evaluate exactly one graph per forward")
    if edges.numel() and (edges.min() < 0 or edges.max() >= len(state)):
        raise ValueError("An intervention edge refers to an invalid node")


def _weighted_degree(state, edges, conductance):
    degree = state.new_zeros(len(state))
    degree.index_add_(0, edges[1], conductance)
    degree.index_add_(0, edges[0], conductance)
    return degree


def _layer_record(state, edges, conductance, output, degree=None):
    import torch

    state, output = state.float(), output.float()
    if (
        conductance.shape != (edges.shape[1],)
        or not torch.isfinite(conductance).all()
        or torch.any(conductance < 0)
        or not torch.isfinite(output).all()
    ):
        raise FloatingPointError("Invalid intervention conductance/output")
    if degree is None:
        degree = _weighted_degree(state, edges, conductance)
    if not torch.isfinite(degree).all():
        raise FloatingPointError("Nonfinite intervention weighted degree")
    rho = 0.95 * degree / degree.max().clamp_min(1e-12)
    c = conductance.detach().double().cpu()
    delta = output.double() - state.double()
    input_squared = float(state.double().square().sum())
    delta_squared = float(delta.square().sum())
    change = delta.norm(dim=1) / state.double().norm(dim=1).clamp_min(1e-12)
    return {
        "_c": c,
        "_rho": rho.cpu(),
        "_degree": degree.cpu(),
        "_node_change": change.cpu(),
        "rho_mean": float(rho.mean()),
        # For graph_off, effective C is zero and the coefficient of variation
        # is undefined, not zero. summarize_layers handles that case below.
        "c_cv": float(c.std(unbiased=False) / c.mean()) if c.numel() and c.mean() else None,
        "c_count": c.numel(),
        "c_sum": float(c.sum()),
        "c_squared_sum": float(c.square().sum()),
        "input_squared_sum": input_squared,
        "delta_squared_sum": delta_squared,
        "global_update_ratio": math.sqrt(delta_squared) / max(math.sqrt(input_squared), 1e-12),
        "zero_input_nodes": int((state.norm(dim=1) == 0).sum()),
    }


def _substituted_forward(operator, state, edges, node_graph, *, mode, chunk_size, generator):
    """Return the actual substituted update and C; never call original Conv twice."""
    import torch

    _check_inputs(state, edges, node_graph)
    with torch.autocast(device_type=state.device.type, enabled=False):
        fp32 = state.float()
        if mode == "graph_off":
            c = fp32.new_zeros(edges.shape[1])
            return state, c, fp32.new_zeros(len(fp32))
        parts = []
        for start in range(0, edges.shape[1], chunk_size):
            tail, head = edges[:, start : start + chunk_size]
            gradient = fp32[head] - fp32[tail]
            c = operator.estimator(gradient, fp32.new_empty((len(tail), 0)))
            if not torch.isfinite(c).all() or torch.any(c <= 0):
                raise FloatingPointError("Nonfinite/nonpositive learned conductance")
            parts.append(c)
        c = torch.cat(parts) if parts else fp32.new_empty(0)
        if c.numel():
            if mode == "mean_C":
                c = c.mean().expand_as(c)
            elif mode == "shuffled_C":
                permutation = torch.randperm(len(c), generator=generator).to(c.device)
                c = c.index_select(0, permutation)
            else:
                raise ValueError(f"Unknown conductance intervention: {mode}")
        elif mode not in {"mean_C", "shuffled_C"}:
            raise ValueError(f"Unknown conductance intervention: {mode}")
        degree = _weighted_degree(fp32, edges, c)
        step = 0.95 / degree.max().clamp_min(1e-12)
        divergence = torch.zeros_like(fp32)
        for start in range(0, edges.shape[1], chunk_size):
            tail, head = edges[:, start : start + chunk_size]
            flux = c[start : start + len(tail), None] * (fp32[head] - fp32[tail])
            divergence.index_add_(0, head, flux)
            divergence.index_add_(0, tail, -flux)
        output = fp32 - step * divergence
    return output.to(state.dtype), c, degree


@contextmanager
def _instrument_operators(model, records, mode, selected_layers, graph_index, seed, chunk_size):
    """Restore exact instance-forward attributes and every installed hook on failure."""
    handles, replacements = [], []
    try:
        for layer_index, operator in enumerate(model.operators):
            pending = {}
            if layer_index in selected_layers:
                had_forward = "forward" in vars(operator)
                original_forward = vars(operator).get("forward")
                replacements.append((operator, had_forward, original_forward))
                generator = _shuffle_generator(seed, graph_index, layer_index)

                def replacement(
                    module,
                    state,
                    edges,
                    node_graph,
                    num_graphs=None,
                    pending=pending,
                    generator=generator,
                ):
                    if num_graphs is not None and num_graphs != 1:
                        raise ValueError("Interventions require one graph per forward")
                    output, c, degree = _substituted_forward(
                        module,
                        state,
                        edges,
                        node_graph,
                        mode=mode,
                        chunk_size=chunk_size,
                        generator=generator,
                    )
                    pending.update(c=c, degree=degree)
                    return output

                operator.forward = MethodType(replacement, operator)
            else:

                def capture_c(_module, _inputs, output, pending=pending):
                    if "c" in pending:
                        raise RuntimeError("Expected one estimator call per original Conv")
                    pending["c"] = output.detach()

                handles.append(operator.estimator.register_forward_hook(capture_c))

            def capture_layer(_module, inputs, output, index=layer_index, pending=pending):
                state, edges, node_graph = inputs[:3]
                _check_inputs(state, edges, node_graph)
                if "c" not in pending:
                    raise RuntimeError("Original Conv did not expose estimator conductance")
                records[index].append(
                    _layer_record(state, edges, pending["c"], output, pending.get("degree"))
                )
                pending.clear()

            handles.append(operator.register_forward_hook(capture_layer))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for operator, had_forward, original_forward in reversed(replacements):
            if had_forward:
                operator.forward = original_forward
            else:
                del operator.forward


def _validation_items(payload):
    """Select validation labels before transfer; no train/test split is accessed."""
    import torch

    selected = payload["splits"]["validation"]
    if payload["dataset"] == "ppi":
        if not len(selected):
            raise ValueError("No validation graphs")
        for graph_index in selected:
            graph_index = int(graph_index)
            raw = payload["graphs"][graph_index]
            yield graph_index, raw, None, raw["y"]
    else:
        if selected.dtype != torch.bool or selected.ndim != 1:
            raise ValueError("Transductive validation split must be a Boolean node mask")
        indices = selected.nonzero(as_tuple=False).flatten()
        if not len(indices):
            raise ValueError("No validation nodes")
        raw = payload["graphs"][0]
        yield 0, raw, indices, raw["y"].index_select(0, indices)


def _summarize_layer(records):
    from scripts.diagnose_conductance import summarize_layers

    # The existing positive-C helper assumes a nonzero mean. Effective C=0
    # is special to the identity intervention, so avoid producing NaN CV.
    if sum(record["c_sum"] for record in records) == 0:
        empty_c_records = [{**record, "_c": record["_c"][:0]} for record in records]
        result = summarize_layers(empty_c_records)
        edge_count = sum(record["c_count"] for record in records)
        result["edge_pooled"]["conductance"] = {
            "count": edge_count,
            "mean": 0.0 if edge_count else None,
            "quantiles": dict.fromkeys(("min", "p10", "median", "p90", "p99", "max"), 0.0)
            if edge_count
            else None,
        }
        return result
    return summarize_layers(records)


def evaluate_interventions(
    model,
    payload,
    device,
    *,
    edge_chunk_size=16384,
    shuffle_seed=0,
    layerwise=True,
    progress=None,
):
    """Evaluate fixed-checkpoint C interventions on validation, never retrain.

    Only one caller-selected model checkpoint is used. ``shuffle_seed`` controls
    edge reassignment, not initialization or a repeated model-seed experiment.
    The return value is JSON-safe. Existing parameters/gradients are untouched;
    module modes, forward attributes, hooks, and PyTorch RNG are restored.
    """
    import torch

    from scripts.diagnose_conductance import merge_predictions, prediction_statistics

    if not isinstance(edge_chunk_size, int) or edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be positive")
    if not isinstance(shuffle_seed, int) or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    if not len(model.operators):
        raise ValueError("Interventions require at least one conductance layer")
    device = torch.device(device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Interventions require an FP32 checkpoint")
    items = list(_validation_items(payload))
    multilabel = payload["dataset"] == "ppi"
    all_layers = list(range(len(model.operators)))
    specifications = [("learned_C", "learned_C", [])]
    for mode in ("mean_C", "shuffled_C", "graph_off"):
        specifications.append((f"{mode}_all", mode, all_layers))
        if layerwise:
            specifications.extend((f"{mode}_layer_{index}", mode, [index]) for index in all_layers)
    module_modes = [(module, module.training) for module in model.modules()]
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    baseline_logits, variants = {}, []
    try:
        model.eval()
        with (
            torch.random.fork_rng(devices=cuda_devices),
            torch.inference_mode(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            for name, mode, selected_layers in specifications:
                if progress is not None:
                    progress(name)
                layers = [[] for _ in model.operators]
                predictions = []
                squared_delta = squared_reference = 0.0
                flipped = node_flipped = prediction_count = node_count = 0
                for graph_index, raw, indices, labels in items:
                    graph = SimpleNamespace(
                        x=raw["x"].to(device),
                        incidence_edge_index=raw["incidence_edge_index"].to(device),
                    )
                    with _instrument_operators(
                        model,
                        layers,
                        mode,
                        selected_layers,
                        graph_index,
                        shuffle_seed,
                        edge_chunk_size,
                    ):
                        logits = model(graph)
                    if indices is not None:
                        logits = logits.index_select(0, indices.to(device))
                    predictions.append(prediction_statistics(logits, labels.to(device), multilabel))
                    current = logits.detach().float().cpu()
                    if mode == "learned_C":
                        baseline_logits[graph_index] = current
                    reference = baseline_logits[graph_index]
                    squared_delta += float((current.double() - reference.double()).square().sum())
                    squared_reference += float(reference.double().square().sum())
                    difference = (
                        (current > 0) != (reference > 0)
                        if multilabel
                        else current.argmax(dim=-1) != reference.argmax(dim=-1)
                    )
                    flipped += int(difference.sum())
                    node_flipped += (
                        int(difference.any(dim=-1).sum()) if multilabel else int(difference.sum())
                    )
                    prediction_count += difference.numel()
                    node_count += len(current)
                prediction = merge_predictions(predictions, multilabel)
                reference_prediction = variants[0]["prediction"] if variants else prediction
                delta = {
                    "loss": prediction["loss"] - reference_prediction["loss"],
                    "metric": prediction["metric"] - reference_prediction["metric"],
                    "logits_relative_l2": math.sqrt(squared_delta)
                    / max(math.sqrt(squared_reference), 1e-12),
                    "prediction_flip_fraction": flipped / prediction_count,
                }
                if multilabel:
                    delta["node_any_label_flip_fraction"] = node_flipped / node_count
                variants.append(
                    {
                        "name": name,
                        "intervention": mode,
                        "selected_layers": selected_layers,
                        "prediction": prediction,
                        "delta_vs_learned": delta,
                        "layers": [_summarize_layer(records) for records in layers],
                    }
                )
    finally:
        for module, training in module_modes:
            module.training = training
    result = {
        "schema_version": 1,
        "split": "validation",
        "shuffle_seed": shuffle_seed,
        "layerwise": layerwise,
        "notes": [
            "One fixed model checkpoint; no training, test-label queries, or model-seed repeats.",
            "Learned/unselected Conv uses its original forward; "
            "C is captured from estimator output.",
            "Selected mean/shuffle C is computed once in FP32 edge chunks; "
            "GEMM/chunk accumulation may change last bits.",
            "Each graph and layer is treated independently; "
            "weighted degree and 0.95/dmax are recomputed after C substitution.",
            "graph_off bypasses selected Conv only; "
            "effective C/degree/rho/update are zero and C-CV is undefined.",
            "Layer statistics cover full transductive graphs or all validation PPI graphs; "
            "metrics/flips use validation labels only.",
            "PPI prediction_flip_fraction is labelwise; "
            "node_any_label_flip_fraction counts nodes with any changed label.",
            "Validation deltas are interventions at this checkpoint, "
            "not causal proof about training or significance tests.",
        ],
        "variants": variants,
    }
    json.dumps(result, allow_nan=False)
    return result
