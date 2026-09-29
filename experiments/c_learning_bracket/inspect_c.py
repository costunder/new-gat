"""Observe the actual task path, then replay each identical layer input."""

import torch

from .diffusion import row_coefficients


def coefficients(c, graph, *, edge_chunk_size):
    tail, head, _ = row_coefficients(
        c,
        graph.incidence_edge_index,
        graph.x.shape[0],
        correction=getattr(graph, "sampling_correction", None),
        edge_chunk_size=edge_chunk_size,
    )
    return torch.cat((tail, head))


def head_summary(value):
    value = value.detach().double().cpu()
    if value.ndim == 1:
        value = value[:, None]
    if value.shape[0] == 0:
        return {"count": 0, "mean": None, "std": None, "max_abs": None, "rms": None}
    return {
        "count": value.shape[0],
        "mean": value.mean(0).tolist(),
        "std": value.std(0, unbiased=False).tolist(),
        "max_abs": value.abs().amax(0).tolist(),
        "rms": value.square().mean(0).sqrt().tolist(),
    }


def compare(before, after):
    return {
        "before": head_summary(before),
        "after": head_summary(after),
        "change": head_summary(after - before),
    }


def generator_parameters(model):
    return {name: p for name, p in model.named_parameters() if ".estimator." in name}


def signal_summary(value, heads):
    """Actual forward values, all nodes/channels; CPU FP64 diagnostic reduction."""
    value = value.detach().cpu().double().reshape(value.shape[0], heads, -1)
    squares = value.square().mean((0, 2))
    return {
        "rms": float(squares.mean().sqrt()),
        "rms_by_channel_group": squares.sqrt().tolist(),
        "nodes": value.shape[0],
        "channels": value.shape[1] * value.shape[2],
    }


def feature_summary(value, groups, count, heads, edges, chunk):
    """Per-head RMS and within-context node spread, for the observation batch only."""
    value = value.detach().double().cpu().reshape(value.shape[0], heads, -1)
    groups, edges = groups.detach().cpu(), edges.detach().cpu()
    counts = torch.bincount(groups, minlength=count).clamp_min(1)
    mean = value.new_zeros((count, heads, value.shape[-1]))
    mean.index_add_(0, groups, value)
    mean /= counts[:, None, None]
    centered = value - mean[groups]
    edge_square = value.new_zeros(heads)
    for start in range(0, edges.shape[1], chunk):
        tail, head = edges[:, start : start + chunk]
        edge_square += (value[tail] - value[head]).square().sum((0, 2))
    return {
        "rms_by_head": value.square().mean((0, 2)).sqrt().tolist(),
        "within_context_centered_rms_by_head": centered.square().mean((0, 2)).sqrt().tolist(),
        "edge_difference_rms_by_head": None
        if edges.shape[1] == 0
        else (edge_square / (edges.shape[1] * value.shape[-1])).sqrt().tolist(),
        "nodes": value.shape[0],
        "edges": edges.shape[1],
    }


def canonical_alpha(c, edges, correction, nodes):
    """CPU FP64, fixed reduction order: diagnostic reference, not model propagation."""
    c, edges = c.detach().double().cpu(), edges.detach().cpu()
    if correction is not None:
        c = c * correction.detach().double().cpu()[:, None]
    degree = c.new_zeros((nodes, c.shape[1]))
    degree.index_add_(0, edges[0], c)
    degree.index_add_(0, edges[1], c)
    safe = torch.where(degree > 0, degree, 1)
    return torch.cat((c / safe[edges[0]], c / safe[edges[1]]))


def resolved_change(raw_changed, actual, noise, canonical):
    reference_floor = torch.maximum(
        noise, noise.new_full(noise.shape, 64 * torch.finfo(torch.float64).eps)
    )
    return raw_changed & (actual > noise) & (canonical > reference_floor)


class UpdateInspection:
    """CPU snapshots only on a declared observation batch, never a new loss.

    The live conductance hook measures d(CE)/dC before clipping. No detached
    last_c tensor is mistaken for the tensor used by propagation.
    """

    def __init__(self, model, example_nodes, optimizer=None):
        if example_nodes < 1:
            raise ValueError("declare a positive number of displayed receiver nodes")
        self.example_nodes = example_nodes
        self.parameters_before = {
            n: p.detach().cpu().clone() for n, p in generator_parameters(model).items()
        }
        self.layers = []
        self.gradients = {}
        self.optimizer_groups = (
            {}
            if optimizer is None
            else {
                id(p): {"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])}
                for group in optimizer.param_groups
                for p in group["params"]
            }
        )

    def __call__(self, index, op, state, c, message, graph, groups, count, kwargs):
        row = {
            "index": index,
            "input": state.detach().cpu().clone(),
            "c": c.detach().cpu().clone(),
            "score": op.estimator.last_scores.detach().cpu().clone(),
            "alpha": torch.cat(op.observed_coefficients).cpu(),
            "groups": groups,
            "count": count,
            "kwargs": kwargs,
            "live_c_gradient": None,
            "signal_stages": {
                label: signal_summary(value, op.heads)
                for label, value in {
                    "input_h": state,
                    **op.observed_signals,
                    "output_projection": message,
                }.items()
            },
        }
        with torch.no_grad(), torch.autocast(device_type=state.device.type, enabled=False):
            replay = op.estimator(state.detach().float(), graph.incidence_edge_index, groups, count)
            replay_alpha = coefficients(replay, graph, edge_chunk_size=op.edge_chunk_size).cpu()
            row["no_update_replay"] = {
                "score": head_summary(op.estimator.last_scores.cpu() - row["score"]),
                "c": head_summary(replay.cpu() - row["c"]),
                "alpha": head_summary(replay_alpha - row["alpha"]),
            }
            features = {
                "input_h": feature_summary(
                    state,
                    groups,
                    count,
                    op.heads,
                    graph.incidence_edge_index,
                    op.edge_chunk_size,
                ),
                "query": None,
                "key": None,
            }
            if hasattr(op.estimator, "query"):
                geometry = state.detach().to(op.estimator.query.weight.dtype)
                for label, projection in (("query", op.estimator.query), ("key", op.estimator.key)):
                    features[label] = feature_summary(
                        projection(geometry),
                        groups,
                        count,
                        op.heads,
                        graph.incidence_edge_index,
                        op.edge_chunk_size,
                    )
            row["features"] = features
        if c.requires_grad:

            def record_gradient(gradient):
                # Compact device tensors; transfer after backward, not in its hot path.
                row["live_c_gradient"] = torch.stack(
                    (gradient.detach().square().sum(0).sqrt(), gradient.detach().abs().amax(0))
                )

            if c.shape[0]:
                c.register_hook(record_gradient)
        self.layers.append(row)

    def record_activation(self, index, activated, state, heads):
        row = self.layers[index]
        row["signal_stages"]["relu"] = signal_summary(activated, heads)
        row["signal_stages"]["dropout_next_h"] = signal_summary(state, heads)

    def record_parameter_gradients(self, model, label):
        self.gradients[label] = {
            name: None
            if p.grad is None
            else {
                "norm": float(p.grad.detach().norm()),
                "max_abs": float(p.grad.detach().abs().max()),
            }
            for name, p in generator_parameters(model).items()
        }

    @torch.no_grad()
    def finish(self, model, graph):
        edges = graph.incidence_edge_index.detach().cpu()
        receivers, senders = torch.cat((edges[0], edges[1])), torch.cat((edges[1], edges[0]))
        degree = torch.bincount(receivers, minlength=graph.x.shape[0])
        # Predetermined rule: lowest occurrence IDs with at least two neighbors.
        selected = (degree >= 2).nonzero().flatten()[: self.example_nodes]
        shown = torch.isin(receivers, selected).nonzero().flatten()
        original = getattr(graph, "global_node_id", None)
        original = torch.arange(graph.x.shape[0]) if original is None else original.detach().cpu()
        correction = getattr(graph, "sampling_correction", None)
        correction = torch.ones(edges.shape[1]) if correction is None else correction.detach().cpu()
        groups = self.layers[0]["groups"].detach().cpu() if self.layers else None
        results = []
        for row in self.layers:
            op = model.layers[row["index"]]
            state = row["input"].to(graph.x.device).float()
            with torch.autocast(device_type=state.device.type, enabled=False):
                after = op.estimator(state, graph.incidence_edge_index, row["groups"], row["count"])
                after_alpha = coefficients(after, graph, edge_chunk_size=op.edge_chunk_size).cpu()
                repeat_alpha = coefficients(after, graph, edge_chunk_size=op.edge_chunk_size).cpu()
                uniform = coefficients(
                    torch.ones_like(after), graph, edge_chunk_size=op.edge_chunk_size
                ).cpu()
            after_c, after_score = after.cpu(), op.estimator.last_scores.cpu()
            raw_changed = (after_c != row["c"]).any(0)
            canonical_before = canonical_alpha(row["c"], edges, correction, graph.x.shape[0])
            canonical_after = canonical_alpha(after_c, edges, correction, graph.x.shape[0])
            canonical_change = head_summary(canonical_after - canonical_before)
            after_noise = head_summary(repeat_alpha - after_alpha)
            alpha_change = head_summary(after_alpha - row["alpha"])
            noise = torch.tensor(row["no_update_replay"]["alpha"]["max_abs"] or [0.0] * op.heads)
            noise = torch.maximum(noise, torch.tensor(after_noise["max_abs"] or [0.0] * op.heads))
            actual = torch.tensor(alpha_change["max_abs"] or [0.0] * op.heads)
            canonical_max = torch.tensor(canonical_change["max_abs"] or [0.0] * op.heads)
            # An observed replay difference is a noise sample, not a rigorous upper bound.
            # Require a raw-C change and a deterministic reference change as well.
            resolved = resolved_change(raw_changed, actual, noise, canonical_max)
            table = []
            for position in shown.tolist():
                edge = position % edges.shape[1]
                receiver, sender = int(receivers[position]), int(senders[position])
                table.append(
                    {
                        "receiver_occurrence": receiver,
                        "sender_occurrence": sender,
                        "receiver_original": int(original[receiver]),
                        "sender_original": int(original[sender]),
                        "context": int(groups[receiver]),
                        "physical_edge": edge,
                        "sampling_correction": float(correction[edge]),
                        "score_before": row["score"][edge].tolist(),
                        "score_after": after_score[edge].tolist(),
                        "c_before": row["c"][edge].tolist(),
                        "c_after": after_c[edge].tolist(),
                        "alpha_before": row["alpha"][position].tolist(),
                        "alpha_after": after_alpha[position].tolist(),
                    }
                )
            gradient = row["live_c_gradient"]
            results.append(
                {
                    "layer": row["index"],
                    "score": compare(row["score"], after_score),
                    "c": compare(row["c"], after_c),
                    "alpha": compare(row["alpha"], after_alpha),
                    "features_before_update": row["features"],
                    "signal_stages_before_update": row["signal_stages"],
                    "coefficient_edge_chunk_size": op.edge_chunk_size,
                    "alpha_before_source": "coefficients actually used by forward",
                    "no_update_replay_before": row["no_update_replay"],
                    "no_update_alpha_repeat_after": after_noise,
                    "canonical_cpu_float64_alpha_change": canonical_change,
                    "raw_c_changed_by_head": raw_changed.tolist(),
                    "alpha_change_resolved_above_observed_replay_by_head": resolved.tolist(),
                    "interpretation": (
                        "numerical distinguishability only, not useful learning or significance"
                    ),
                    "alpha_minus_c_ones": head_summary(after_alpha - uniform),
                    "live_c_gradient_norm_and_max_by_head": None
                    if gradient is None
                    else gradient.cpu().tolist(),
                    "neighbors": table,
                }
            )
        updates = {}
        for name, p in generator_parameters(model).items():
            before, after = self.parameters_before[name], p.detach().cpu()
            updates[name] = {
                "before_norm": float(before.norm()),
                "update_norm": float((after - before).norm()),
            }
            settings = self.optimizer_groups.get(id(p))
            if settings is not None:
                ideal_decay = -settings["lr"] * settings["weight_decay"] * before.double()
                updates[name].update(
                    {
                        **settings,
                        "ideal_adamw_decay_update_norm": float(ideal_decay.norm()),
                        "actual_update_minus_ideal_decay_norm": float(
                            (after.double() - before.double() - ideal_decay).norm()
                        ),
                        "decomposition_scope": (
                            "residual includes Adam moments and floating-point rounding; "
                            "not current CE gradient alone"
                        ),
                    }
                )
        result = {
            "scope": "first training batch of epoch; all its edges/layers/heads summarized",
            "input_policy": "layer input, graph and correction frozen across optimizer update",
            "display_rule": "lowest receiver IDs with degree >= 2; all incident neighbors",
            "example_receiver_count": self.example_nodes,
            "layers": results,
            "generator_gradients": self.gradients,
            "generator_updates": updates,
            "fixed_condition_has_no_generator_parameters": model.condition == "fixed",
            "observation_revision": (
                "signal_audit_2: actual coefficients, replay, FP64 reference, intra-layer RMS"
            ),
            "signal_stage_scope": (
                "actual forward before optimizer; all nodes/channels; CPU FP64 RMS; "
                "channel groups after output projection are not preserved attention heads"
            ),
        }
        self.layers.clear()
        return result
