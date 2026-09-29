"""Observe the actual task path, then replay each identical layer input."""

import torch

from research.conductance_gat.v5.model import graph_context_features
from research.conductance_gat.v5.operator import conductance_propagation_coefficients


def coefficients(c, graph):
    tail, head, _ = conductance_propagation_coefficients(
        c,
        graph.incidence_edge_index,
        graph.x.shape[0],
        sampling_correction=getattr(graph, "sampling_correction", None),
        normalization="row",
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


class UpdateInspection:
    """CPU snapshots only on a declared observation batch, never a new loss.

    The live conductance hook measures d(CE)/dC before clipping. No detached
    last_c tensor is mistaken for the tensor used by propagation.
    """

    def __init__(self, model, example_nodes):
        if example_nodes < 1:
            raise ValueError("declare a positive number of displayed receiver nodes")
        self.example_nodes = example_nodes
        self.parameters_before = {
            n: p.detach().cpu().clone() for n, p in generator_parameters(model).items()
        }
        self.layers = []
        self.gradients = {}

    def __call__(self, index, op, state, c, message, graph, groups, count, kwargs):
        row = {
            "index": index,
            "input": state.detach().cpu().clone(),
            "c": c.detach().cpu().clone(),
            "cost": op.estimator.last_scores.detach().cpu().clone(),
            "alpha": coefficients(c.detach(), graph).cpu(),
            "groups": groups,
            "count": count,
            "kwargs": kwargs,
            "live_c_gradient": None,
        }
        if c.requires_grad:

            def record_gradient(gradient):
                # Compact device tensors; transfer after backward, not in its hot path.
                row["live_c_gradient"] = torch.stack(
                    (gradient.detach().square().sum(0).sqrt(), gradient.detach().abs().amax(0))
                )

            if c.shape[0]:
                c.register_hook(record_gradient)
        self.layers.append(row)

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
            kwargs = row["kwargs"]
            with torch.autocast(device_type=state.device.type, enabled=False):
                context, sample_degree, full_degree = graph_context_features(
                    state,
                    graph.incidence_edge_index,
                    row["groups"],
                    row["count"],
                    kwargs["full_degree"],
                    kwargs["graph_structure"],
                    static_context=kwargs["static_context"],
                )
                after = op.estimator(
                    state,
                    graph.incidence_edge_index,
                    row["groups"],
                    row["count"],
                    graph_context=context,
                    sample_degree=sample_degree,
                    full_degree=full_degree,
                    edge_normalization_weight=kwargs["edge_normalization_weight"],
                )
                after_alpha = coefficients(after, graph).cpu()
                uniform = coefficients(torch.ones_like(after), graph).cpu()
            after_c, after_cost = after.cpu(), op.estimator.last_scores.cpu()
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
                        "cost_before": row["cost"][edge].tolist(),
                        "cost_after": after_cost[edge].tolist(),
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
                    "cost": compare(row["cost"], after_cost),
                    "c": compare(row["c"], after_c),
                    "alpha": compare(row["alpha"], after_alpha),
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
        result = {
            "scope": "first training batch of epoch; all its edges/layers/heads summarized",
            "input_policy": "layer input, graph and correction frozen across optimizer update",
            "display_rule": "lowest receiver IDs with degree >= 2; all incident neighbors",
            "example_receiver_count": self.example_nodes,
            "layers": results,
            "generator_gradients": self.gradients,
            "generator_updates": updates,
            "fixed_condition_has_no_generator_parameters": model.condition == "fixed",
        }
        self.layers.clear()
        return result
