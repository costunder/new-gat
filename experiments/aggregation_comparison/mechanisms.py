"""GPU reductions over every validation graph; no sampled diagnostic subset."""

from __future__ import annotations

import math
import time

import torch

from .model import conductance_contract
from .validation import require_reproduction


class MechanismCollector:
    """Retain only small per-layer/head accumulators, never node/edge histories."""

    bins = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]

    def __init__(self):
        self.layers = {}

    def _moments(self, row, key, values):
        # H x observations, pooled by observations rather than averaging graphs.
        values = values.detach().double()
        count = values.shape[1]
        if not count:
            return
        current = {
            "count": count,
            "sum": values.sum(1),
            "squared": values.square().sum(1),
            "absolute": values.abs().sum(1),
            "min": values.amin(1),
            "max": values.amax(1),
        }
        if key not in row:
            row[key] = current
        else:
            previous = row[key]
            previous["count"] += count
            for name in ("sum", "squared", "absolute"):
                previous[name] += current[name]
            previous["min"] = torch.minimum(previous["min"], current["min"])
            previous["max"] = torch.maximum(previous["max"], current["max"])

    def record(self, layer, c, gram, diagonal, message, branch):
        if torch.is_grad_enabled():
            raise RuntimeError("mechanism collection must not retain autograd state")
        heads = gram.shape[1]
        weights = c[:, None] if c.ndim == 1 else c
        weights = weights.expand(-1, heads).T
        row = self.layers.setdefault(layer, {"heads": heads, "batches": 0})
        row["batches"] += 1
        self._moments(row, "conductance", weights)
        for name, mask in (("diagonal", diagonal), ("cross", ~diagonal)):
            self._moments(row, name, gram[..., mask].permute(1, 0, 2).flatten(1))
        boundaries = weights.new_tensor(self.bins)
        bucket = torch.bucketize(weights.contiguous(), boundaries, right=True)
        bucket += torch.arange(heads, device=weights.device)[:, None] * (len(self.bins) + 1)
        hist = torch.bincount(bucket.flatten(), minlength=heads * (len(self.bins) + 1)).reshape(
            heads, -1
        )
        row["c_histogram"] = row.get("c_histogram", torch.zeros_like(hist)) + hist
        for key, tensor in (("message_squared", message), ("branch_squared", branch)):
            total = tensor.detach().double().square().sum()
            row[key] = row.get(key, torch.zeros_like(total)) + total

    def finish(self):
        rows = []
        for index, row in sorted(self.layers.items()):
            result = {"layer": index, "heads": row["heads"], "validation_batches": row["batches"]}
            for key in ("conductance", "diagonal", "cross"):
                stats = row.get(key)
                if stats is None:
                    result[key] = {"observations_per_head": 0, "reason": "no such edges or pairs"}
                    continue
                mean = stats["sum"] / stats["count"]
                rms = (stats["squared"] / stats["count"]).sqrt()
                std = (stats["squared"] / stats["count"] - mean.square()).clamp_min(0).sqrt()
                combined = torch.stack(
                    (mean, rms, std, stats["absolute"] / stats["count"], stats["min"], stats["max"])
                )
                values = (
                    combined.cpu().tolist()
                )  # Only reduced H-vectors, once after full validation.
                if not all(math.isfinite(v) for vector in values for v in vector):
                    raise ValueError("nonfinite mechanism statistics")
                result[key] = dict(
                    zip(("mean", "rms", "std", "mean_abs", "min", "max"), values, strict=True)
                )
                result[key]["observations_per_head"] = stats["count"]
            base, energy = (
                torch.stack((row["message_squared"], row["branch_squared"])).cpu().tolist()
            )
            if not math.isfinite(base) or not math.isfinite(energy):
                raise ValueError("nonfinite branch magnitude")
            result.update(
                message_l2=math.sqrt(base),
                energy_output_l2=math.sqrt(energy),
                energy_to_message_l2=(math.sqrt(energy / base) if base else None),
                zero_message_norm=(base == 0),
                c_histogram=row["c_histogram"].cpu().tolist(),
                c_histogram_upper_boundaries=self.bins,
                c_histogram_right_inclusive=False,
            )
            rows.append(result)
        return {
            "scope": "all nodes/edges of full validation input graphs, including transductive "
            "context nodes outside the validation-label mask; "
            "observation-weighted layer/head statistics",
            "gram_coordinates": "pre-lift linear value projection",
            "rows": rows,
        }


def mechanism_audit(model, inputs, args, device, selected, evaluate):
    c_config = conductance_contract(model.arm)
    if c_config is None:
        return {"applicable": False, "reason": "external comparator has no incidence energy branch"}
    model.eval()
    if torch.is_grad_enabled():
        raise RuntimeError("mechanism audit requires no_grad")
    collector = MechanismCollector()
    started = time.perf_counter()
    prior = model.diagnostic_collector
    try:
        model.diagnostic_collector = collector
        baseline = evaluate(model, inputs, args, device)
    finally:
        model.diagnostic_collector = prior
    require_reproduction(selected, baseline, label="mechanism-capture baseline")
    diagnostics = collector.finish()
    interventions = []
    if len(model.energy_readouts):
        interventions.append("energy_off")
        if not model.diagonal_only:
            interventions.extend(("cross_off", "diagonal_off"))
    if c_config["regime"] != "fixed":
        interventions.extend(("c_ones", "c_mean", "c_shuffle"))
    if any(op.lift_projection is not None for op in model.layers):
        interventions.append("lift_second_off")
    reports = {}
    for name in interventions:
        with model.intervention(name):
            result = evaluate(model, inputs, args, device)
        reports[name] = {
            "validation": result,
            "delta_pp": 100 * (result["metric"] - baseline["metric"]),
        }
    # Verify restoration through a real forward, not just state_dict equality.
    restored = evaluate(model, inputs, args, device)
    require_reproduction(selected, restored, label="post-intervention restoration")
    return {
        "applicable": True,
        "baseline": baseline,
        "diagnostics": diagnostics,
        "interventions": reports,
        "restored_validation": restored,
        "seconds": time.perf_counter() - started,
        "interpretation": "whole-network inference interventions with downstream recomputation; "
        "not retrained ablations, inverse reconstruction, or fixed-H local effects",
        "c_shuffle_semantics": "deterministic edge-order reversal within each graph; "
        "the complete head vector moves together; no cross-graph permutation",
    }
