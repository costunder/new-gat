"""Graph macro metrics and interventions using frozen student weights."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import torch

from .model import apply_weighted_pair, apply_weighted_wedge


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def scalar_values(model, index: int) -> dict:
    return {
        name: float(getattr(model, name)[index].detach().cpu()) if hasattr(model, name) else None
        for name in ("u", "v", "beta")
    }


def metrics_for_batch(cases, batch, pred, c, target, condition, seeds, model, epsilon):
    """Serialize measured tensors once per batch; feature draws are not graph samples."""
    if tuple(case.graph_id for case in cases) != batch.graph_ids:
        raise ValueError("case IDs/order do not match the packed graph batch")
    nodes = sum(case.num_nodes for case in cases)
    paths = sum(case.wedges.shape[1] for case in cases)
    fields = batch.x.shape[1]
    if pred.shape != (len(seeds), nodes, fields):
        raise ValueError("prediction node/seed/realization axes do not match cases")
    if batch.targets[target].shape != (nodes, fields) or batch.teacher_c.shape != (paths, fields):
        raise ValueError("target/teacher axes do not match cases")
    if c is not None and c.shape != (len(seeds), paths, fields):
        raise ValueError("student weight seed/path/realization axes do not match cases")
    prediction = pred.detach().cpu().double().numpy()
    truth = batch.targets[target].detach().cpu().double().numpy()
    weights = c.detach().cpu().double().numpy() if c is not None else None
    teacher = batch.teacher_c.detach().cpu().double().numpy()
    if not np.isfinite(prediction).all() or (
        weights is not None and not np.isfinite(weights).all()
    ):
        raise ArithmeticError("nonfinite student prediction or path weight")
    rows, patterns = [], {}
    n_start = p_start = 0
    scalars = [scalar_values(model, index) for index in range(len(seeds))]
    for case in cases:
        n, paths = case.num_nodes, case.wedges.shape[1]
        y = truth[n_start : n_start + n]
        estimate = prediction[:, n_start : n_start + n]
        error = estimate - y[None]
        relative = np.linalg.norm(error, axis=1) / (np.linalg.norm(y, axis=0)[None] + epsilon)
        rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
        current = weights[:, p_start : p_start + paths] if weights is not None else None
        expected = teacher[p_start : p_start + paths]
        if current is not None:
            patterns[case.graph_id] = current
        for index, seed in enumerate(seeds):
            weight_error = correlation = mean_error = None
            reason = "condition has no learned true-path weights"
            if current is not None and paths:
                mean_error = float(np.max(np.abs(current[index].mean(0) - 1)))
                if condition == "learned" and target == "path":
                    reason = "diagnostic only; output loss does not uniquely identify path weights"
                    weight_error = float(
                        np.mean(
                            np.linalg.norm(current[index] - expected, axis=0)
                            / np.linalg.norm(expected, axis=0)
                        )
                    )
                    left = current[index] - current[index].mean(0)
                    right = expected - expected.mean(0)
                    denom = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
                    defined = denom > 1e-12
                    if defined.any():
                        correlation = float(
                            np.mean((left * right).sum(0)[defined] / denom[defined])
                        )
                elif target != "path":
                    reason = "L/L2 target has no ground-truth path weights"
                else:
                    reason = "random-pair weights have different row correspondence"
            elif current is not None:
                reason = "graph has no paths"
            row = {
                "target": target,
                "condition": condition,
                "seed": seed,
                "split": case.split,
                "graph_id": case.graph_id,
                "family": case.family,
                "num_nodes": n,
                "num_edges": case.edges.shape[1],
                "num_paths": paths,
                "num_realizations": y.shape[1],
                "message_relerr": float(relative[index].mean()),
                "message_abs_rmse": float(rmse[index]),
                "weight_relerr": weight_error,
                "weight_corr": _number(correlation) if correlation is not None else None,
                "weight_mean_error": mean_error,
                "weight_diagnostic_reason": reason,
                **scalars[index],
            }
            rows.append(row)
        n_start += n
        p_start += paths
    return rows, patterns


@torch.no_grad()
def evaluate(model, cached, target, condition, seeds, epsilon):
    model.eval()
    rows, patterns = [], {}
    for cases, batch in cached:
        prediction, c = model(batch)
        measured, values = metrics_for_batch(
            cases, batch, prediction, c, target, condition, seeds, model, epsilon
        )
        rows.extend(measured)
        patterns.update(values)
    return rows, patterns


def seed_macro(rows, seeds):
    return np.asarray(
        [np.mean([row["message_relerr"] for row in rows if row["seed"] == seed]) for seed in seeds],
        dtype=np.float64,
    )


def intervention_manifest(cases, master_seed):
    """Use only same-split, same-size donors; map unequal path counts explicitly."""
    by_group = defaultdict(list)
    for case in cases:
        by_group[(case.split, case.num_nodes)].append(case)
    result = {}
    for members in by_group.values():
        members.sort(key=lambda case: case.graph_id)
        if len(members) < 2:
            raise ValueError("other_graph_pattern requires another graph in the same split/size")
        for index, case in enumerate(members):
            donor = members[(index + 1) % len(members)]
            if case.wedges.shape[1] and not donor.wedges.shape[1]:
                candidates = [
                    other
                    for other in members
                    if other.graph_id != case.graph_id and other.wedges.shape[1]
                ]
                if not candidates:
                    raise ValueError("no nonempty donor pattern in the same split/size")
                donor = candidates[0]
            digest = hashlib.sha256(f"{master_seed}:shuffle:{case.graph_id}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little")
            permutation = np.random.default_rng(seed).permutation(case.wedges.shape[1])
            result[case.graph_id] = {
                "donor_graph_id": donor.graph_id,
                "donor_split": donor.split,
                "donor_num_paths": donor.wedges.shape[1],
                "recipient_num_paths": case.wedges.shape[1],
                "shuffle_seed": seed,
                "weight_permutation": permutation.tolist(),
                "donor_mapping": "linear path-order interpolation, then graph mean normalization",
                "correspondence_operator": "saved random distinct-edge pairs, row norm sqrt(6)",
            }
    return result


def _means(c, groups, graph_count):
    sums = c.new_zeros((c.shape[0], graph_count, c.shape[2]))
    sums.index_add_(1, groups, c)
    counts = torch.bincount(groups, minlength=graph_count).to(c.dtype).clamp_min(1)
    return sums / counts[None, :, None]


@torch.no_grad()
def interventions(model, cached, patterns, plan, target, seeds, epsilon):
    """Freeze C and beta; perform all GPU message operations across whole batches."""
    all_cases = [case for cases, _ in cached for case in cases]
    offsets, total = {}, 0
    for case in all_cases:
        offsets[case.graph_id] = total
        total += case.wedges.shape[1]
    device = cached[0][1].x.device
    dtype = cached[0][1].x.dtype
    global_c = torch.as_tensor(
        np.concatenate([patterns[case.graph_id] for case in all_cases], axis=1),
        device=device,
        dtype=dtype,
    )
    rows = []
    for cases, batch in cached:
        original, shuffled, left, right, fraction = [], [], [], [], []
        for case in cases:
            paths = case.wedges.shape[1]
            original.append(np.arange(paths) + offsets[case.graph_id])
            entry = plan[case.graph_id]
            shuffled.append(
                np.asarray(entry["weight_permutation"], dtype=np.int64) + offsets[case.graph_id]
            )
            donor_paths = entry["donor_num_paths"]
            if paths and donor_paths == 0:
                raise ValueError("cannot transfer a donor pattern with no paths")
            positions = np.linspace(0, max(0, donor_paths - 1), paths)
            low = positions.astype(np.int64)
            high = np.minimum(low + 1, max(0, donor_paths - 1))
            left.append(low + offsets[entry["donor_graph_id"]])
            right.append(high + offsets[entry["donor_graph_id"]])
            fraction.append(positions - low)
        indices = [
            torch.as_tensor(np.concatenate(value), device=device, dtype=torch.long)
            for value in (original, shuffled, left, right)
        ]
        alpha = torch.as_tensor(np.concatenate(fraction), device=device, dtype=dtype)[None, :, None]
        current = global_c.index_select(1, indices[0])
        donor = global_c.index_select(1, indices[2]) * (1 - alpha)
        donor += global_c.index_select(1, indices[3]) * alpha
        donor /= _means(donor, batch.path_graph, batch.num_graphs).index_select(1, batch.path_graph)
        variants = {
            "identity": torch.ones_like(current),
            "mean": _means(current, batch.path_graph, batch.num_graphs).index_select(
                1, batch.path_graph
            ),
            "weight_shuffle": global_c.index_select(1, indices[1]),
            "other_graph_pattern": donor,
            "correspondence_randomization": current,
        }
        for name, weights in variants.items():
            if name == "correspondence_randomization":
                prediction = apply_weighted_pair(
                    batch.edges, batch.pair_edges, batch.pair_coefficients, batch.x, weights
                )
            else:
                prediction = apply_weighted_wedge(batch.wedges, batch.x, weights)
            prediction *= model.beta[:, None, None]
            measured, _ = metrics_for_batch(
                cases, batch, prediction, weights, target, "learned", seeds, model, epsilon
            )
            for row in measured:
                row["intervention"] = name
                if name == "correspondence_randomization":
                    row["weight_relerr"] = row["weight_corr"] = None
                    row["weight_diagnostic_reason"] = "different operator row correspondence"
            rows.extend(measured)
    return rows
