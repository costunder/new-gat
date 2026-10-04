"""Independent DEBUG algebra and one-update CE checks before any dataset work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ...local_energy_relations.topology import build_topology
from ..operators import prepare_geometry
from ..verify import _dense as _raw_dense
from .common import condition_metadata
from .model import PackedClassifier
from .operators import prepare_normalized_geometry, sandwich, mixed_action, immediate


def dense_reference(n, pairs, mode, intra, cross, device="cpu"):
    """Construct S/G from physical edges and independent dense A/K references."""
    r, m, a, k = _raw_dense(n, pairs, mode, device)
    degree = a.diagonal()
    if intra == "graph":
        maximum = degree.max()
        eta = .5 / maximum if float(maximum) > 0 else maximum.new_zeros(())
        s = eta * a
    elif intra == "local":
        neighbors = [set() for _ in range(n)]
        for i, j in pairs:
            neighbors[i].add(j); neighbors[j].add(i)
        s = torch.zeros_like(a)
        offset = 0
        for center in range(n):
            end = offset + len(neighbors[center]) + 1
            maximum = degree[offset:end].max()
            eta = .5 / maximum if float(maximum) > 0 else maximum.new_zeros(())
            s[offset:end, offset:end] = eta * a[offset:end, offset:end]
            offset = end
    else:
        raise ValueError("unknown dense intra policy")
    kd = k.diagonal()
    if cross in ("graph", "none"):
        maximum = kd.max()
        gamma = .5 / maximum if float(maximum) > 0 else maximum.new_zeros(())
        g = gamma * k
    elif cross == "edge":
        g = torch.zeros_like(k)
        for left in range(k.shape[0]):
            for right in range(left + 1, k.shape[0]):
                if float(k[left, right]) < 0:
                    weight = .5 / torch.maximum(kd[left], kd[right])
                    g[left, left] += weight; g[right, right] += weight
                    g[left, right] -= weight; g[right, left] -= weight
    else:
        raise ValueError("unknown dense cross policy")
    return r, m, s, g


def _classifier_check(device, mode, intra, cross):
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]
    base = prepare_geometry(build_topology(5, np.asarray(pairs, dtype=np.int64).T), mode)
    geometry = prepare_normalized_geometry(base, intra, cross).to(device, torch.float64)
    condition = f"{mode}__{intra}__{cross}__learned"
    model = PackedClassifier(condition, 6, 3, (11, 23, 37), edge_chunk=3).to(device=device, dtype=torch.float64)
    x = torch.sin(torch.arange(30, dtype=torch.float64, device=device)).reshape(5, 6)
    labels = torch.tensor([0, 1, 0, 2, 1], dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.weight_decay_groups(.0005), lr=.01)
    before = model.theta_cross.detach().clone()
    logits, _ = model((x, geometry), epoch=0)
    loss = F.cross_entropy(logits.reshape(-1, 3), labels.repeat(3))
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) or float(parameter.grad.abs().sum()) <= 0:
            raise AssertionError(f"DEBUG normalized classifier disconnected from CE: {name}")
    gradient = float(model.theta_cross.grad.abs().max().detach().cpu())
    optimizer.step()
    change = float((model.theta_cross.detach() - before).abs().max().cpu())
    if change <= 0:
        raise AssertionError("DEBUG normalized cross parameter was not updated")
    return {"condition": condition, **condition_metadata(condition),
            "scope": "DEBUG fixture labels only; one update, no dataset training",
            "layers": 2, "hidden": 64, "independent_seeds": 3,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "ce": float(loss.detach().cpu()), "cross_gradient_max_abs": gradient,
            "cross_parameter_update_max_abs": change, "optimizer_updates": 1}


def run_checks(device="cpu"):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; normalized DEBUG verification has no CPU fallback")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    records = []
    try:
        fixtures = (
            ("path", 3, [(0, 1), (1, 2)]),
            ("shared_neighbor_private_regions", 5, [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]),
            ("clique", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
            ("components_and_isolate", 6, [(0, 1), (1, 2), (3, 4)]),
            ("empty", 4, []),
        )
        for name, n, pairs in fixtures:
            topology = build_topology(n, np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T.copy())
            for mode in ("unit", "local_degree"):
                base = prepare_geometry(topology, mode)
                for intra in ("graph", "local"):
                    for cross in ("graph", "edge"):
                        geometry = prepare_normalized_geometry(base, intra, cross).to(device, torch.float64)
                        r, m, s, g = dense_reference(n, pairs, mode, intra, cross, device)
                        eye = torch.eye(s.shape[0], dtype=torch.float64, device=device)
                        p = eye - s
                        h = torch.sin(torch.arange(n * 4, dtype=torch.float64, device=device)).reshape(n, 4)
                        on = sandwich(geometry, h, edge_chunk=3)
                        off = sandwich(geometry, h, cross=False, edge_chunk=3)
                        delta = -m @ s @ g @ s @ r @ h
                        values = {
                            "sparse_dense_on_error": (on - m @ p @ (eye - g) @ p @ r @ h).abs().max(),
                            "sparse_dense_off_error": (off - m @ p @ p @ r @ h).abs().max(),
                            "difference_identity_error": (on - off - delta).abs().max(),
                            "mixed_action_error": (mixed_action(geometry, h, 3) + delta).abs().max(),
                            "initial_copy_null_error": (g @ r).abs().max(),
                            "mean_cross_null_error": (m @ g).abs().max(),
                            "immediate_merge_error": (immediate(geometry, h, edge_chunk=3) - m @ p @ r @ h).abs().max(),
                        }
                        numbers = {key: float(value.detach().cpu()) for key, value in values.items()}
                        eigen_s, eigen_g = torch.linalg.eigvalsh(s), torch.linalg.eigvalsh(g)
                        if (any(not np.isfinite(value) or value > 1e-11 for value in numbers.values())
                                or float(eigen_s.min()) < -1e-12 or float(eigen_g.min()) < -1e-12
                                or float(eigen_s.max()) > 1 + 1e-12 or float(eigen_g.max()) > 1 + 1e-12):
                            raise AssertionError(f"DEBUG normalization reference failure {name}/{mode}/{intra}/{cross}")
                        context, effect = torch.linalg.vector_norm(g @ p @ r @ h), torch.linalg.vector_norm(on - off)
                        if float(context) > 1e-10 and float(effect) < 1e-12:
                            raise AssertionError("normalized nonzero context lost despite theorem assumptions")
                        records.append({"fixture": name, "weight_mode": mode, "intra_policy": intra,
                                        "cross_policy": cross, "energy_operator": "applied_S_G", **numbers,
                                        "context_disagreement_norm": float(context), "matched_output_difference_norm": float(effect),
                                        "s_eigenvalue_max": float(eigen_s.max()), "g_eigenvalue_max": float(eigen_g.max())})
        classifier = [_classifier_check(device, mode, intra, cross)
                      for mode in ("unit", "local_degree") for intra in ("graph", "local") for cross in ("graph", "edge")]
    finally:
        torch.set_num_threads(previous)
    return {"status": "passed", "scope": "DEBUG only", "device": str(device),
            "precision": "float64", "algebra_cases": records,
            "classifier_connections": classifier,
            "preflight_debug_optimizer_updates": len(classifier),
            "scientific_dataset_optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks(args.device)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
