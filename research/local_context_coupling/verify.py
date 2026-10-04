"""Small, explicitly DEBUG reference checks before a complete fixed audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..local_energy_relations.topology import build_topology
from .model import PackedClassifier
from .operators import prepare_geometry, sandwich


def _dense(n, pairs, mode, device):
    """Independent construction from physical neighbors, not sparse coefficients."""
    neighbors = [set() for _ in range(n)]
    for i, j in pairs:
        neighbors[i].add(j)
        neighbors[j].add(i)
    maps, original = [], []
    for v in range(n):
        nodes = sorted({v} | neighbors[v])
        maps.append({i: len(original) + k for k, i in enumerate(nodes)})
        original.extend(nodes)
    r = F.one_hot(torch.tensor(original, device=device), num_classes=n).double()
    m = r.T / r.sum(0)[:, None]
    a = torch.zeros((len(original), len(original)), dtype=torch.float64, device=device)
    for mapping in maps:
        selected = [(i, j) for i, j in pairs if i in mapping and j in mapping]
        degree = {i: sum(i in edge for edge in selected) for i in mapping}
        for i, j in selected:
            c = 1.0 if mode == "unit" else 2.0 / (degree[i] + degree[j])
            p, q = mapping[i], mapping[j]
            a[p, p] += c
            a[q, q] += c
            a[p, q] -= c
            a[q, p] -= c
    k = torch.zeros_like(a)
    for v, u in pairs:
        for i in set(maps[v]) & set(maps[u]):
            p, q = maps[v][i], maps[u][i]
            k[p, p] += 1
            k[q, q] += 1
            k[p, q] -= 1
            k[q, p] -= 1
    return r, m, a, k


def _classifier_check(device):
    """Real forward -> CE -> backward -> optimizer on DEBUG labels only."""
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]
    geometry = prepare_geometry(build_topology(5, np.asarray(pairs, dtype=np.int64).T), "unit")
    geometry = geometry.to(device, torch.float64)
    x = torch.sin(torch.arange(30, dtype=torch.float64, device=device)).reshape(5, 6)
    labels = torch.tensor([0, 1, 0, 2, 1], dtype=torch.long, device=device)
    model = PackedClassifier(6, 3, seeds=(11, 23, 37), condition="cross_on", mode="unit")
    model = model.to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    cross = {name: p for name, p in model.named_parameters() if "cross" in name}
    if not cross:
        raise AssertionError("cross_on has no registered trainable cross parameter")
    before = {name: p.detach().clone() for name, p in cross.items()}
    logits, _ = model((x, geometry), epoch=0, diagnostics=True)
    loss = F.cross_entropy(logits.reshape(-1, 3), labels.repeat(3))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {}
    for name, p in cross.items():
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0:
            raise AssertionError(f"no finite nonzero CE gradient: {name}")
        gradients[name] = float(p.grad.abs().max().detach().cpu())
    optimizer.step()
    changes = {name: float((p.detach() - before[name]).abs().max().cpu()) for name, p in cross.items()}
    if any(value == 0 for value in changes.values()):
        raise AssertionError("cross parameter was not updated by its optimizer")
    return {"scope": "DEBUG fixture and labels; one update, no dataset training",
            "layers": 2, "hidden": 64, "independent_seeds": 3,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "ce": float(loss.detach().cpu()), "cross_gradient_max_abs": gradients,
            "cross_parameter_update_max_abs": changes, "optimizer_updates": 1}


def run_checks(device="cpu"):
    """Fixtures remain separate from every scientific input and result."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; DEBUG verification has no CPU fallback")
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
        eta, gamma = 0.1, 0.35
        for name, n, pairs in fixtures:
            edges = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T.copy()
            top = build_topology(n, edges)
            for mode in ("unit", "local_degree"):
                geo = prepare_geometry(top, mode).to(device, torch.float64)
                r, m, a, k = _dense(n, pairs, mode, device)
                eye = torch.eye(a.shape[0], dtype=torch.float64, device=device)
                h = torch.sin(torch.arange(n * 4, dtype=torch.float64, device=device)).reshape(n, 4)
                p = eye - eta * a
                on = sandwich(geo, h, eta=eta, gamma=gamma)
                off = sandwich(geo, h, cross=False, eta=eta, gamma=gamma)
                expected = m @ p @ (eye - gamma * k) @ p @ r @ h
                expected_off = m @ p @ p @ r @ h
                difference = -eta ** 2 * gamma * (m @ a @ k @ a @ r @ h)
                values = {
                    "sparse_dense_on_error": (on - expected).abs().max(),
                    "sparse_dense_off_error": (off - expected_off).abs().max(),
                    "difference_identity_error": (on - off - difference).abs().max(),
                    "initial_copy_null_error": (k @ r).abs().max(),
                    "mean_cross_null_error": (m @ k).abs().max(),
                    "immediate_merge_error": (m @ (eye - gamma * k) @ p @ r @ h - m @ p @ r @ h).abs().max(),
                }
                numbers = {key: float(value.detach().cpu()) for key, value in values.items()}
                if any(not np.isfinite(value) or value > 1e-11 for value in numbers.values()):
                    raise AssertionError(f"DEBUG reference failure {name}/{mode}: {numbers}")
                context, effect = torch.linalg.vector_norm(k @ p @ r @ h), torch.linalg.vector_norm(on - off)
                if float(context.detach().cpu()) > 1e-10 and float(effect.detach().cpu()) < 1e-12:
                    raise AssertionError("nonzero disagreement lost despite theorem assumptions")
                records.append({"fixture": name, "weight_mode": mode, **numbers,
                                "context_disagreement_norm": float(context.detach().cpu()),
                                "matched_output_difference_norm": float(effect.detach().cpu())})
        classifier = _classifier_check(device)
    finally:
        torch.set_num_threads(previous)
    return {"status": "passed", "scope": "DEBUG only", "device": str(device),
            "precision": "float64", "algebra_cases": records, "classifier_connection": classifier}


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
