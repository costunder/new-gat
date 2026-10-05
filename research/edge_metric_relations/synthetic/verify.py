"""Independent dense teacher algebra and actual DEBUG MSE update checks."""
from __future__ import annotations

import math
import torch

from ...local_energy_relations.topology import build_topology
from ..geometry import prepare_geometry
from .data import SyntheticBatch
from .model import RawStudent, normalized_mse, target_message


def dense_diagonal(g):
    b = g.c0.new_zeros((g.num_edges, g.n))
    index = torch.arange(g.num_edges, device=b.device)
    b[index, g.edges[0]] = -1
    b[index, g.edges[1]] = 1
    return b.T @ torch.diag(g.cbar0) @ b


def dense_analytic_teacher(g, x):
    """Construct each scalar field's physical dense metric independently.

    The explicit anchor differences, eps RMS and fixed tanh formula are not
    delegated to the shared pair-feature or teacher implementation.
    """
    b = g.c0.new_zeros((g.num_edges, g.n))
    ids = torch.arange(g.num_edges, device=b.device)
    b[ids, g.edges[0]] = -1; b[ids, g.edges[1]] = 1
    anchor = x.index_select(-2, g.pair_anchor)
    left = x.index_select(-2, g.pair_other_left) - anchor
    right = x.index_select(-2, g.pair_other_right) - anchor
    scale = ((left.square().mean(-1) + right.square().mean(-1)) / 2 + 1e-8).sqrt()
    inner = ((left / scale[..., None]) * (right / scale[..., None])).mean(-1)
    overlap = g.pair_structure[:, 3]
    raw = (2 * inner + .5 * (2 * overlap - 1)).tanh()
    el, er = g.occ_edge[g.pair_left], g.occ_edge[g.pair_right]
    coefficient = .5 * raw * g.pair_sign * g.pair_norm
    coefficient = coefficient * (g.c0[g.pair_left] * g.c0[g.pair_right]).sqrt()
    coefficient = coefficient * g.occ_scale[g.pair_left] * g.occ_scale[g.pair_right]
    metric = torch.diag(g.cbar0).expand(x.shape[:-2] + (g.num_edges, g.num_edges)).clone()
    for pair in range(g.num_pairs):
        metric[..., el[pair], er[pair]] += coefficient[..., pair]
        metric[..., er[pair], el[pair]] += coefficient[..., pair]
    q = b.T @ metric @ b
    return q @ x


def run_math_checks(device="cpu"):
    device = torch.device(device)
    fixtures = {
        "path": (3, [(0, 1), (1, 2)]),
        "triangle": (3, [(0, 1), (0, 2), (1, 2)]),
        "star": (5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
        "irregular": (5, [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]),
        "isolate": (4, [(0, 1)]),
    }
    rows, maximum = [], 0.
    for name, (n, edges) in fixtures.items():
        for recipe in ("unit", "local_degree"):
            top = build_topology(n, torch.tensor(edges, dtype=torch.long).T)
            g = prepare_geometry(top, recipe).to(device, torch.float64)
            generator = torch.Generator().manual_seed(871 + n)
            x = torch.randn((1, 4, n, 1), generator=generator, dtype=torch.float64).to(device)
            ld = dense_diagonal(g)
            expected = {"diagonal": ld @ x, "diagonal_squared": ld @ (ld @ x),
                        "analytic_pair": dense_analytic_teacher(g, x)}
            for target, reference in expected.items():
                actual = target_message(g, x, target, pair_chunk=3)
                error = float((actual - reference).abs().max())
                maximum = max(maximum, error)
                if not torch.allclose(actual, reference, atol=1e-10, rtol=1e-10):
                    raise AssertionError(f"dense raw teacher differs: {name}/{recipe}/{target}")
            rows.append({"fixture": name, "recipe": recipe, "targets": 3,
                         "nodes": n, "eligible_pairs": g.num_pairs, "passed": True})
    n, edges = fixtures["irregular"]
    g = prepare_geometry(build_topology(n, torch.tensor(edges, dtype=torch.long).T), "local_degree").to(device, torch.float64)
    x = torch.tensor([-.8, .5, .9, -1.2, .3], device=device, dtype=torch.float64)[None, None, :, None]
    model = RawStudent("F2", (11, 23)).to(device=device, dtype=torch.float64)
    batch = SyntheticBatch((), g, x)
    target = target_message(g, x, "analytic_pair", pair_chunk=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    prediction, _ = model(batch, pair_chunk=3)
    loss = normalized_mse(prediction, target, g).sum()
    loss.backward()
    if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in model.parameters()):
        raise AssertionError("DEBUG raw-message gate is disconnected from MSE")
    gradient = max(float(value.grad.abs().max()) for value in model.parameters())
    optimizer.step()
    delta = max(float((value.detach() - before[name]).abs().max()) for name, value in model.named_parameters())
    if not math.isfinite(gradient) or gradient <= 0 or delta <= 0:
        raise AssertionError("DEBUG raw-message gate optimizer did not update")
    return {"passed": True, "scope": "independent float64 math and one explicit DEBUG fixture MSE update",
            "full_scientific_training": False, "device": str(device), "dense_fixture_conditions": len(rows),
            "dense_teacher_checks": len(rows) * 3, "max_dense_absolute_error": maximum,
            "fixture_rows": rows, "debug_seed_optimizer_updates": 2,
            "gradient_abs_max": gradient, "optimizer_change_abs_max": delta}
