"""DEBUG independent dense references and actual gate autograd checks.

No public-data model is fitted here. The tiny fixture labels are explicitly
DEBUG CE connection probes and are never exported as scientific accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

from .geometry import prepare_geometry
from .gates import EdgeMetricGate
from .operators import apply_metric
from ..local_energy_relations.topology import build_topology


def dense_reference(g, multiplier, pair_coefficient):
    """Independent occurrence-space B/C assembly; small DEBUG graphs only."""
    if multiplier.ndim == 2:
        multiplier = multiplier[0]
    if pair_coefficient.ndim == 2:
        pair_coefficient = pair_coefficient[0]
    b = g.c0.new_zeros((g.num_edges, g.n))
    if g.num_edges:
        rows = torch.arange(g.num_edges, device=b.device)
        b[rows, g.edges[0]] = -1
        b[rows, g.edges[1]] = 1
    bcal = b[g.occ_edge]*g.occ_scale[:, None]
    diagonal = g.c0*multiplier[g.occ_edge]
    k = b.new_zeros((g.num_occurrences, g.num_occurrences))
    if g.num_pairs:
        k.index_put_((g.pair_left, g.pair_right), pair_coefficient, accumulate=True)
        k.index_put_((g.pair_right, g.pair_left), pair_coefficient, accumulate=True)
    metric = diagonal.sqrt()[:, None]*(torch.eye(g.num_occurrences, device=b.device, dtype=b.dtype)+g.rho*k)*diagonal.sqrt()[None, :]
    return bcal.T@metric@bcal, b, metric


def fixture():
    edges = np.asarray([[0, 0, 1, 1, 2, 3], [1, 2, 2, 3, 4, 4]], dtype=np.int64)
    return build_topology(5, edges)


def _activate(gate):
    with torch.no_grad():
        for number, parameter in enumerate(gate.parameters()):
            data = torch.arange(parameter.numel(), dtype=parameter.dtype, device=parameter.device)
            parameter.copy_((.07*torch.sin(data*.31+number+1)).reshape(parameter.shape))


def run_checks(device="cpu", *, input_gradcheck=True):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; DEBUG never silently falls back")
    checks, maximum_error = 0, 0.
    for recipe in ("unit", "local_degree"):
        g = prepare_geometry(fixture(), recipe).to(device, torch.float64)
        z = torch.linspace(-.8, 1.2, 15, dtype=torch.float64, device=device).reshape(1, 5, 3)
        for condition in ("D0", "D1", "F0", "F1", "F2", "DA"):
            gate = EdgeMetricGate(condition, seeds=(11,), hidden=64).to(device=device, dtype=torch.float64)
            _activate(gate)
            coefficients = gate.coefficients(g, z, pair_chunk=3)
            q, b, metric = dense_reference(g, coefficients["diagonal_multiplier"], coefficients["pair_coefficient"])
            actual, detail = gate.apply(g, z, z, pair_chunk=3, diagnostics=True)
            expected = q@z
            error = float((actual-expected).detach().abs().max().cpu())
            maximum_error = max(maximum_error, error)
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(detail["energy_total"].sum(), .5*(z*actual).sum(), atol=1e-11, rtol=1e-11)
            if metric.numel():
                assert float(torch.linalg.eigvalsh(metric.detach()).min().cpu()) > 0
            assert float(torch.linalg.eigvalsh(q.detach()).min().cpu()) >= -1e-10
            n0 = g.n0
            p = torch.eye(g.n, device=device, dtype=z.dtype)-g.tau*n0[:, None]*q*n0[None, :]
            assert float(torch.linalg.eigvalsh(p.detach()).abs().max().cpu()) <= 1+1e-10
            torch.testing.assert_close(q@torch.ones((g.n, 1), device=device, dtype=z.dtype),
                                       torch.zeros((g.n, 1), device=device, dtype=z.dtype), atol=1e-11, rtol=1e-11)
            # Orientation is a representation choice, not an input change.
            signs = torch.where(torch.arange(g.num_edges, device=device)%2 == 0, -1., 1.).to(z.dtype)
            edges = torch.where(signs[None, :] < 0, g.edges.flip(0), g.edges)
            occurrence_sign = signs[g.occ_edge]
            altered = replace(g, edges=edges, pair_sign=g.pair_sign*occurrence_sign[g.pair_left]*occurrence_sign[g.pair_right])
            reoriented, _ = gate.apply(altered, z, z, pair_chunk=2)
            torch.testing.assert_close(actual, reoriented, atol=1e-11, rtol=1e-11)
            off = gate.coefficients(g, z, intervention="offdiag_zero")
            off_q, _, _ = dense_reference(g, off["diagonal_multiplier"], off["pair_coefficient"])
            expected_off = b.T@torch.diag(g.cbar0*off["diagonal_multiplier"][0])@b
            torch.testing.assert_close(off_q, expected_off, atol=1e-11, rtol=1e-11)
            checks += 7
        gate = EdgeMetricGate("F2", seeds=(11,), hidden=64).to(device=device, dtype=torch.float64)
        _activate(gate)
        z = torch.sin(torch.arange(10, dtype=torch.float64, device=device)*.713+.37).reshape(1, 5, 2)
        if input_gradcheck:
            for source in (z, torch.zeros_like(z), torch.full_like(z, .2)):
                source = source.requires_grad_()
                function = lambda x: gate.apply(g, x, x*g.n0[:, None], pair_chunk=3)[0]
                assert torch.autograd.gradcheck(function, (source,), eps=1e-7, atol=3e-5, rtol=3e-4)
                checks += 1
        # Complete input->gate->CE->backward->Adam path with explicit DEBUG labels.
        optimizer = torch.optim.Adam(gate.parameters(), lr=.003)
        before = {name: parameter.detach().clone() for name, parameter in gate.named_parameters()}
        logits, _ = gate.apply(g, z, z*g.n0[:, None], pair_chunk=2)
        labels = torch.tensor([0, 1, 0, 1, 0], device=device)
        torch.nn.functional.cross_entropy(logits[0], labels).backward()
        for name, parameter in gate.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f"gate disconnected or nonfinite: {name}")
        optimizer.step()
        changed = {name: float((parameter.detach()-before[name]).norm().cpu()) for name, parameter in gate.named_parameters()}
        assert any(value > 0 for name, value in changed.items() if name.startswith("pair_gate"))
        assert any(value > 0 for name, value in changed.items() if name.startswith("diagonal_gate"))
        checks += 2
        # Two-hop support concerns the value action with coefficients frozen.
        # The adaptive coefficient generator may inspect a wider input context.
        frozen = {key: value.detach() if isinstance(value, torch.Tensor) else value
                  for key, value in gate.coefficients(g, z).items()}
        jacobian = torch.autograd.functional.jacobian(
            lambda x: apply_metric(g, x*g.n0[:, None], frozen["diagonal_multiplier"],
                                   frozen["pair_coefficient"]), z)
        adjacency = torch.eye(g.n, device=device, dtype=torch.bool)
        adjacency[g.edges[0], g.edges[1]] = True
        adjacency[g.edges[1], g.edges[0]] = True
        two_hop = (adjacency.to(torch.float64)@adjacency.to(torch.float64)) > 0
        dependence = jacobian[0, :, :, 0].abs().amax((1, 3))
        assert float(dependence.masked_select(~two_hop).max().cpu()) < 1e-10 if (~two_hop).any() else True
        checks += 1
    return {"passed": True, "scope": "DEBUG_independent_dense_RMS_gradcheck_and_CE_connection_only",
            "math_checks": checks, "maximum_dense_absolute_error": maximum_error,
            "actual_data": False, "scientific_training_run": False,
            "debug_optimizer_updates": 2, "device": str(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import json
    print(json.dumps(run_checks(args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
