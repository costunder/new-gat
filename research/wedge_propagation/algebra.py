"""Experiment 0: executable mathematical checks, with no trainable model."""

from __future__ import annotations

import torch

from .operators import (
    build_wedges,
    dense_incidence,
    dense_R,
    dense_wedge,
    fixed_wedge_apply,
    fixed_wedge_fast_apply,
)


def run_algebra() -> list[dict]:
    """Check independent dense references on explicitly named debug fixtures."""
    fixtures = {
        "path": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        "cycle": (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        "star": (5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
        "clique": (4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
        "irregular": (6, [(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]),
        "isolates": (5, [(0, 1)]),
        "empty": (4, []),
    }
    result = []
    rng = torch.Generator().manual_seed(20261002)
    for name, (n, pairs) in fixtures.items():
        edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
        wedges = build_wedges(edges, n)
        b = dense_incidence(edges, n)
        a = dense_wedge(wedges, n)
        r = dense_R(edges, wedges, n)
        lap = b.T @ b
        q = a.T @ a
        degree = torch.bincount(edges.flatten(), minlength=n).double()
        weight = degree[edges[0]] + degree[edges[1]] - 4
        identity = lap @ lap + b.T @ (weight[:, None] * b)
        torch.testing.assert_close(r @ b, a, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(identity, q, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(a.sum(1), torch.zeros(a.shape[0], dtype=a.dtype))
        if q.numel() and torch.linalg.eigvalsh(q).min() < -1e-10:
            raise AssertionError(f"{name}: path operator is not PSD")
        reversed_a = dense_wedge(wedges.flip(0), n)
        torch.testing.assert_close(reversed_a, a)
        flipped = edges.flip(0).contiguous()
        torch.testing.assert_close(dense_wedge(build_wedges(flipped, n), n), a)
        torch.testing.assert_close(dense_R(flipped, wedges, n) @ dense_incidence(flipped, n), a)
        perm = torch.randperm(n, generator=rng)
        perm_a = dense_wedge(perm[wedges], n)
        torch.testing.assert_close(perm_a[:, perm], a)
        rebuilt_a = dense_wedge(build_wedges(perm[edges], n), n)
        torch.testing.assert_close((rebuilt_a.T @ rebuilt_a)[perm[:, None], perm[None, :]], q)
        x = torch.randn((n, 3), dtype=torch.float64, generator=rng, requires_grad=True)
        direct = fixed_wedge_apply(wedges, x)
        fast = fixed_wedge_fast_apply(edges, x)
        torch.testing.assert_close(direct, q @ x, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(fast, direct, rtol=1e-10, atol=1e-10)
        probe = torch.randn((n, 3), dtype=torch.float64, generator=rng)
        direct_grad = torch.autograd.grad((direct * probe).sum(), x, retain_graph=True)[0]
        fast_grad = torch.autograd.grad((fast * probe).sum(), x, retain_graph=True)[0]
        torch.testing.assert_close(direct_grad, fast_grad, rtol=1e-10, atol=1e-10)
        energy = (a @ x).square().sum()
        torch.testing.assert_close(energy, (x * direct).sum(), rtol=1e-10, atol=1e-10)
        grad = torch.autograd.grad(energy, x)[0]
        torch.testing.assert_close(grad, 2 * direct, rtol=1e-10, atol=1e-10)
        if name == "cycle":
            torch.testing.assert_close(q, lap @ lap)
        if name == "star":
            torch.testing.assert_close(q, lap @ lap + (n - 4) * lap)
        row = {
            "fixture": name,
            "nodes": n,
            "edges": len(pairs),
            "wedges": wedges.shape[1],
            "status": "passed",
            "dtype": "float64",
        }
        result.append(row)
        print(f"[algebra] {name}: n={n} e={len(pairs)} p={wedges.shape[1]} passed", flush=True)
    return result
