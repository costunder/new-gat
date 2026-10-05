"""Batched FP64 finite-dimensional witnesses and matrix-free recovery.

Numerical SVD rank is always qualified by its reported tolerance. CG recovery
is a numerical probe; a failed convergence criterion never becomes a rank or
exact identifiability claim. All right-hand sides are processed together.
"""
from __future__ import annotations

import torch
import time


def relative(numerator, denominator):
    """Tensor quotient and availability mask, without a hidden epsilon metric."""
    numerator, denominator = torch.broadcast_tensors(numerator, denominator)
    available = denominator > 0
    return numerator / torch.where(available, denominator, torch.ones_like(denominator)), available


def null_projector(matrix, tolerance=1e-10):
    if matrix.dtype != torch.float64 or matrix.ndim < 2 or min(matrix.shape[-2:]) < 1:
        raise ValueError("direct null audit requires nonempty float64 matrices")
    if not 0 < tolerance < 1:
        raise ValueError("rank relative tolerance must lie in (0,1)")
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=True)
    threshold = tolerance * singular.amax(-1, keepdim=True)
    nonzero = singular > threshold
    rank = nonzero.sum(-1)
    n = matrix.shape[-1]
    null = torch.arange(n, device=matrix.device) >= rank[..., None]
    basis = vh.transpose(-2, -1)
    projector = (basis * null[..., None, :]) @ vh
    inverse_s = torch.where(nonzero, singular.reciprocal(), torch.zeros_like(singular))
    width = singular.shape[-1]
    inverse = (basis[..., :, :width] * inverse_s[..., None, :]) @ u[..., :, :width].transpose(-2, -1)
    return {"projector": projector, "rank": rank, "singular": singular,
            "absolute_threshold": threshold.squeeze(-1), "pseudoinverse": inverse}


def quadratic_witness(observation, target, projector, epsilon=1.):
    """Construct x±=K n±eps n for every supplied symmetric target K.

Inputs may have arbitrary broadcastable leading batch dimensions. The output
contains the actual witness vectors; unavailable zero targets remain explicit.
"""
    if epsilon <= 0:
        raise ValueError("collision epsilon must be positive")
    image = target @ projector
    column = image.square().sum(-2).argmax(-1)
    selected = projector.gather(-1, column[..., None, None].expand(*projector.shape[:-1], 1)).squeeze(-1)
    length = selected.square().sum(-1).sqrt()
    n = selected / torch.where(length > 0, length, torch.ones_like(length))[..., None]
    x0 = (target @ n[..., None]).squeeze(-1)
    plus, minus = x0 + epsilon*n, x0 - epsilon*n
    observed_difference = (observation @ (plus-minus)[..., None]).squeeze(-1)
    target_difference = ((plus[..., None, :] @ target @ plus[..., :, None])
                         -(minus[..., None, :] @ target @ minus[..., :, None])).squeeze(-1).squeeze(-1)
    return {"direction": n, "plus": plus, "minus": minus,
            "observation_residual": observed_difference.square().sum(-1).sqrt(),
            "target_difference": target_difference,
            "predicted_difference": 4*epsilon*x0.square().sum(-1),
            "hidden_norm": image.square().sum((-2, -1)).sqrt()}


def conjugate_gradient(action, rhs, *, tolerance=1e-8, maximum_iterations,
                       check_every=16, progress=None):
    """CG on each [batch,node,channel] RHS with no sample/channel loop.

The caller must supply a symmetric PSD action and a consistent RHS, or an
explicit regularizer. Breakdown and unconverged RHS are returned as statuses.
They are never replaced with a successful zero estimate.
"""
    if (rhs.ndim != 3 or rhs.dtype != torch.float64 or maximum_iterations < 1
            or not 0 < tolerance < 1 or check_every < 1):
        raise ValueError("CG requires packed float64 RHS and a positive numerical iteration ceiling")
    x = torch.zeros_like(rhs)
    r, p = rhs.clone(), rhs.clone()
    rr = r.square().sum(-2)
    baseline = rr.clone()
    target = baseline * tolerance*tolerance
    converged = rr <= target
    breakdown = torch.zeros_like(converged)
    iterations = torch.zeros_like(rr, dtype=torch.long)
    used = 0
    reported = time.perf_counter()
    for step in range(maximum_iterations):
        ap = action(p)
        pap = (p*ap).sum(-2)
        active = ~(converged | breakdown)
        invalid = active & ((pap <= 0) | ~torch.isfinite(pap))
        breakdown |= invalid
        active &= ~invalid
        alpha = torch.where(active, rr/torch.where(active, pap, torch.ones_like(pap)), 0.)
        x = x + alpha[..., None, :]*p
        r = r - alpha[..., None, :]*ap
        updated = r.square().sum(-2)
        converged |= updated <= target
        iterations = torch.where(active, iterations.new_full(iterations.shape, step+1), iterations)
        beta = torch.where(active & ~converged, updated/torch.where(rr > 0, rr, torch.ones_like(rr)), 0.)
        p = r + beta[..., None, :]*p
        p = torch.where((~(converged | breakdown))[..., None, :], p, 0.)
        rr, used = updated, step+1
        if used % check_every == 0 and bool((converged | breakdown).all()):
            break
        if progress is not None and used % check_every == 0 and time.perf_counter()-reported >= 20:
            progress(used, maximum_iterations)
            reported = time.perf_counter()
    actual = action(x)-rhs
    residual = actual.square().sum(-2).sqrt()
    denominator = baseline.sqrt()
    rel, available = relative(residual, denominator)
    # The recurrence is not a substitute for the true matvec residual.
    verified = torch.where(available, rel <= tolerance, residual == 0)
    return x, {"iterations": iterations, "loop_iterations": used,
               "relative_residual": rel, "rhs_nonzero": available,
               "converged": verified, "breakdown": breakdown}


def deterministic_noise(shape, reference, *, target_offset=0, channel_offset=0):
    """Reproducible bounded probe noise; explicitly not simulated natural data."""
    batch, nodes, channels = shape
    a = torch.arange(target_offset, target_offset+batch, device=reference.device, dtype=reference.dtype)
    b = torch.arange(nodes, device=reference.device, dtype=reference.dtype)
    c = torch.arange(channel_offset, channel_offset+channels, device=reference.device, dtype=reference.dtype)
    return torch.sin((a[:, None, None]+1)*.731 + (b[None, :, None]+1)*1.213 + (c[None, None, :]+1)*.417)
