"""Small FP64 algebra checks for a PROPOSED design; no training or public-data run.

The latest repository is not modified. Random matrices below are test fixtures,
not learned parameters or scientific benchmark results.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def incidence(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    b = np.zeros((len(edges), n))
    for k, (u, v) in enumerate(edges):
        b[k, u], b[k, v] = -1., 1.
    return b


def nullspace(a: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    rank = int((s > tol * max(1., float(s.max(initial=0.)))).sum())
    return vt[rank:].T


def err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main() -> dict:
    rng = np.random.default_rng(73129)
    n = 6
    edges = [(0,1), (1,2), (1,3), (2,3), (3,4), (4,5), (1,5)]
    b = incidence(n, edges)
    m = len(edges)
    c = np.exp(rng.uniform(-.4, .4, m))
    lc = b.T @ (c[:,None] * b)
    l0 = b.T @ b
    h = rng.normal(size=(n,3))
    result: dict[str, object] = {
        'scope': 'small synthetic FP64 algebra only; no optimizer, no classifier',
        'baseline_power_identity_abs': err(l0 @ l0, b.T @ (b @ b.T) @ b),
        'weighted_power_identity_abs': err(lc @ lc, b.T @ (c[:,None] * (b @ b.T) * c[None,:]) @ b),
        'weighted_L_is_not_unweighted_L_squared_abs': err(lc, l0 @ l0),
        'known_C_q_reconstruction_abs': err(c[:,None]*(b @ h), c[:,None]*(b @ (np.linalg.pinv(lc) @ (lc @ h)))),
    }
    neighbors = [set() for _ in range(n)]
    for u,v in edges:
        neighbors[u].add(v); neighbors[v].add(u)
    sets = [sorted({v}|neighbors[v]) for v in range(n)]
    occurrences: list[tuple[int,int]] = []
    for v, sv in enumerate(sets):
        for e,(u,w) in enumerate(edges):
            if u in sv and w in sv:
                occurrences.append((v,e))
    counts = np.bincount([e for _,e in occurrences], minlength=m)
    uocc = np.zeros((len(occurrences),m))
    for k,(_,e) in enumerate(occurrences):
        uocc[k,e] = 1. / np.sqrt(counts[e])
    f = uocc @ b
    d = np.array([c[e] for _,e in occurrences])
    a = np.zeros((len(occurrences),len(occurrences)))
    for k,(v,e) in enumerate(occurrences):
        for l,(u,e2) in enumerate(occurrences[:k]):
            if u not in neighbors[v] or e == e2:
                continue
            common = set(edges[e]) & set(edges[e2])
            if not common:
                continue
            anchor = next(iter(common))
            a[k,l] = a[l,k] = b[e,anchor] * b[e2,anchor] * rng.uniform(-1,1)
    degree = np.maximum(1., (a != 0.).sum(axis=1).astype(float))
    kappa = a / np.sqrt(degree[:,None] * degree[None,:])
    rho = .5
    metric = np.sqrt(d[:,None]*d[None,:]) * (np.eye(len(d)) + rho*kappa)
    q = f.T @ metric @ f
    q0 = f.T @ (d[:,None]*f)
    ceff = uocc.T @ metric @ uocc
    intra_e = float(np.sum((f @ h)**2 * d[:,None]))
    cross_e = float(np.sum((np.sqrt(d)[:,None]*(f @ h)) * (kappa @ (np.sqrt(d)[:,None]*(f @ h)))))
    result.update({
        'occurrence_partition_of_unity_abs': err(uocc.T @ uocc, np.eye(m)),
        'off_equals_physical_weighted_L_abs': err(q0, lc),
        'effective_metric_factorization_abs': err(q, b.T @ ceff @ b),
        'effective_metric_diagonal_unchanged_abs': err(np.diag(ceff), c),
        'energy_block_expansion_abs': abs(float(np.sum(h*(q@h))) - intra_e - rho*cross_e),
        'metric_smallest_eigenvalue': float(np.linalg.eigvalsh(metric).min()),
        'lower_Loewner_bound_min_eigenvalue': float(np.linalg.eigvalsh(q-(1-rho)*q0).min()),
        'upper_Loewner_bound_min_eigenvalue': float(np.linalg.eigvalsh((1+rho)*q0-q).min()),
    })
    signs = rng.choice([-1.,1.],m)
    so = np.array([signs[e] for _,e in occurrences])
    b2 = signs[:,None]*b
    metric2 = so[:,None]*metric*so[None,:]
    result['orientation_invariance_abs'] = err(q, (uocc@b2).T @ metric2 @ (uocc@b2))
    dnode = 1. + np.diag(q0)
    s = q / np.sqrt(dnode[:,None]*dnode[None,:])
    # Shared topology-based step also allows diagonal multipliers up to two.
    p = np.eye(n) - s/(2*(1+rho))
    result['frozen_propagation_spectral_norm'] = float(np.abs(np.linalg.eigvalsh(p)).max())
    assert result['metric_smallest_eigenvalue'] > 0
    assert result['frozen_propagation_spectral_norm'] <= 1+1e-12

    # Actual one-hop induced ego copies, local intra and same-node copy coupling.
    copies = [(v,i) for v,sv in enumerate(sets) for i in sv]
    index = {vi:k for k,vi in enumerate(copies)}
    r = np.zeros((len(copies),n))
    for k,(_,i) in enumerate(copies): r[k,i] = 1
    merge = r.T / np.diag(r.T@r)[:,None]
    a_intra = np.zeros((len(copies),len(copies)))
    for v,sv in enumerate(sets):
        inds = [index[(v,i)] for i in sv]
        local_edges = [(sv.index(u),sv.index(w)) for u,w in edges if u in sv and w in sv]
        bv = incidence(len(sv),local_edges)
        lv = bv.T@bv
        step = .5/np.diag(lv).max(initial=0.) if local_edges else 0.
        a_intra[np.ix_(inds,inds)] = step*lv
    g = np.zeros_like(a_intra)
    for v,u in edges:
        for i in set(sets[v]) & set(sets[u]):
            j,k = index[(v,i)],index[(u,i)]
            g[j,j]+=1; g[k,k]+=1; g[j,k]-=1; g[k,j]-=1
    g *= .5 / np.diag(g).max()
    a_id = np.eye(len(copies))-a_intra
    t0 = merge@a_id@a_id@r
    t1 = merge@a_id@(np.eye(len(copies))-.5*g)@a_id@r
    nk = nullspace(t0)
    weighted = np.diag(r.T@r)
    result.update({
        'copy_GR_abs': float(np.linalg.norm(g@r)),
        'copy_MG_abs': float(np.linalg.norm(merge@g)),
        'copy_delta_formula_abs': err(t1-t0,-.5*merge@a_intra@g@a_intra@r),
        'copy_T0_kernel_dimension': int(nk.shape[1]),
        'copy_kernel_containment_residual_abs': float(np.linalg.norm(t1@nk)),
        'copy_energy_factorization_abs': abs(float(np.sum(h*(weighted[:,None]*(t0@h))))-float(np.sum((a_id@r@h)**2))),
    })
    # Force a nontrivial off nullspace to test the general algebraic statement.
    rn = np.vstack([np.eye(3),np.eye(3)])
    mn = .5*rn.T
    proj = np.diag([0.,1.,1.,0.,1.,1.])
    gn = .5*np.block([[np.eye(3),-np.eye(3)],[-np.eye(3),np.eye(3)]])
    tt0=mn@proj@proj@rn;tt1=mn@proj@(np.eye(6)-.7*gn)@proj@rn
    nk=nullspace(tt0)
    result['nontrivial_null_control_dim']=int(nk.shape[1])
    result['nontrivial_null_control_on_residual_abs']=float(np.linalg.norm(tt1@nk))
    assert nk.shape[1] == 1
    assert result['nontrivial_null_control_on_residual_abs'] < 1e-12
    for key,value in result.items():
        if key.endswith('_abs') and 'is_not' not in key:
            assert float(value)<1e-9,(key,value)
    result['passed']=True
    return result


if __name__ == '__main__':
    result=main()
    path=Path(__file__).with_name('math_reference_results.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
