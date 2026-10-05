"""All small-graph target observations, batched direct decompositions."""
from __future__ import annotations

import torch

from .numerics import null_projector, quadratic_witness, relative, deterministic_noise
from .targets import dense_target_matrices, observation_masks, evaluate_targets


def dense_matrix(action, repetitions=1):
    g = action.geometry
    eye = torch.eye(g.n, dtype=g.c0.dtype, device=g.c0.device)
    return action(eye.expand(action.coefficient_batches, g.n, g.n), repetitions)


def observability_rows(topology, matrix, baseline, recipe, *, kind,
                       target_chunk, rank_tolerance=1e-10):
    """Yield actual individual target certificates and explicit witness vectors.

One observer v is paired with its q_v/E_v and each outgoing J_vu. Full
observation evaluates every v/pair once. All target-node and one-hop observers
are included; chunks change only work memory, never target coverage.
"""
    n, references = topology.n, matrix.shape[0]
    targets = dense_target_matrices(topology, recipe)
    q, kinds = targets["q"], ("E", "J_shared", "J_node", "J_distinct")
    ranges = [(0, n)] if kind == "full" else [(s, min(s+target_chunk, n)) for s in range(0, n, target_chunk)]
    for start, stop in ranges:
        mask = observation_masks(topology, kind, start, stop)
        observed_counts = mask.sum(-1).to(torch.long).cpu().tolist()
        observed = mask[:, None, :, None]*matrix[None]
        reference = mask[:, None, :, None]*baseline[None].expand(mask.shape[0], references, n, n)
        current, old = null_projector(observed, rank_tolerance), null_projector(reference, rank_tolerance)
        p, p0 = current["projector"], old["projector"]
        both = torch.stack(((reference@p).square().sum((-2, -1)).sqrt(),
                            (observed@p0).square().sum((-2, -1)).sqrt()), -1)
        # q_v is a vector target. Null-space action is reduced per local center.
        observer = topology.local_edge_center if kind == "full" else topology.local_edge_center-start
        selected = torch.ones_like(observer, dtype=torch.bool) if kind == "full" else (observer >= 0) & (observer < stop-start)
        owners = observer[selected]
        projector = p[0].unsqueeze(0).expand(stop-start, -1, -1, -1) if kind == "full" else p
        hidden = torch.einsum("en,ernm->erm", q[selected], projector[owners])
        q_hidden = matrix.new_zeros((stop-start, references)).index_add(0, owners, hidden.square().sum(-1))
        q_norm = matrix.new_zeros(stop-start).index_add(0, owners, q[selected].square().sum(-1))
        scores = hidden.square().sum(-1)
        maximum = matrix.new_zeros((stop-start, references)).scatter_reduce_(
            0, owners[:, None].expand(-1, references), scores, reduce="amax")
        candidate_ids = torch.arange(hidden.shape[0], device=matrix.device)[:, None].expand(-1, references)
        candidates = torch.where(scores == maximum[owners], candidate_ids, hidden.shape[0])
        chosen = torch.full((stop-start, references), hidden.shape[0], dtype=torch.long, device=matrix.device)
        chosen.scatter_reduce_(0, owners[:, None].expand(-1, references), candidates, reduce="amin")
        padded = torch.cat((hidden, hidden.new_zeros((1, references, topology.n))), 0)
        reference_ids = torch.arange(references, device=matrix.device)[None].expand(stop-start, -1)
        direction = padded[chosen, reference_ids]
        length = direction.square().sum(-1).sqrt()
        direction = direction/torch.where(length > 0, length, 1.)[..., None]
        qdifference = 2*torch.einsum("en,ern->er", q[selected], direction[owners])
        qtarget_difference = matrix.new_zeros((stop-start, references)).index_add(0, owners, qdifference.square()).sqrt()
        q_observation = 2*((matrix[None] if kind == "full" else observed)@direction[..., None]).squeeze(-1)
        if kind != "full":
            q_observation = q_observation*mask[..., None, :]
        q_observation_norm = q_observation.square().sum(-1).sqrt()
        q_proof = torch.stack((q_observation_norm, qtarget_difference), -1).cpu().numpy()
        q_directions = direction.cpu().numpy()
        ratio, available = relative(q_hidden.sqrt(), q_norm.sqrt()[:, None])
        host = torch.stack((ratio, available.to(ratio.dtype)), -1).cpu().numpy()
        summary = torch.cat((current["rank"][..., None].to(matrix.dtype), current["absolute_threshold"][..., None], both), -1).cpu().numpy()
        for local in range(stop-start):
            obs = 0 if kind == "full" else local
            for realization in range(references):
                yield {"target_kind": "q", "target": start+local, "receiver": None,
                       "reference_realization": realization, "observation": kind,
                       "observed_nodes": observed_counts[obs],
                       "numerical_rank": int(summary[obs, realization, 0]),
                       "rank_threshold": float(summary[obs, realization, 1]),
                       "hidden_relative_norm": float(host[local, realization, 0]) if host[local, realization, 1] else None,
                       "baseline_visible_current_hidden": float(summary[obs, realization, 2]),
                       "current_visible_baseline_hidden": float(summary[obs, realization, 3]),
                       "collision_observation_residual": float(q_proof[local, realization, 0]) if host[local, realization, 0] > rank_tolerance else None,
                       "collision_target_difference": float(q_proof[local, realization, 1]) if host[local, realization, 0] > rank_tolerance else None,
                       "collision_formula_residual": None,
                       "collision_plus": q_directions[local, realization].tolist() if host[local, realization, 0] > rank_tolerance else None,
                       "collision_minus": (-q_directions[local, realization]).tolist() if host[local, realization, 0] > rank_tolerance else None,
                       "method": "FP64_direct_SVD_with_declared_tolerance_not_symbolic_rank"}
        for name in kinds:
            if name == "E":
                ids = torch.arange(start, stop, device=matrix.device)
                owners = ids-start
            else:
                sender = topology.pair_centers[0]
                ids = torch.nonzero((sender >= start) & (sender < stop), as_tuple=False).flatten()
                owners = sender[ids]-start
            if not ids.numel():
                continue
            k = targets[name][ids]
            obs_ids = torch.zeros_like(owners) if kind == "full" else owners
            witnesses = quadratic_witness(observed[obs_ids], k[:, None], p[obs_ids])
            norms = k.square().sum((-2, -1)).sqrt()[:, None]
            hidden, available = relative(witnesses["hidden_norm"], norms)
            fields = torch.stack((hidden, available.to(hidden.dtype), witnesses["observation_residual"],
                                  witnesses["target_difference"],
                                  (witnesses["target_difference"]-witnesses["predicted_difference"]).abs()), -1).cpu().numpy()
            plus, minus = witnesses["plus"].cpu().numpy(), witnesses["minus"].cpu().numpy()
            index_host, obs_host = ids.cpu().tolist(), obs_ids.cpu().tolist()
            receivers = topology.pair_centers[1, ids].cpu().tolist() if name != "E" else [None]*len(index_host)
            for row, identifier in enumerate(index_host):
                for realization in range(references):
                    values = fields[row, realization]
                    proof = bool(values[1] and values[0] > rank_tolerance)
                    yield {"target_kind": name, "target": identifier, "receiver": receivers[row],
                           "reference_realization": realization, "observation": kind,
                           "observed_nodes": observed_counts[obs_host[row]],
                           "numerical_rank": int(summary[obs_host[row], realization, 0]),
                           "rank_threshold": float(summary[obs_host[row], realization, 1]),
                           "hidden_relative_norm": float(values[0]) if values[1] else None,
                           "baseline_visible_current_hidden": float(summary[obs_host[row], realization, 2]),
                           "current_visible_baseline_hidden": float(summary[obs_host[row], realization, 3]),
                           "collision_observation_residual": float(values[2]) if proof else None,
                           "collision_target_difference": float(values[3]) if proof else None,
                           "collision_formula_residual": float(values[4]) if proof else None,
                           "collision_plus": plus[row, realization].tolist() if proof else None,
                           "collision_minus": minus[row, realization].tolist() if proof else None,
                           "method": "FP64_direct_SVD_with_declared_tolerance_not_symbolic_rank"}


def reconstruction_errors(topology, original, reconstructed, recipe, observed_kind, start, stop):
    """Full q vector and individual E/J error rows; no averaging relation signs."""
    expected, recovered = evaluate_targets(topology, original, recipe), evaluate_targets(topology, reconstructed, recipe)
    qdifference = (recovered["q"]-expected["q"]).square().sum(-1)
    qerror = reconstructed.new_zeros(reconstructed.shape[:-2]+(topology.n,)).index_add(-1, topology.local_edge_center, qdifference)
    qnorm = original.new_zeros(original.shape[:-2]+(topology.n,)).index_add(-1, topology.local_edge_center, expected["q"].square().sum(-1))
    for name in ("q", "E", "J_shared", "J_node", "J_distinct"):
        ids = torch.arange(start, stop, device=original.device) if name in ("q", "E") else torch.nonzero(
            (topology.pair_centers[0] >= start) & (topology.pair_centers[0] < stop), as_tuple=False).flatten()
        observer = ids-start if name in ("q", "E") else topology.pair_centers[0, ids]-start
        observed = torch.zeros_like(observer) if observed_kind == "full" else observer
        if name == "q":
            error, scale = qerror[observed, :, ids].sqrt(), qnorm[:, ids].transpose(0, 1).sqrt()
        else:
            error = (recovered[name][observed, :, ids]-expected[name][:, ids].transpose(0, 1)).abs()
            scale = expected[name][:, ids].transpose(0, 1).abs()
        ratio, defined = relative(error, scale)
        host = torch.stack((error, scale, ratio, defined.to(error.dtype)), -1).cpu().numpy()
        for row, target in enumerate(ids.cpu().tolist()):
            for realization in range(original.shape[0]):
                value = host[row, realization]
                yield {"target_kind": name, "target": target, "reference_realization": realization,
                       "absolute_error": float(value[0]), "target_norm": float(value[1]),
                       "relative_error": float(value[2]) if value[3] else None}
