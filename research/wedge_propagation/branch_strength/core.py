"""Frozen Experiment 4.1 probes: separate path weighting from branch strength.

The classifier, features, graph, learned alpha/beta and two-layer forward stay
unchanged. Reference C and kappa are always calculated at the CURRENT projected
Z, including the second layer after an upstream intervention. A norm match uses
the complete node/channel Frobenius norm separately for every packed seed.

This package deliberately does not edit Experiment 4 source files: its saved
checkpoint/source identities and resume contract remain verifiable.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch
from torch import Tensor

from research.wedge_propagation.classification.model import PackedClassifier, _qbar

TREATMENTS = (
    "baseline",
    "true_c_unit_denominator",
    "identity_hold_reference",
    "identity_unit_denominator",
    "identity_norm_matched",
    "branch_off",
    "shuffle_hold_reference",
    "shuffle_norm_matched",
)
LEARNED_CONDITIONS = ("learned_wedge_raw", "learned_wedge_rms")
_Q_INDEX_CACHE: dict[int, tuple[Any, int, Tensor]] = {}


def _norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value, dim=(1, 2))


def _ratio(numerator: Tensor, denominator: Tensor) -> tuple[Tensor, Tensor]:
    defined = denominator > 0
    safe = torch.where(defined, denominator, torch.ones_like(denominator))
    result = numerator / safe
    return torch.where(defined, result, torch.full_like(result, torch.nan)), defined


def _cosine(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    return _ratio((left * right).sum((1, 2)), _norm(left) * _norm(right))


def _match_norm(candidate: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
    target_norm, candidate_norm = _norm(reference), _norm(candidate)
    available = candidate_norm > 0
    if bool(((~available) & (target_norm > 0)).any()):
        raise ValueError("cannot match a positive reference norm with a zero candidate message")
    safe = torch.where(available, candidate_norm, torch.ones_like(candidate_norm))
    gain = torch.where(available, target_norm / safe, torch.ones_like(safe))
    return candidate * gain[:, None, None], gain


def _strength_statistics(
    detail: dict[str, Any], candidate: Tensor, reference: Tensor
) -> dict[str, Tensor]:
    """Scalar tensors per seed; NaN plus a boolean denotes an undefined ratio."""
    z, l_message = detail["z"], detail["l_message"]
    if l_message is None:
        l_message = torch.zeros_like(z)
    alpha_l = detail["alpha"][:, None, None] * l_message
    beta_t = detail["beta"][:, None, None] * candidate
    beta_reference = detail["beta"][:, None, None] * reference
    delta = beta_t - beta_reference
    stats = {
        "z_norm": _norm(z),
        "l_norm": _norm(l_message),
        "reference_branch_norm": _norm(reference),
        "branch_norm": _norm(candidate),
        "alpha_l_norm": _norm(alpha_l),
        "beta_t_norm": _norm(beta_t),
        "reference_beta_t_norm": _norm(beta_reference),
        "branch_delta_norm": _norm(delta),
    }
    stats.update(
        input_norm=stats["z_norm"],
        first_norm=stats["alpha_l_norm"],
        second_norm=stats["beta_t_norm"],
    )
    for name, numerator, denominator in (
        ("alpha_l_to_input", stats["alpha_l_norm"], stats["z_norm"]),
        ("beta_t_to_input", stats["beta_t_norm"], stats["z_norm"]),
        ("beta_t_to_alpha_l", stats["beta_t_norm"], stats["alpha_l_norm"]),
        ("branch_delta_to_input", stats["branch_delta_norm"], stats["z_norm"]),
        ("first_to_input", stats["first_norm"], stats["input_norm"]),
        ("second_to_input", stats["second_norm"], stats["input_norm"]),
        ("second_to_first", stats["second_norm"], stats["first_norm"]),
        ("message_norm_ratio", stats["branch_norm"], stats["reference_branch_norm"]),
        ("message_relative_change", _norm(candidate - reference), stats["reference_branch_norm"]),
    ):
        value, defined = _ratio(numerator, denominator)
        stats[name], stats[name + "_defined"] = value, defined
    for name, left, right in (
        ("cosine_l_input", l_message, z),
        ("cosine_t_input", candidate, z),
        ("cosine_l_t", l_message, candidate),
        ("branch_reference_cosine", candidate, reference),
        ("cosine_correction_input", -alpha_l - beta_t, z),
        ("first_second_cosine", alpha_l, beta_t),
        ("second_input_cosine", beta_t, z),
        ("message_reference_cosine", candidate, reference),
    ):
        value, defined = _cosine(left, right)
        stats[name], stats[name + "_defined"] = value, defined
    return {name: value.detach() for name, value in stats.items()}


def _active_q_indices(qdiag: Tensor) -> Tensor:
    """Cache static positive-node indices without retaining the source graph."""
    version = None if torch.is_inference(qdiag) else qdiag._version
    key = id(qdiag)
    previous = _Q_INDEX_CACHE.get(key)
    if (
        version is not None
        and previous is not None
        and previous[0]() is qdiag
        and previous[1] == version
    ):
        return previous[2]
    indices = (qdiag.detach().cpu() > 0).nonzero().flatten().to(qdiag.device)
    if version is not None:

        def discard(reference):
            cached = _Q_INDEX_CACHE.get(key)
            if cached is not None and cached[0] is reference:
                _Q_INDEX_CACHE.pop(key, None)

        _Q_INDEX_CACHE[key] = (weakref.ref(qdiag, discard), version, indices)
    return indices


def _kappa_statistics(graph: Any, c: Tensor) -> dict[str, Tensor]:
    """Distribution over q>0 nodes; near maximum means within 1% of maximum."""
    active_nodes = _active_q_indices(graph.qdiag)
    count = active_nodes.numel()
    defined = torch.full((c.shape[0],), count > 0, device=c.device, dtype=torch.bool)
    names = (
        "kappa_ratio_mean",
        "kappa_ratio_p50",
        "kappa_ratio_p90",
        "kappa_ratio_p99",
        "kappa_ratio_max",
        "kappa_near_max_fraction",
    )
    if graph.paths.shape[1] == 0:
        result = {name: c.new_full((c.shape[0],), torch.nan) for name in names}
        result["kappa_max_node"] = torch.full((c.shape[0],), -1, device=c.device, dtype=torch.long)
    else:
        diagonal = c.new_zeros((c.shape[0], graph.x.shape[0]))
        diagonal.index_add_(1, graph.paths[0], c)
        diagonal.index_add_(1, graph.paths[1], 4 * c)
        diagonal.index_add_(1, graph.paths[2], c)
        ratios = (
            diagonal.index_select(1, active_nodes) / graph.qdiag.index_select(0, active_nodes)[None]
        )
        maximum, location = ratios.max(1)
        quantiles = torch.quantile(ratios, c.new_tensor((0.5, 0.9, 0.99)), dim=1)
        result = {
            "kappa_ratio_mean": ratios.mean(1),
            "kappa_ratio_p50": quantiles[0],
            "kappa_ratio_p90": quantiles[1],
            "kappa_ratio_p99": quantiles[2],
            "kappa_ratio_max": maximum,
            "kappa_max_node": active_nodes.index_select(0, location),
            "kappa_near_max_fraction": (ratios >= 0.99 * maximum[:, None]).to(c.dtype).mean(1),
        }
    result["kappa_ratio_defined"] = defined
    result["kappa_active_node_count"] = torch.full(
        (c.shape[0],), count, device=c.device, dtype=torch.long
    )
    result["kappa_near_max_relative_tolerance"] = c.new_full((c.shape[0],), 0.01)
    return {name: value.detach() for name, value in result.items()}


class FrozenStrengthClassifier(PackedClassifier):
    """Reuse the exact Experiment 4 forward with a frozen branch probe.

    Construct with ``from_classifier`` to copy a verified selected checkpoint.
    The copy owns identical weights/buffers and has no trainable parameters.
    No C tensor from a clean forward is retained for a later-layer intervention.
    """

    @classmethod
    def from_classifier(cls, model: PackedClassifier) -> FrozenStrengthClassifier:
        if not isinstance(model, PackedClassifier):
            raise TypeError("source must be an Experiment 4 PackedClassifier")
        if model.condition not in LEARNED_CONDITIONS:
            raise ValueError("branch-strength interventions require a learned C condition")
        reference = model.projections[0]
        result = cls(
            model.condition,
            model.input_dim,
            model.classes,
            model.seeds,
            hidden=model.hidden,
            gate_hidden=model.gate_hidden,
            dropout=model.dropout,
            path_chunk=model.path_chunk,
            checkpoint_paths=False,
            dataset_name=model.dataset_name,
        ).to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(model.state_dict(), strict=True)
        # Nonpersistent buffers also belong to the frozen identity.
        original_buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, value in result.named_buffers():
                value.copy_(original_buffers[name])
        return result.requires_grad_(False).eval()

    def set_treatment(
        self, treatment: str, target: str = "both", manifest: dict | None = None
    ) -> FrozenStrengthClassifier:
        if treatment not in TREATMENTS:
            raise ValueError(f"unknown branch-strength treatment {treatment!r}")
        if target not in ("layer_0", "layer_1", "both"):
            raise ValueError("target must be layer_0, layer_1 or both")
        if treatment.startswith("shuffle_") and (manifest is None or "permutation" not in manifest):
            raise ValueError("shuffle treatment requires manifest['permutation']")
        self._strength_treatment = treatment
        self._strength_target = target
        self._strength_manifest = manifest
        return self

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("Experiment 4.1 classifier is frozen; training is disabled")
        return super().train(False)

    def probe_layer(
        self,
        graph: Any,
        layer_index: int,
        z: Tensor,
        input_scale: float = 1.0,
        intervention: str | None = None,
        manifest: dict | None = None,
        kappa_mode: str = "recompute",
        diagnostics: bool = True,
        _l_message: Tensor | None = None,
    ):
        if self.training:
            raise RuntimeError("Experiment 4.1 probes require eval mode")
        treatment = (
            getattr(self, "_strength_treatment", "baseline")
            if intervention is None
            else intervention
        )
        target = getattr(self, "_strength_target", "both")
        if target != "both" and target != f"layer_{layer_index}":
            treatment = "baseline"
        if manifest is None:
            manifest = getattr(self, "_strength_manifest", None)
        if treatment not in TREATMENTS:
            raise ValueError(f"unknown branch-strength treatment {treatment!r}")
        if kappa_mode != "recompute":
            raise ValueError("denominator treatment is explicit; leave kappa_mode='recompute'")
        reference, detail = super().probe_layer(
            graph,
            layer_index,
            z,
            input_scale=input_scale,
            intervention=None,
            diagnostics=True,
            _l_message=_l_message,
        )
        current_z = detail["z"]
        reference_c = detail["c_reference"]
        reference_kappa = detail["kappa_reference"]
        c, denominator = reference_c, reference_kappa
        gain = torch.ones_like(reference_kappa)
        if treatment == "baseline":
            candidate = reference
        elif treatment == "true_c_unit_denominator":
            denominator = torch.ones_like(reference_kappa)
            candidate = reference * reference_kappa[:, None, None]
        elif treatment.startswith("identity_"):
            c = torch.ones_like(reference_c)
            denominator = (
                reference_kappa
                if treatment == "identity_hold_reference"
                else torch.ones_like(reference_kappa)
            )
            candidate = _qbar(graph, current_z) / denominator[:, None, None]
            if treatment == "identity_norm_matched":
                candidate, gain = _match_norm(candidate, reference)
        elif treatment == "branch_off":
            candidate = torch.zeros_like(reference)
        else:
            if manifest is None or "permutation" not in manifest:
                raise ValueError("shuffle treatment requires manifest['permutation']")
            permutation = manifest["permutation"]
            if permutation.dtype != torch.long or permutation.shape != (graph.paths.shape[1],):
                raise ValueError("permutation must be Long[P]")
            if permutation.device != current_z.device:
                raise ValueError("permutation must be on the model device")
            self._validate_manifest(graph, (permutation,), "permutation")
            c = reference_c.index_select(1, permutation)
            candidate = self._weighted_message(graph, current_z, c, denominator)
            if treatment == "shuffle_norm_matched":
                candidate, gain = _match_norm(candidate, reference)
        detail.update(
            kappa=denominator.detach(),
            kappa_used=denominator.detach(),
            normalization_gain=gain.detach(),
            applied_gain=gain.detach(),
            **self._stats(c),
            **_strength_statistics(detail, candidate, reference),
            **_kappa_statistics(graph, reference_c),
        )
        if diagnostics:
            detail.update(
                c=c.detach(),
                t_message=candidate.detach(),
                reference_t_message=reference.detach(),
            )
        else:
            for name in ("z", "c", "c_reference", "l_message", "t_message"):
                detail.pop(name, None)
        return candidate, detail


@torch.inference_mode()
def fixed_z_statistics(
    model: PackedClassifier,
    graph: Any,
    layer_index: int,
    z: Tensor,
    treatment: str | None = None,
    manifest: dict | None = None,
) -> dict[str, Tensor | None]:
    """Measure a clean fixed-Z probe without labels or a subsequent projection.

    For existing control models only the unmodified baseline is allowed. The
    learned gate wrapper supports all explicit treatments. Statistics are
    node/channel norms within a seed, never reductions across independent seeds.
    """
    if model.training:
        raise ValueError("fixed-Z statistics require eval mode")
    if isinstance(model, FrozenStrengthClassifier):
        _, detail = model.probe_layer(
            graph, layer_index, z, intervention=treatment, manifest=manifest, diagnostics=True
        )
    else:
        if treatment not in (None, "baseline"):
            raise ValueError("control models support only the baseline fixed-Z diagnostic")
        message, detail = model.probe_layer(graph, layer_index, z, diagnostics=True)
        detail.update(
            normalization_gain=torch.ones_like(detail["kappa"]),
            applied_gain=torch.ones_like(detail["kappa"]),
            kappa_used=detail["kappa"],
            **_strength_statistics(detail, message, message),
        )
    return {name: value for name, value in detail.items() if value is None or value.ndim <= 1}


__all__ = ["TREATMENTS", "LEARNED_CONDITIONS", "FrozenStrengthClassifier", "fixed_z_statistics"]
