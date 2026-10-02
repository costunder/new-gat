"""Portable debug checks; source artifacts come from an actual complete debug run."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest
import torch

from research.wedge_propagation.generalization.data import load_source_run
from research.wedge_propagation.generalization.frozen import (
    frozen_fingerprint,
    load_models,
    scale_diagnostics,
)
from research.wedge_propagation.learned.data import pack_cases
from research.wedge_propagation.learned.model import make_model, teacher_weights
from research.wedge_propagation.operators import build_wedges


@pytest.fixture(scope="module")
def completed_source(wedge_completed_debug_run):
    return load_source_run(wedge_completed_debug_run, "debug")


def changed_source(source, directory):
    """Copy authentic artifacts to a private debug directory before corruption."""
    for relative in source.hashes:
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.run_dir / relative, destination)
    return replace(source, run_dir=directory, config=copy.deepcopy(source.config),
                   contract=copy.deepcopy(source.contract), hashes=dict(source.hashes))


def change_checkpoint(source, relative, change):
    path = source.run_dir / relative
    saved = torch.load(path, map_location="cpu", weights_only=True)
    change(saved)
    torch.save(saved, path)
    source.hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()


def debug_cases(amplitude=1.0):
    """Two actual scalar graph fixtures; one has no wedge, one tests epsilon drift."""
    data = [
        ("tree-debug", [(0, 1), (1, 2), (1, 3), (3, 4)],
         [[-4e-4, .4, 1.8], [1e-4, -.2, .1], [3e-4, 1.2, -.6],
          [9e-4, -.8, .4], [8e-4, .1, -.1]]),
        ("edge-debug", [(0, 1)], [[.3, -.6, .8], [.1, .8, -.1], [.7, -.5, .9]]),
    ]
    cases = []
    for graph_id, pairs, values in data:
        x = torch.tensor(values, dtype=torch.float64) * amplitude
        n, r = x.shape
        edges = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        wedges = build_wedges(edges, n)
        p = wedges.shape[1]
        b, a = torch.zeros(len(pairs), n).double(), torch.zeros(p, n).double()
        for index, (u, v) in enumerate(pairs):
            b[index, u], b[index, v] = -1, 1
        for index, (i, j, k) in enumerate(wedges.t().tolist()):
            a[index, i], a[index, j], a[index, k] = 1, -2, 1
        selected_pairs = list(combinations(range(len(pairs)), 2))[:p]
        pair_edges = torch.tensor(selected_pairs, dtype=torch.long).reshape(-1, 2).t()
        coefficients = torch.ones(2, p).double()
        coefficients[1, ::2] = -1
        if p:
            pair_a = (-coefficients[0, :, None] * b[pair_edges[0]]
                      + coefficients[1, :, None] * b[pair_edges[1]])
            coefficients *= (6**.5 / pair_a.norm(dim=1))[None]
        g1, g2 = x[wedges[1]] - x[wedges[0]], x[wedges[2]] - x[wedges[1]]
        c = teacher_weights(g1, g2, torch.zeros(p, dtype=torch.long), 1)
        laplacian, fixed = b.t() @ b, a.t() @ a
        cases.append(SimpleNamespace(
            graph_id=graph_id, family="debug", split="id", num_nodes=n,
            features=x, edges=edges, wedges=wedges, pair_edges=pair_edges,
            pair_coefficients=coefficients, teacher_c=c, lx=laplacian @ x,
            l2x=laplacian @ laplacian @ x, qx=fixed @ x,
            target_path=a.t() @ (c * (a @ x)), feature_seed=884,
            metadata={"feature_status": "fresh", "amplitude": amplitude,
                      "fresh_feature_seed": 884}, debug_a=a, debug_r=r,
        ))
    return cases


def debug_packed(amplitude=1.0, device="cpu", dtype=torch.float64):
    cases = debug_cases(amplitude)
    return cases, pack_cases(cases, device=device, dtype=dtype)


def debug_frozen(kind, device="cpu", dtype=torch.float64):
    seeds = [-1] if kind in ("first", "polynomial", "fixed") else [11, 23]
    model = make_model(kind, seeds, hidden=16, tau=1).to(device=device, dtype=dtype)
    with torch.no_grad():
        if hasattr(model, "u"):
            model.u.fill_(.7)
        if hasattr(model, "v"):
            model.v.fill_(.2)
        if hasattr(model, "beta"):
            model.beta.fill_(1.3)
    return model.requires_grad_(False).eval(), seeds


def test_loads_all_authentic_artifacts_without_fit_or_optimizer(completed_source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("frozen loading may not fit or optimize")

    monkeypatch.setattr("research.wedge_propagation.learned.model.fit_baseline", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    models = load_models(completed_source, "cpu", torch.float32)
    conditions = ("first", "polynomial", "fixed", "learned", "random_pair")
    assert set(models) == {(target, condition) for target in ("L", "L2", "path")
                          for condition in conditions}
    for (_target, condition), frozen in models.items():
        assert frozen.seeds == tuple([-1] if condition in ("first", "polynomial", "fixed")
                                     else completed_source.config["model_seeds"])
        assert not any(module.training for module in frozen.model.modules())
        assert not any(parameter.requires_grad for parameter in frozen.model.parameters())
        assert frozen.provenance["refit"] is False
        assert frozen.provenance["checkpoint_reselection"] is False
        path = completed_source.run_dir / frozen.provenance["artifact"]
        assert frozen.provenance["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if condition in ("first", "polynomial", "fixed"):
            values = json.loads(path.read_text())
            for name, value in values.items():
                expected = getattr(frozen.model, name).new_tensor([value])
                torch.testing.assert_close(getattr(frozen.model, name), expected, rtol=0, atol=0)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert frozen.provenance["best_epochs"] == saved["best_epochs"]
            for name, value in saved["state_dict"].items():
                torch.testing.assert_close(frozen.model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("damage,match", [
    ("target", "identity/config"), ("seeds", "identity/config"),
    ("config_hash", "identity/config"), ("source", "source mismatch"),
    ("best_epochs", "validation epochs"), ("scores", "validation scores"),
    ("shape", "state mismatch"), ("nan", "state mismatch"),
    ("extra_parameter", "state keys"),
])
def test_rejects_corrupted_selected_metadata_and_state(completed_source, tmp_path, damage, match):
    source = changed_source(completed_source, tmp_path)
    relative = "checkpoints/path-learned-selected.pt"

    def change(saved):
        if damage == "target":
            saved["target"] = "L"
        elif damage == "seeds":
            saved["seeds"] = list(reversed(saved["seeds"]))
        elif damage == "config_hash":
            saved["config_hash"] = "0" * 64
        elif damage == "source":
            saved["source"]["git"] = "different-source"
        elif damage == "best_epochs":
            saved["best_epochs"] = [completed_source.config["epochs"] + 1] * 2
        elif damage == "scores":
            saved["validation_message_relerr"][0] = float("nan")
        elif damage == "shape":
            saved["state_dict"]["beta"] = saved["state_dict"]["beta"][:1]
        elif damage == "nan":
            saved["state_dict"]["gate.w1"][0, 0, 0] = float("nan")
        else:
            saved["state_dict"]["u"] = torch.ones(2)

    change_checkpoint(source, relative, change)
    with pytest.raises(ValueError, match=match):
        load_models(source, "cpu", torch.float32)


def test_rejects_source_config_code_hash_and_parameter_count(completed_source):
    source = replace(completed_source, config=copy.deepcopy(completed_source.config))
    source.config["epochs"] += 1
    with pytest.raises(ValueError, match="config/contract"):
        load_models(source, "cpu", torch.float32)
    source = replace(completed_source, contract=copy.deepcopy(completed_source.contract))
    name = next(key for key in source.contract["source"]["sha256"]
                if key.replace("\\", "/") == "learned/model.py")
    source.contract["source"]["sha256"][name] = "0" * 64
    with pytest.raises(ValueError, match="mathematical code"):
        load_models(source, "cpu", torch.float32)
    source = replace(completed_source, contract=copy.deepcopy(completed_source.contract))
    source.contract["jobs"][0]["parameters_per_seed"] += 1
    with pytest.raises(ValueError, match="parameter count"):
        load_models(source, "cpu", torch.float32)


def test_rejects_changed_artifact_and_scalar_fit_schema(completed_source, tmp_path):
    source = changed_source(completed_source, tmp_path)
    relative = "L-first-fit.json"
    path = source.run_dir / relative
    path.write_text('{"u": 0.1}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_models(source, "cpu", torch.float32)
    path.write_text('{"u": 0.1, "v": 0.3}', encoding="utf-8")
    source.hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="scalar fit schema"):
        load_models(source, "cpu", torch.float32)


def test_source_sha_path_separators_are_normalized(completed_source, tmp_path):
    source = changed_source(completed_source, tmp_path)
    source.contract["source"]["sha256"] = {
        name.replace("\\", "/"): value
        for name, value in source.contract["source"]["sha256"].items()
    }
    models = load_models(source, "cpu", torch.float32)
    assert len(models) == 15


@pytest.mark.parametrize("kind", ["first", "polynomial", "fixed", "learned", "random_pair"])
def test_scale_audit_is_frozen_batched_and_has_correct_missing_metrics(kind):
    model, seeds = debug_frozen(kind)
    reference, scaled = debug_packed(), debug_packed(3.0)
    before = frozen_fingerprint(model)
    rows = scale_diagnostics(model, reference, scaled, 3.0, seeds, 1e-8)
    assert len(rows) == 2 * len(seeds)
    assert frozen_fingerprint(model) == before
    assert not any(parameter.grad is not None for parameter in model.parameters())
    for row in rows:
        if row["graph_id"] == "edge-debug":
            assert row["student_weight_scale_relerr"] is None
            assert row["teacher_weight_scale_relerr"] is None
        else:
            assert row["teacher_weight_scale_relerr"] > .001
            assert row["teacher_message_scale_equivariance_relerr"] > .001
            if kind in ("first", "polynomial", "fixed"):
                assert row["student_weight_scale_relerr"] is None
                assert row["message_scale_equivariance_relerr"] < 1e-12
            else:
                assert row["student_weight_scale_relerr"] > 0
    if kind in ("learned", "random_pair"):
        with torch.inference_mode():
            _, c = model(scaled[1])
        mask = scaled[1].path_graph == 0
        torch.testing.assert_close(c[:, mask].mean(1), torch.ones(len(seeds), 3).double())


@pytest.mark.parametrize("target", ["L", "L2", "path"])
def test_scale_metrics_match_independent_dense_physics_per_realization(target):
    model, seeds = debug_frozen("learned")
    reference, scaled = debug_packed(), debug_packed(.3)
    rows = scale_diagnostics(model, reference, scaled, .3, seeds, 1e-8, target=target)
    with torch.inference_mode():
        prediction, c = model(reference[1])
        scaled_prediction, scaled_c = model(scaled[1])
    n = 5
    for index, seed in enumerate(seeds):
        row = next(row for row in rows if row["graph_id"] == "tree-debug" and row["seed"] == seed)
        first, second = reference[0][0], scaled[0][0]
        dense_message = first.debug_a.t() @ (c[index, :4] * (first.debug_a @ first.features))
        dense_scaled = second.debug_a.t() @ (scaled_c[index, :4]
                                           * (second.debug_a @ second.features))
        dense_message *= model.beta[index]
        dense_scaled *= model.beta[index]
        torch.testing.assert_close(prediction[index, :n], dense_message, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(scaled_prediction[index, :n], dense_scaled,
                                   rtol=1e-12, atol=1e-12)
        expected = ((dense_scaled / .3 - dense_message).norm(dim=0)
                    / (dense_message.norm(dim=0) + 1e-8)).mean()
        assert row["message_scale_equivariance_relerr"] == pytest.approx(float(expected), abs=1e-12)
        expected_teacher = ((scaled[1].targets[target][:n] / .3
                             - reference[1].targets[target][:n]).norm(dim=0)
                            / (reference[1].targets[target][:n].norm(dim=0) + 1e-8)).mean()
        assert row["teacher_message_scale_equivariance_relerr"] == pytest.approx(
            float(expected_teacher), abs=1e-12
        )
        if target != "path":
            assert row["teacher_weight_scale_relerr"] is None


def test_alpha_one_and_graph_batching_equivalence():
    model, seeds = debug_frozen("learned")
    reference, scaled = debug_packed(), debug_packed(1.0)
    rows = scale_diagnostics(model, reference, scaled, 1.0, seeds, 1e-8)
    for row in rows:
        assert row["message_scale_equivariance_relerr"] == 0
        assert row["teacher_message_scale_equivariance_relerr"] == 0
        assert row["student_weight_scale_relerr"] in (0, None)
    for first, second in zip(reference[0], scaled[0], strict=True):
        ref = ([first], pack_cases([first], "cpu", torch.float64))
        other = ([second], pack_cases([second], "cpu", torch.float64))
        individual = scale_diagnostics(model, ref, other, 1.0, seeds, 1e-8)
        assert individual == [row for row in rows if row["graph_id"] == first.graph_id]


def test_pair_integrity_and_cache_detect_mutation():
    model, seeds = debug_frozen("learned")
    reference, scaled = debug_packed(), debug_packed(3.0)
    scale_diagnostics(model, reference, scaled, 3.0, seeds, 1e-8)
    scaled[1].x[0, 0] += .1
    with pytest.raises(ValueError, match="packed reference/scaled"):
        scale_diagnostics(model, reference, scaled, 3.0, seeds, 1e-8)
    scaled = debug_packed(3.0)
    scaled[0][0].metadata["fresh_feature_seed"] += 1
    with pytest.raises(ValueError, match="paired fresh"):
        scale_diagnostics(model, reference, scaled, 3.0, seeds, 1e-8)
    scaled = debug_packed(3.0)
    scaled[0][0].graph_id = "wrong-graph"
    with pytest.raises(ValueError, match="graph IDs"):
        scale_diagnostics(model, reference, scaled, 3.0, seeds, 1e-8)


def test_fingerprint_covers_weights_eval_and_grad_flags():
    model, _ = debug_frozen("learned")
    original = frozen_fingerprint(model)
    model.train()
    assert frozen_fingerprint(model) != original
    model.eval()
    model.beta.requires_grad_(True)
    assert frozen_fingerprint(model) != original
    model.beta.requires_grad_(False)
    with torch.no_grad():
        model.beta[0] += .1
    assert frozen_fingerprint(model) != original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_scale_diagnostics_match_cpu_and_preserve_fingerprint(completed_source):
    loaded = load_models(completed_source, "cuda", torch.float32)
    assert len(loaded) == 15
    cpu_model, seeds = debug_frozen("random_pair")
    gpu_model, _ = debug_frozen("random_pair", "cuda")
    cpu = scale_diagnostics(cpu_model, debug_packed(), debug_packed(3), 3, seeds, 1e-8)
    before = frozen_fingerprint(gpu_model)
    gpu = scale_diagnostics(gpu_model, debug_packed(device="cuda"),
                            debug_packed(3, "cuda"), 3, seeds, 1e-8)
    assert frozen_fingerprint(gpu_model) == before
    for first, second in zip(cpu, gpu, strict=True):
        for key, value in first.items():
            if isinstance(value, float):
                assert second[key] == pytest.approx(value, abs=1e-10)
            else:
                assert second[key] == value
