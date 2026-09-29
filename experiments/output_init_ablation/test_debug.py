"""Synthetic debug checks; production model shape retained for the ablation."""

import copy
import json
import math

import pytest
import torch
from torch.nn import functional as F

from experiments.c_learning_bracket import model as accepted_model
from experiments.c_learning_bracket.inspect_c import UpdateInspection
from experiments.c_learning_bracket.test_debug import (
    arguments as accepted_arguments,
)
from experiments.c_learning_bracket.test_debug import (
    cpu_threads,
    debug_payload,
    fixture_batch,
)

from .model import make_model, non_output_digest
from .study import command, compare_results, signal_trajectory
from .train import evaluate_selected, parser, research_contract, train_one, validate

# Imported pytest fixture explicitly applies the same CPU thread cap.
__all__ = ["cpu_threads"]


def arguments(initialization):
    args = accepted_arguments()
    args.output_initialization = initialization
    return args


@pytest.mark.parametrize("condition", ["learned", "fixed"])
def test_only_output_weights_differ_and_random_state_is_identical(condition):
    payload = debug_payload()
    a = make_model(payload, arguments("baseline"), "cpu", condition)
    rng = torch.get_rng_state()
    b = make_model(payload, arguments("kaiming_relu"), "cpu", condition)
    assert torch.equal(rng, torch.get_rng_state())
    assert non_output_digest(a) == non_output_digest(b)
    changes = []
    for name, value in a.state_dict().items():
        other = b.state_dict()[name]
        if name.endswith("output_projection.weight"):
            torch.testing.assert_close(other, value * math.sqrt(6), rtol=0, atol=0)
            assert other.abs().max() <= math.sqrt(6 / 256)
            changes.append(name)
        else:
            assert torch.equal(value, other), name
    assert len(changes) == 8
    assert b.contract()["output_initialization"] == "kaiming_relu"


@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("condition", ["learned", "fixed"])
def test_baseline_matches_accepted_model_forward_and_gradients(checkpoint, condition):
    batch = fixture_batch()
    payload = {"graphs": [{"x": batch.graph.x}], "classes": 3}
    args = arguments("baseline")
    args.activation_checkpoint = checkpoint
    old = accepted_model.make_model(payload, args, "cpu", condition)
    new = make_model(payload, args, "cpu", condition)
    torch.manual_seed(111)
    logits = old(batch.graph)
    F.cross_entropy(logits, batch.graph.y).backward()
    rng = torch.get_rng_state()
    torch.manual_seed(111)
    other = new(batch.graph)
    F.cross_entropy(other, batch.graph.y).backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(logits, other, rtol=0, atol=0)
    for a, b in zip(old.parameters(), new.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_first_layer_input_c_value_mixing_unchanged_only_output_scales():
    batch = fixture_batch()
    payload = {"graphs": [{"x": batch.graph.x}], "classes": 3}
    records = []
    for initialization in ("baseline", "kaiming_relu"):
        model = make_model(payload, arguments(initialization), "cpu").eval()
        observer = UpdateInspection(model, 2)
        with torch.no_grad():
            model(batch.graph, observer=observer)
        records.append(observer.layers[0])
    a, b = records
    torch.testing.assert_close(a["c"], b["c"], rtol=0, atol=0)
    for key in ("input_h", "value_projection", "neighbor_mixing"):
        assert a["signal_stages"][key] == b["signal_stages"][key]
    for key in ("output_projection", "relu", "dropout_next_h"):
        assert b["signal_stages"][key]["rms"] / a["signal_stages"][key]["rms"] == pytest.approx(
            math.sqrt(6), rel=2e-6
        )


def test_contract_records_the_only_factor_and_production_budget(tmp_path):
    argv = command("kaiming_relu", "calibrate", tmp_path / "unused", tmp_path)[4:]
    args = parser().parse_args(argv)
    validate(args)
    assert (args.layers, args.hidden_channels, args.heads, args.epochs) == (8, 256, 8, 200)
    a = research_contract(args, {})
    args.output_initialization = "baseline"
    b = research_contract(args, {})
    assert a["configuration"].pop("output_initialization") == "kaiming_relu"
    assert b["configuration"].pop("output_initialization") == "baseline"
    assert a == b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration smoke")
def test_debug_four_conditions_train_save_reload_evaluate_and_compare(tmp_path):
    # Explicit synthetic debug only. Both arms keep full 8/256/8 architecture.
    device = torch.device("cuda")
    payload = debug_payload()
    for initialization in ("baseline", "kaiming_relu"):
        args = arguments(initialization)
        args.epochs = 2
        args.sample_seed_batch_size = 64
        args.sample_context_seed_batch_size = 32
        args.eval_context_seeds = [16, 32]
        args.pin_memory = False
        args.sample_prefetch = False
        folder = tmp_path / ("train-" + initialization)
        folder.mkdir()
        learned = train_one(payload, args, "learned", folder, device)
        fixed = train_one(payload, args, "fixed", folder, device, learned)
        evaluate_selected(payload, args, {"learned": learned, "fixed": fixed}, folder, device)
    result = compare_results(tmp_path, expected_epochs=2)
    assert result["complete"] and result["pairing_verified"]
    trajectory = signal_trajectory(tmp_path, expected_epochs=2)
    assert len(trajectory["observations"]) == 8
    assert all(len(row["layers"]) == 8 for row in trajectory["observations"])
    (tmp_path / "debug-comparison.json").write_text(json.dumps(result, indent=2))
    (tmp_path / "debug-signal-trajectory.json").write_text(json.dumps(trajectory, indent=2))
    for arm in result["conditions"].values():
        for condition in arm.values():
            assert condition["full_validation"]["seed_coverage"]["unique_count"] == 16
    with pytest.raises(ValueError, match="every declared epoch"):
        compare_results(tmp_path)
    # Saved comparison inputs must catch a broken cross-arm pairing.
    path = tmp_path / "train-kaiming_relu/learned-initial.json"
    saved = json.loads(path.read_text())
    altered = copy.deepcopy(saved)
    altered["initial_non_output_sha256"] = "deliberately-corrupt-debug-hash"
    original_bytes = path.read_bytes()
    try:
        path.write_text(json.dumps(altered))
        with pytest.raises(RuntimeError, match="non-output"):
            compare_results(tmp_path, expected_epochs=2)
    finally:
        path.write_bytes(original_bytes)
