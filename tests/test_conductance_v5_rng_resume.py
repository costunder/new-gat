"""Checkpoint/RNG unit regressions; these are not production training runs."""

from __future__ import annotations

import inspect

import pytest
import torch

from research.conductance_gat.v5 import train


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def _unit_model(device):
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(8, 2),
    ).to(device)


def _unit_step(model, optimizer, device):
    optimizer.zero_grad(set_to_none=True)
    # CPU sampling and model-device dropout exercise both RNG streams on CUDA.
    inputs = torch.randn(7, 4).to(device)
    target = torch.randn(7, 2).to(device)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    return loss.detach().cpu()


def _all_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _all_tensors(child)


def _assert_resumed_step_matches_uninterrupted(tmp_path, device):
    torch.manual_seed(71)
    model = _unit_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    _unit_step(model, optimizer, device)
    path = tmp_path / "last.pt"
    train._save(
        path,
        {
            "schema_version": 3,
            "epoch": 1,
            "history": [{"epoch": 1, "scope": "unit_test"}],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
            ),
        },
    )
    checkpoint_bytes = path.read_bytes()
    expected_loss = _unit_step(model, optimizer, device)
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_cpu_draw = torch.rand(11)
    expected_cuda_draw = torch.rand(11, device=device) if device.type == "cuda" else None

    resumed = _unit_model(device)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.005)
    saved = train.load_checkpoint_on_cpu(path)
    assert all(value.device.type == "cpu" for value in _all_tensors(saved))
    assert saved["epoch"] == 1 and saved["history"][0]["epoch"] == 1
    resumed.load_state_dict(saved["model_state"])
    resumed_optimizer.load_state_dict(saved["optimizer_state"])
    for parameter, state in resumed_optimizer.state.items():
        # The non-capturable Adam step scalar stays on CPU; moments follow params.
        assert state["exp_avg"].device == parameter.device
        assert state["exp_avg_sq"].device == parameter.device
        assert state["step"].device.type == "cpu"
    train.restore_checkpoint_rng(saved, device)
    del saved
    actual_loss = _unit_step(resumed, resumed_optimizer, device)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for name, actual in resumed.state_dict().items():
        torch.testing.assert_close(actual, expected_parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(11), expected_cpu_draw, rtol=0, atol=0)
    if expected_cuda_draw is not None:
        torch.testing.assert_close(
            torch.rand(11, device=device), expected_cuda_draw, rtol=0, atol=0
        )
    assert path.read_bytes() == checkpoint_bytes


def test_cpu_checkpoint_resume_preserves_next_dropout_and_adam_step(tmp_path, monkeypatch):
    calls = []

    def capture_cuda_state(state, device):
        assert state.device.type == "cpu" and state.dtype == torch.uint8 and state.ndim == 1
        calls.append(device)

    monkeypatch.setattr(torch.cuda, "set_rng_state", capture_cuda_state)
    _assert_resumed_step_matches_uninterrupted(tmp_path, torch.device("cpu"))
    assert calls == [torch.device("cpu")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an actual CUDA GPU")
def test_cuda_checkpoint_resume_stages_on_cpu_and_restores_both_rngs(tmp_path):
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[device.index]):
        _assert_resumed_step_matches_uninterrupted(tmp_path, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an actual CUDA GPU")
def test_rng_restore_normalizes_accidentally_cuda_mapped_byte_states():
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[device.index]):
        states = {
            "cpu_rng_state": torch.get_rng_state().to(device),
            "cuda_rng_state": torch.cuda.get_rng_state(device).to(device),
        }
        expected_cpu, expected_cuda = torch.rand(9), torch.rand(9, device=device)
        train.restore_checkpoint_rng(states, device)
        assert torch.equal(torch.rand(9), expected_cpu)
        assert torch.equal(torch.rand(9, device=device), expected_cuda)


@pytest.mark.parametrize("name", ["cpu_rng_state", "cuda_rng_state"])
@pytest.mark.parametrize(
    "invalid", [None, [1, 2], torch.ones(3), torch.zeros((2, 3), dtype=torch.uint8)]
)
def test_rng_restore_rejects_invalid_bytes_before_changing_generators(name, invalid, monkeypatch):
    states = {"cpu_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.get_rng_state()}
    states[name] = invalid
    monkeypatch.setattr(torch, "set_rng_state", lambda *_: pytest.fail("must validate first"))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *_: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match=name):
        train.restore_checkpoint_rng(states, torch.device("cuda:0"))


def test_checkpoint_loader_explicitly_uses_cpu_and_rejects_nonmapping(tmp_path, monkeypatch):
    calls = []

    def fake_load(path, *, map_location, weights_only):
        calls.append((path, map_location, weights_only))
        return []

    monkeypatch.setattr(torch, "load", fake_load)
    path = tmp_path / "last.pt"
    with pytest.raises(ValueError, match="checkpoint payload"):
        train.load_checkpoint_on_cpu(path)
    assert calls == [(path, "cpu", False)]


def test_training_path_uses_cpu_staging_and_releases_checkpoint_references():
    source = inspect.getsource(train._train_model_impl)
    assert "saved = load_checkpoint_on_cpu(last_path)" in source
    assert "restore_checkpoint_rng(saved, device)" in source
    assert "del saved" in source
    assert "selected = load_checkpoint_on_cpu(checkpoint)" in source
    assert "del selected" in source
    assert "map_location=device" not in source
