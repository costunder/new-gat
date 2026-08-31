"""Execution configuration tests; no research training or CUDA compilation."""

import argparse
from unittest.mock import Mock

import pytest
import torch

from chartgat.execution import add_execution_arguments, configure_execution


def test_compile_is_explicit_and_eager_does_not_wrap_model(monkeypatch):
    parser = argparse.ArgumentParser()
    add_execution_arguments(parser)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    compiler = Mock()
    monkeypatch.setattr(torch, "compile", compiler)
    before = tuple(model.state_dict())
    report = configure_execution(model, parser.parse_args([]), torch.device("cpu"))
    assert report["backend"] == "eager"
    assert tuple(model.state_dict()) == before
    compiler.assert_not_called()
    assert parser.parse_args(["--compile", "--no-compile"]).compile is False


def test_compile_targets_forward_without_changing_state_keys(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    original_forward = model.forward
    compiled_forward = Mock(wraps=original_forward)
    compiler = Mock(return_value=compiled_forward)
    monkeypatch.setattr(torch, "compile", compiler)
    report = configure_execution(model, argparse.Namespace(compile=True), torch.device("cuda"))
    compiler.assert_called_once_with(original_forward, backend="inductor", dynamic=True)
    assert model.forward is compiled_forward
    assert report["torch_compile"] is True
    assert list(model.state_dict()) == ["0.weight", "0.bias"]
    assert report["compiled_modules"] == ["<root>"]
    assert report["scope"] == "tensor_mlp_blocks"


def test_compile_refuses_cpu():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        configure_execution(torch.nn.Linear(2, 1), argparse.Namespace(compile=True), "cpu")


def test_compiler_errors_are_not_silently_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    monkeypatch.setattr(torch, "compile", Mock(side_effect=RuntimeError("compiler missing")))
    with pytest.raises(RuntimeError, match="compiler missing"):
        configure_execution(model, argparse.Namespace(compile=True), "cuda")
