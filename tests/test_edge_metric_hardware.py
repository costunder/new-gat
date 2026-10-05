"""DEBUG allocation policy checks; no full training or physical MIG claim."""
from argparse import Namespace
from dataclasses import dataclass

import pytest
import torch

from research.edge_metric_relations import hardware
from research.edge_metric_relations.classification.common import read_config, validate_config
from research.edge_metric_relations.classification import study as classification
from research.edge_metric_relations.study import command


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_mig_policy_preserves_complete_scientific_contract(profile):
    original = read_config(profile=profile)
    changed = hardware.configure_classification_runtime(original, "a100-mig-10gb")
    assert validate_config(changed, profile) is changed
    for field in original:
        if field != "runtime":
            assert changed[field] == original[field]
    assert changed["runtime"]["activation_checkpointing"] is True
    assert set(original["runtime"]["relation_chunk_candidates"]) <= set(changed["runtime"]["relation_chunk_candidates"])
    assert {128, 512} <= set(changed["runtime"]["relation_chunk_candidates"])
    assert changed["training"]["total_updates"] == original["training"]["total_updates"]
    assert hardware.configure_classification_runtime(original, "auto") == original


def test_memory_budget_uses_assigned_instance_free_capacity_and_reserved_peak(monkeypatch):
    gib = 1 << 30
    # A nominal 10GB instance with 8GiB currently free and some live allocations.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8*gib, 10*gib))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2*gib)
    window = hardware.memory_window("cuda", hardware.get_policy("a100-mig-10gb"))
    assert window["total_bytes"] == 10*gib
    assert window["allocation_budget_bytes"] == int(8*gib*.70)
    assert hardware.headroom_safe(gib+window["allocation_budget_bytes"], 2*gib+window["allocation_budget_bytes"], window)
    assert not hardware.headroom_safe(gib, window["reserved_ceiling_bytes"]+1, window)
    assert not hardware.headroom_safe(window["allocated_ceiling_bytes"]+1, 2*gib, window)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (gib//2, 10*gib))
    assert hardware.memory_window("cuda", hardware.get_policy("a100-mig-10gb"))["allocation_budget_bytes"] == 0


def test_unique_storage_accounting_handles_graph_dataclasses_and_recipe_sharing():
    @dataclass
    class Graph:
        x: torch.Tensor
        geometry: dict
    values = torch.arange(100, dtype=torch.float32)
    graph = Graph(values, {"unit": values[:50], "local_degree": values.reshape(10, 10)})
    assert hardware.unique_tensor_bytes(graph) == values.untyped_storage().nbytes()
    extra = values.clone()
    assert hardware.unique_tensor_bytes([graph, extra]) == 2*values.untyped_storage().nbytes()


@pytest.mark.parametrize("phase", ("A", "B", "C"))
def test_every_pipeline_phase_receives_mig_allocation_policy(tmp_path, phase):
    args = Namespace(profile="full", device="cuda", hardware_profile="a100-mig-10gb",
                     source_dir=str(tmp_path/"original"), data_root="data",
                     audit_dir=tmp_path/"A", synthetic_dir=tmp_path/"B", offline=True)
    cmd = command(phase, args, tmp_path/phase)
    assert cmd[cmd.index("--hardware-profile")+1] == "a100-mig-10gb"
    assert cmd[cmd.index("--profile")+1] == "full"


def test_classification_worker_plan_retains_mig_policy(tmp_path, monkeypatch):
    # Dispatch metadata fixture; no model training is asserted here.
    from research.edge_metric_relations.common import read_json, write_json
    output = tmp_path/"DEBUG-dispatch"; output.mkdir(); (output/"plans").mkdir()
    config = read_config(profile="debug")
    jobs = [{"dataset": "DEBUG-Cora", "condition": "unit__F2"}]
    graphs = {"DEBUG-Cora": {"num_local_edges": 40, "num_cross_edges": 100, "num_local_copies": 80}}
    seen = []
    def fake_worker(path):
        plan = read_json(path); seen.append(plan)
        folder = output/"workers"/"tuning-0"; folder.mkdir(parents=True)
        write_json(folder/"results.json", {key: [] for key in (
            "selection_rows", "resource_rows", "metric_rows", "intervention_rows", "branch_rows", "provenance")})
    monkeypatch.setattr(classification, "_worker", fake_worker)
    classification._dispatch(output, config, {}, graphs, jobs, ["cpu"], "tuning",
                             hardware_profile="a100-mig-10gb")
    assert len(seen) == 1 and seen[0]["hardware_profile"] == "a100-mig-10gb"
    assert seen[0]["jobs"] == jobs


def test_unknown_hardware_policy_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown hardware"):
        hardware.get_policy("small-model-fallback")
