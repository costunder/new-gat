"""Independent DEBUG classifier fixtures; never citation performance evidence."""
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from ...local_energy_relations.topology import build_topology
from ..geometry import prepare_geometry
from .common import CONDITIONS
from .model import PackedClassifier


def fixture(device="cpu", dtype=torch.float64):
    pairs = np.asarray([(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(4,5)], dtype=np.int64).T
    topology = build_topology(7, pairs)
    geometry = {r: prepare_geometry(topology,r).to(device,dtype) for r in ("unit","local_degree")}
    edges = torch.as_tensor(pairs, device=device)
    loops = torch.arange(7, device=device).repeat(2,1)
    gc = torch.cat((edges,edges.flip(0),loops),1)
    degree = torch.bincount(gc[1],minlength=7).to(dtype)
    return SimpleNamespace(
        name="DEBUG-independent-classifier",num_nodes=7,num_classes=3,
        x=torch.sin(torch.arange(42,device=device,dtype=dtype)).reshape(7,6),
        y=torch.tensor([0,1,2,0,1,2,0],device=device),
        gcn_edges=gc,gcn_weights=(degree[gc[0]]*degree[gc[1]]).rsqrt(),
        geometry_for=lambda r:geometry[r],
    )


def run_checks(device="cpu", include_core=True):
    device=torch.device(device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested DEBUG CUDA unavailable; no fallback")
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        core=None
        if include_core:
            from ..verify import run_checks as core_checks
            core=core_checks(device)
        graph=fixture(device)
        records=[]
        for condition in CONDITIONS:
            model=PackedClassifier(condition,6,3,(11,23),hidden=64,path_chunk=7).to(device=device,dtype=torch.float64)
            optimizer=torch.optim.Adam(model.weight_decay_groups(.0005),lr=.01)
            initial={n:p.detach().clone() for n,p in model.named_parameters()}
            # Output-zero initialization makes hidden generator gradients zero at
            # update 1. Three fixture updates verify the complete active path.
            for epoch in range(3):
                model.train(); optimizer.zero_grad(set_to_none=True)
                logits,_=model(graph,epoch=epoch)
                ce=F.cross_entropy(logits.flatten(0,1),graph.y.repeat(2),reduction="none").reshape(2,-1).mean(1)
                ce.sum().backward()
                for name,p in model.named_parameters():
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise AssertionError(f"Disconnected/nonfinite DEBUG classifier parameter: {condition}/{name}")
                optimizer.step()
            changed={n:float((p.detach()-initial[n]).abs().max().cpu()) for n,p in model.named_parameters()}
            if any(v<=0 for v in changed.values()):
                raise AssertionError(f"Active DEBUG parameter did not change: {condition}/{changed}")
            model.eval()
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            with torch.no_grad():
                original,_=model(graph,diagnostics=True)
                noop,_=model(graph,diagnostics=True,intervention="no_op")
            if not torch.equal(original,noop) or not torch.isfinite(original).all():
                raise AssertionError("DEBUG no-op or finite-logit check failed")
            if any(not torch.equal(p,before[n]) for n,p in model.named_parameters()):
                raise AssertionError("DEBUG frozen forward modified parameters")
            records.append({"condition":condition,"layers":2,"hidden":64,"seeds":2,
                            "parameters_per_seed":model.parameters_per_seed,
                            "fixture_optimizer_updates_per_seed":3,"all_active_parameters_changed":True,
                            "no_op_exact":True,"final_fixture_ce":ce.detach().cpu().tolist()})
        return {"scope":"DEBUG independent fixtures; no real classifier training",
                "passed":True,"device":str(device),"core":core,"classifiers":records}
    finally:
        torch.set_num_threads(previous)
