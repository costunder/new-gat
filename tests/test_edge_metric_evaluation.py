"""Explicit DEBUG fixtures for locked metrics, probes and statistical reports."""
from copy import deepcopy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import ttest_1samp

from research.edge_metric_relations.classification.common import CONDITIONS, condition_metadata, digest, read_config
from research.edge_metric_relations.classification.evaluation import frozen_evaluate, intervention_scopes, intervention_variants
from research.edge_metric_relations.classification.model import PackedClassifier
from research.edge_metric_relations.classification.report import (
    branch_estimates, comparisons, expected_parameters, holm, intervention_changes,
    metric_estimates, paired_comparisons, paired_test, validate_rows, write_report,
)
from research.edge_metric_relations.geometry import prepare_geometry
from research.local_energy_relations.topology import build_topology
from research.wedge_propagation.classification.evaluation import model_state_hash


@pytest.fixture(autouse=True, scope="module")
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def graph(name="DEBUG-Cora", zero=False):
    config = read_config(profile="debug")
    shape = config["data"]["expected_shapes"][name]
    n = shape["nodes"]
    # Complete DEBUG fixture: every node retained, with path edges and triangles.
    pairs = [(i,i+1) for i in range(n-1)] + [(0,2),(3,5)]
    edges = torch.tensor(pairs,dtype=torch.long).T
    top = build_topology(n,edges)
    geometries = {r:prepare_geometry(top,r) for r in ("unit","local_degree")}
    gcn_edges = torch.cat((edges,edges.flip(0),torch.arange(n).repeat(2,1)),1)
    degree = torch.bincount(gcn_edges[1],minlength=n).double()
    weights = (degree[gcn_edges[0]]*degree[gcn_edges[1]]).rsqrt()
    x = torch.randn((n,shape["features"]),dtype=torch.float64,generator=torch.Generator().manual_seed(118)).relu()
    x = x / x.sum(-1,keepdim=True).clamp_min(1.)
    if zero:
        x.zero_()
    masks = {}
    offset = 0
    for split,attribute in (("train","train_mask"),("validation","val_mask"),("test","test_mask")):
        mask=torch.zeros(n,dtype=torch.bool); mask[offset:offset+shape[split]]=True
        masks[attribute]=mask; offset+=shape[split]
    result = SimpleNamespace(name=name,x=x,y=torch.arange(n)%shape["classes"],edges=edges,
                             gcn_edges=gcn_edges,gcn_weights=weights,num_nodes=n,num_features=shape["features"],
                             num_classes=shape["classes"],geometries=geometries,**masks)
    result.geometry_for=lambda recipe: geometries[recipe]
    return result


def model(data,condition,seeds=(11,23)):
    config=read_config(profile="debug")
    result=PackedClassifier(condition,data.num_features,data.num_classes,seeds,hidden=config["backbone"]["hidden_dim"],
                            dropout=.5,dataset_name=data.name,path_chunk=64,checkpoint_paths=True).double()
    with torch.no_grad():
        for layer,g in enumerate(result.gates):
            for part in (g.diagonal_gate,g.pair_gate):
                if part is not None:
                    part.w2.fill_(.02*(layer+1)); part.b2.fill_(.2)
    return result


@pytest.fixture(scope="module")
def report_fixture():
    config=read_config(profile="debug")
    evaluated={"metric_rows":[],"intervention_rows":[],"branch_rows":[],"provenance":[]}
    counts, selections = {}, []
    for dataset in config["data"]["datasets"]:
        data=graph(dataset); counts[dataset]={}
        for condition in CONDITIONS:
            instance=model(data,condition,tuple(config["training"]["final_seeds"]))
            result=frozen_evaluate(instance,data,instance.seeds,config)
            for field in ("metric_rows","intervention_rows","branch_rows"):
                evaluated[field].extend(result[field])
            evaluated["provenance"].append(result["provenance"])
            counts[dataset][condition]=instance.parameters_per_seed
            selections.append({"dataset":dataset,"condition":condition,**condition_metadata(condition),
                               "selected_lr":config["training"]["learning_rate_candidates"][0],
                               "mean_tuning_validation_ce":1.,"selection_scope":"validation_only_independent_tuning_seeds"})
    t=config["training"]
    # These are metadata fixtures for validation, not claims of executed training.
    hashes={"DEBUG_METADATA_FIXTURE_ONLY":"a"*64}
    completion={"profile":"debug","actual_data":False,"config_digest":digest(config),
                "source":{"sha256":hashes,"code_digest":digest(hashes)},"data_manifest_digest":"b"*64,
                "parameter_counts":counts,"coverage":{"tuning_runs":t["tuning_runs"],"final_runs":t["final_runs"],
                    "total_runs":t["total_runs"],"contract_optimizer_updates":t["total_updates"],
                    "all_datasets_conditions_seeds_splits":True}}
    resources=[{"status":"measured","seconds_per_epoch":.01,"scope":"DEBUG_report_metadata_fixture"}]
    return config,evaluated,selections,resources,completion


@pytest.mark.parametrize("condition",CONDITIONS)
def test_frozen_complete_condition_contract_and_state_preservation(condition):
    config,data=read_config(profile="debug"),graph()
    instance=model(data,condition); instance.train()
    before=model_state_hash(instance)
    evaluated=frozen_evaluate(instance,data,instance.seeds,config)
    variants=intervention_variants(instance.variant)
    assert len(evaluated["metric_rows"])==len(instance.seeds)*3
    assert len(evaluated["intervention_rows"])==len(instance.seeds)*3*3*len(variants)
    assert len(evaluated["branch_rows"])==len(instance.seeds)*2*(1+3*len(variants))
    assert instance.training and model_state_hash(instance)==before
    proof=evaluated["provenance"]
    assert proof["before_sha256"]==proof["after_sha256"]==before
    assert proof["optimizer_updates"]==0 and proof["no_op_checks"]==3
    for row in evaluated["intervention_rows"]:
        assert row["model_state_sha256"]==before
        if row["intervention"]=="no_op":
            assert row["logit_delta_norm"]==0 and row["prediction_flip_fraction"]==0
    for row in evaluated["branch_rows"]:
        if instance.operator_family=="baseline":
            assert row["energy_total"] is None and row["matched_delta_norm"] is None
        else:
            assert row["energy_total"]==pytest.approx(row["energy_intra"]+row["energy_cross"])


def test_full_counts_without_running_scientific_training():
    config=read_config(profile="full")
    cells=len(config["data"]["datasets"])*len(config["training"]["final_seeds"])
    original=cells*len(CONDITIONS)*3
    total=sum(len(intervention_variants(condition.split("__")[-1])) for condition in CONDITIONS)
    probes=cells*total*3*3
    branches=cells*(len(CONDITIONS)*2+total*3*2)
    assert (original,probes,branches)==(675,3915,3060)
    assert all(intervention_scopes(c)==("layer_0","layer_1","both") for c in CONDITIONS)
    assert sum(primary for _,_,_,primary in comparisons())==6


def test_zero_norm_ratios_remain_nullable():
    config,data=read_config(profile="debug"),graph(zero=True)
    instance=model(data,"unit__F2")
    result=frozen_evaluate(instance,data,instance.seeds,config)
    assert all(row["off_nonzero"] is False and row["matched_delta_relative"] is None for row in result["branch_rows"])
    assert all(row["logit_delta_relative"] is None for row in result["intervention_rows"])


def test_layer0_frozen_changes_recompute_layer1_and_keep_noop_equal():
    config,data=read_config(profile="debug"),graph()
    instance=model(data,"unit__F2"); instance.eval()
    original,details=instance(data,diagnostics=True)
    changed,newdetails=instance(data,diagnostics=True,intervention="offdiag_zero",intervention_layers=(0,))
    assert not torch.equal(changed,original)
    assert not torch.equal(details[1]["projected_norm"],newdetails[1]["projected_norm"])
    unchanged,_=instance(data,intervention="no_op",intervention_layers=(0,))
    torch.testing.assert_close(unchanged,original,atol=0,rtol=0)


def test_evaluation_restores_mode_when_invalid_forward_fails():
    config,data=read_config(profile="debug"),graph()
    instance=model(data,"unit__D0"); instance.train()
    data.test_mask.zero_()
    with pytest.raises(ValueError,match="empty final split"):
        frozen_evaluate(instance,data,instance.seeds,config)
    assert instance.training


def test_paired_t_and_holm_match_known_math():
    differences=[.2,-.1,.7,.4,.3]
    result=paired_test(differences)
    reference=ttest_1samp(differences,0)
    assert result["t_statistic"]==pytest.approx(reference.statistic)
    assert result["p_value"]==pytest.approx(reference.pvalue)
    assert holm([.01,.04,.03,.9])==pytest.approx([.04,.09,.09,.9])
    assert paired_test([0.,0.])["p_value"]==1
    assert paired_test([.2,.2])["p_value"]==0


def test_report_full_debug_coverage_statistics_and_primary_family(report_fixture):
    config,evaluated,selections,resources,completion=report_fixture
    checks=validate_rows(config,evaluated,selections,resources,completion)
    assert checks["primary_metric_rows"]==len(evaluated["metric_rows"])
    paired=paired_comparisons(evaluated["metric_rows"])
    primary=[row for row in paired if row["primary_accuracy_test"]]
    assert len(primary)==18
    assert {row["contrast"] for row in primary}=={"F2-D1","F2-DA","F2-P2"}
    assert all(row["split"]=="test" and row["metric"]=="accuracy" for row in primary)
    assert all(row["p_holm"] is not None for row in primary)
    assert all(row["p_holm"] is None for row in paired if not row["primary_accuracy_test"])
    assert len(metric_estimates(evaluated["metric_rows"]))==15*3*3
    branches=branch_estimates(evaluated["branch_rows"])
    assert all(r["energy_total_count"]==0 for r in branches if r["operator_family"]=="baseline")
    changes=intervention_changes(evaluated["metric_rows"],evaluated["intervention_rows"])
    assert all(r["mean"]==0 for r in changes if r["no_op_expected"])


@pytest.mark.parametrize("fault",["missing_metric","missing_probe","missing_branch","bad_metadata","bad_hash","missing_noop","bad_counts","baseline_fake_energy","bad_ratio","wrong_primary"])
def test_report_rejects_incomplete_or_mislabelled_evidence(report_fixture,fault):
    config,evaluated,selections,resources,completion=deepcopy(report_fixture)
    if fault=="missing_metric": evaluated["metric_rows"].pop()
    elif fault=="missing_probe": evaluated["intervention_rows"].pop()
    elif fault=="missing_branch": evaluated["branch_rows"].pop()
    elif fault=="bad_metadata": evaluated["metric_rows"][0]["learned_pair"]=True
    elif fault=="bad_hash": evaluated["provenance"][0]["after_sha256"]="c"*64
    elif fault=="missing_noop": evaluated["provenance"][0]["no_op_checks"]=2
    elif fault=="bad_counts": completion["parameter_counts"][config["data"]["datasets"][0]]["unit__F2"]=0
    elif fault=="baseline_fake_energy": next(r for r in evaluated["branch_rows"] if r["operator_family"]=="baseline")["energy_total"]=0.
    elif fault=="bad_ratio": next(r for r in evaluated["branch_rows"] if r["off_nonzero"])["matched_delta_relative"]=99.
    elif fault=="wrong_primary": config["evaluation"]["primary_contrasts"][2]="F2-F1"
    with pytest.raises(ValueError): validate_rows(config,evaluated,selections,resources,completion)


def test_report_generates_csv_summary_and_scientific_figures_exclusively(report_fixture,tmp_path):
    config,evaluated,selections,resources,completion=report_fixture
    checks=write_report(tmp_path,config,evaluated,selections,resources,completion)
    assert checks["primary_holm_tests"]==18
    summary=(tmp_path/"EDGE_METRIC_CLASSIFICATION_SUMMARY.md").read_text(encoding="utf-8")
    assert "Holm" in summary and "DEBUG" in summary and "315,000" not in summary
    for name in ("primary_contrasts","classification","branch_use"):
        for suffix in ("png","pdf"):
            assert (tmp_path/"figures"/f"{name}.{suffix}").stat().st_size>1000
    with pytest.raises(FileExistsError): write_report(tmp_path,config,evaluated,selections,resources,completion)
