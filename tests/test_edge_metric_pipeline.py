"""DEBUG phase ordering and strict predecessor coverage/provenance guards."""
from argparse import Namespace
import json

import pytest

from research.edge_metric_relations.predecessors import check_predecessors,assert_predecessors_unchanged
from research.edge_metric_relations.study import command


def records(tmp_path):
    a,b=tmp_path/"A",tmp_path/"B"
    for folder,experiment,fields in ((a,"edge_metric_relations_audit",{"graphs":201}),
                                      (b,"B_raw_message_recovery",{"graphs":531,"learned_runs":240,"seed_optimizer_updates":120000})):
        folder.mkdir()
        complete={"completed":True,"profile":"full","experiment":experiment,"math_checks_passed":True,"source_digest":"DEBUG-hash",**fields}
        (folder/"completion.json").write_text(json.dumps(complete),encoding="utf-8")
        (folder/"source_manifest.json").write_text(json.dumps({"code_digest":"DEBUG-hash"}),encoding="utf-8")
    return a,b


def test_full_predecessors_accept_complete_math_coverage_not_score(tmp_path):
    a,b=records(tmp_path)
    record=check_predecessors(a,b,"full","DEBUG-hash")
    assert all(row["score_based_pass_gate"] is False for row in record.values())
    assert_predecessors_unchanged(record)
    (a/"completion.json").write_text("{}",encoding="utf-8")
    with pytest.raises(RuntimeError):assert_predecessors_unchanged(record)


@pytest.mark.parametrize("field,value",[("graphs",530),("learned_runs",239),("seed_optimizer_updates",119999),("math_checks_passed",False),("source_digest","other")])
def test_missing_or_changed_b_contract_rejected(tmp_path,field,value):
    a,b=records(tmp_path);p=b/"completion.json";record=json.loads(p.read_text());record[field]=value;p.write_text(json.dumps(record))
    with pytest.raises(ValueError):check_predecessors(a,b,"full","DEBUG-hash")


def test_pipeline_c_uses_completed_a_b_and_original_source(tmp_path):
    args=Namespace(profile="full",device="cuda",source_dir=str(tmp_path/"original"),data_root="data",
                   audit_dir=tmp_path/"A",synthetic_dir=tmp_path/"B",offline=True)
    cmd=command("C",args,tmp_path/"C")
    assert "research.edge_metric_relations.classification.study" in cmd
    assert cmd[cmd.index("--mechanism-audit-dir")+1]==str(args.audit_dir)
    assert cmd[cmd.index("--synthetic-dir")+1]==str(args.synthetic_dir)
    assert "--offline" in cmd
