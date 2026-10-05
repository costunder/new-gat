"""Verify completed predecessor phases without interpreting a score as success."""
from pathlib import Path

from .common import file_sha256,read_json


def check_predecessors(audit_dir,synthetic_dir,profile,source_digest):
    if not audit_dir or not synthetic_dir:
        raise ValueError("A and B completion paths are required before classifier training")
    records={}
    for name,folder,experiment in (("A",audit_dir,"edge_metric_relations_audit"),
                                   ("B",synthetic_dir,"B_raw_message_recovery")):
        path=Path(folder).resolve()
        complete=read_json(path/"completion.json")
        provenance=read_json(path/"source_manifest.json")
        if complete.get("completed") is not True or complete.get("profile")!=profile or complete.get("experiment")!=experiment:
            raise ValueError(f"{name} is not a completed matching-profile phase")
        if complete.get("math_checks_passed") is not True:
            raise ValueError(f"{name} mathematical checks have not passed")
        if complete.get("source_digest")!=source_digest or provenance.get("code_digest")!=source_digest:
            raise ValueError(f"{name} scientific source differs; preserve it and run the current contract in a new directory")
        if profile=="full":
            expected={"graphs":201} if name=="A" else {"graphs":531,"learned_runs":240,"seed_optimizer_updates":120000}
            if any(complete.get(k)!=v for k,v in expected.items()):
                raise ValueError(f"{name} FULL graph/run/update coverage differs")
        records[name]={"path":str(path),"completion_sha256":file_sha256(path/"completion.json"),
                       "source_manifest_sha256":file_sha256(path/"source_manifest.json"),
                       "completion":complete,"score_based_pass_gate":False}
    return records


def assert_predecessors_unchanged(records):
    for name,record in records.items():
        path=Path(record["path"])
        if file_sha256(path/"completion.json")!=record["completion_sha256"] or file_sha256(path/"source_manifest.json")!=record["source_manifest_sha256"]:
            raise RuntimeError(f"Predecessor {name} changed during classification")
