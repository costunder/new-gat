"""Package current code, original proposal and selected labelled DEBUG evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from .common import ROOT,FOLDER,file_sha256,source_manifest

PROMPT="""# 검토 요청

이 ZIP은 2026-10-05 edge_metric_relations 새 A/B/C 트랙이다.
기존 scalar E/J와 persistent node-copy 실험의 성능을 새 모델의 결과로 사용하지 말아 주세요.

읽는 순서: code/research/edge_metric_relations/README.md,
MODEL_MATH.md, EXPERIMENT_DESIGN.md, VERIFICATION.md, 실제 geometry/gates/operators,
audit/synthetic/classification 코드와 DEBUG evidence.
reference/의 ORIGINAL_*는 받은 제안의 원문이며 현재 구현이나 FULL 결과가 아닙니다.

우선 실제 forward를 수식으로 설명한 뒤 아래를 검토해 주세요.

1. 중복 보정 Bcal, 양의 Ccal, 내부 이차항/쌍선형항과 actual propagation이 일치하는가?
2. ε=1e−4 RMS, DA row SUM/occurrence mean, B same-raw-teacher oracle가 올바른가?
3. B raw Qθ(X)X와 C normalized residual을 섞어 설명하지 않았는가?
4. 제곱 teacher의 크기 제한, 부분 관측과 cycle 복원의 조건을 정확히 명시했는가?
5. 모든 generator가 loss/backward/optimizer에 연결되어 있고 모든 선언된 입력/조건을 사용했는가?
6. FULL A201graph/B240learned run120000update/C630run315000update와 데이터 누수 방지가 유지되는가?
7. frozen 개입과 동일 Z의 순수 비대각 제거를 구분했는가?

수학적으로 성립하는 사실, 구현 검증, DEBUG fixture, 실제 FULL 결과, 미검증 연구가설을 구분해 주세요.
SERVER FULL 결과가 없으면 유용성/일반화/분류 우월성을 입증했다고 말하지 마세요.
수정 제안에는 코드 위치와 문제가 되는 실제 경로를 적어 주세요.

DEBUG 그림과 표는 코드 검증용 fixture 결과입니다. B의 그림은 3epoch/2seed이며
로그 그림의 표시 하한은 1e−16입니다. 원래 CSV 수치는 바꾸지 않았습니다.
대용량 원문 CSV.gz, per-graph/per-realization CSV, tensor/checkpoint는 이 ZIP에서
생략하고 MANIFEST의 evidence_all_files_index에 전체 경로·크기·해시를 남겼습니다.
요약·선택·학습·개입 CSV와 그림은 함께 들어 있습니다. 생략 파일을 읽었다고 말하지 마세요.
"""

SUMMARY_CSV = frozenset({
    "resources.csv", "paired_comparisons.csv", "summary.csv", "training.csv",
    "metrics.csv", "metric_estimates.csv", "branch_diagnostics.csv",
    "branch_estimates.csv", "interventions.csv", "intervention_changes.csv",
    "final_validation_selection.csv", "tuning_validation.csv",
    "epoch_history.csv",
})


def package(output,evidence):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Review ZIP already exists; preserve it")
    output.parent.mkdir(parents=True,exist_ok=True)
    files={}
    source=source_manifest()
    # Include unchanged import dependencies so the actual data adapters,
    # seed/dropout helpers and audited legacy controls can be inspected.
    for name in source["sha256"]:
        files[f"code/{name}"]=ROOT/name
    files["code/AGENTS.md"]=ROOT/"AGENTS.md"
    # Package bootstrap and an inherited adapter's unused lazy import too.
    # The new synthetic track calls only make_specs/_seed from that adapter.
    for name in ("research/__init__.py", "research/wedge_propagation/learned/model.py"):
        files[f"code/{name}"]=ROOT/name
    for folder,prefix in ((FOLDER,"code/research/edge_metric_relations"),
                          (ROOT/"docs/edge_metric_redesign_20261005","reference")):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[f"{prefix}/{path.relative_to(folder).as_posix()}"]=path
    for path in sorted((ROOT/"tests").glob("test_edge_metric*.py")):
        files[f"code/tests/{path.name}"]=path
    index=[]
    evidence_folders=[]
    executed_sources=[]
    for number,folder in enumerate(evidence):
        folder=Path(folder).resolve()
        if not folder.is_dir():raise FileNotFoundError(folder)
        complete=folder/"completion.json"
        if complete.is_file() and json.loads(complete.read_text(encoding="utf-8")).get("profile")!="debug":
            raise ValueError("This initial handoff accepts labelled DEBUG only; use a separate FULL evidence handoff")
        if not complete.is_file() and "DEBUG" not in folder.name:
            raise ValueError("Evidence without completion must have an explicit DEBUG folder name")
        prefix=f"evidence/DEBUG-{number+1}"
        evidence_folders.append({"source":str(folder),"archive_prefix":prefix,"scope":"DEBUG"})
        recorded=folder/"source_manifest.json"
        if recorded.is_file():
            executed=json.loads(recorded.read_text(encoding="utf-8"))
            changed=sorted(name for name in set(executed["sha256"])|set(source["sha256"])
                           if executed["sha256"].get(name)!=source["sha256"].get(name))
            if set(changed)-{"research/edge_metric_relations/package_review.py"}:
                raise ValueError("DEBUG scientific source differs from the current implementation; use matching evidence")
            executed_sources.append({"archive_prefix":prefix,"execution_code_digest":executed["code_digest"],
                                     "current_code_digest":source["code_digest"],"changed_files":changed,
                                     "all_model_data_training_solver_files_identical":True,
                                     "difference_scope":"artifact tooling only; not executed by A/B/C" if changed else "none"})
        for path in sorted(folder.rglob("*")):
            if not path.is_file():continue
            relative=path.relative_to(folder).as_posix()
            include=(path.suffix in (".json",".md",".txt",".log",".xml")
                     or path.name in SUMMARY_CSV
                     or path.name=="package_review_before.py"
                     or ("figures" in path.parts and path.suffix in (".png",".pdf")))
            if "tmp" in path.relative_to(folder).parts:
                # Unit tests intentionally create mocked FULL contract records.
                # Ship their test report, not those records as scientific evidence.
                include=False
            index.append({"source":str(path),"relative":relative,"bytes":path.stat().st_size,
                          "sha256":file_sha256(path),"included":include,"scope":"DEBUG"})
            if include:files[f"{prefix}/{relative}"]=path
    manifest={"scope":"new implementation plus original proposal and DEBUG; no server FULL results",
              "source":source,"files":{name:{"sha256":file_sha256(path),"bytes":path.stat().st_size} for name,path in files.items()},
              "evidence_all_files_index":index,"evidence_folders":evidence_folders,
              "execution_source_comparisons":executed_sources,"original_results_modified":False}
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items():archive.write(path,name)
        archive.writestr("REVIEW_PROMPT.md",PROMPT)
        archive.writestr("EVIDENCE_INDEX.md", "# DEBUG evidence\n\n"+
                         "Repository results links map to these ZIP prefixes. Original execution source digests remain in the evidence; the final packaging-only changes are listed in MANIFEST.json. All model, data, solver and training files are identical.\n\n"+
                         "\n".join(f"- `{row['source']}` → `{row['archive_prefix']}/`" for row in evidence_folders))
        archive.writestr("MANIFEST.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise RuntimeError("Review archive CRC failed")
    print(json.dumps({"path":str(output),"sha256":file_sha256(output),"files":len(files)+3,"bytes":output.stat().st_size},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--debug-evidence",action="append",default=[],type=Path)
    args=parser.parse_args();package(args.output,args.debug_evidence)


if __name__=="__main__":main()
