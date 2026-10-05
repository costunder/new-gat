"""A -> B -> C server pipeline, with visible progress in the same terminal."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from .common import assert_source_unchanged,read_json,source_manifest,write_json
from .predecessors import check_predecessors
from .hardware import HARDWARE_PROFILES, get_policy, memory_window


def command(phase,args,output):
    module={"A":"audit","B":"synthetic","C":"classification"}[phase]
    cmd=[sys.executable,"-u","-B","-m",f"research.edge_metric_relations.{module}.study",
         "--profile",args.profile,"--device",args.device,"--output-dir",str(output),
         "--hardware-profile",getattr(args,"hardware_profile","auto")]
    if phase in ("A","C"):
        cmd += ["--source-dir",str(Path(args.source_dir).resolve()),"--data-root",args.data_root]
    if phase=="C":
        cmd += ["--mechanism-audit-dir",str(args.audit_dir),"--synthetic-dir",str(args.synthetic_dir)]
        if args.offline:cmd.append("--offline")
    return cmd


def run(args):
    policy=get_policy(getattr(args,"hardware_profile","auto"))
    if args.profile=="full" and (sys.platform!="linux" or torch.device(args.device).type!="cuda" or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("FULL requires a Linux server and explicitly allocated CUDA_VISIBLE_DEVICES")
    if torch.device(args.device).type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no fallback")
    output=Path(args.output_dir).resolve()
    if output.exists():raise FileExistsError("Preserve existing results; choose a NEW pipeline output directory")
    if not Path(args.source_dir).is_dir():raise FileNotFoundError("Original completed local-energy source directory is required")
    source=source_manifest()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"source_manifest.json",source)
    write_json(output/"hardware_policy.json",{"policy":policy,"memory":memory_window(args.device,policy)})
    write_json(output/"plan.json",{"profile":args.profile,"source_dir":str(Path(args.source_dir).resolve()),
                                  "phases":["A","B","C"],"device":args.device,"hardware_profile":policy["name"],
                                  "original_results_preserved":True,"full_training_location":"Linux server only"})
    start=time.perf_counter();records={}
    try:
        for phase in ("A","B","C"):
            supplied=args.audit_dir if phase=="A" else args.synthetic_dir if phase=="B" else None
            folder=Path(supplied).resolve() if supplied else output/phase
            if not supplied:
                cmd=command(phase,args,folder)
                print(f"[phase start] {phase} output={folder} command={cmd}",flush=True)
                result=subprocess.run(cmd,check=False)
                if result.returncode!=0:
                    raise RuntimeError(f"Phase {phase} failed with code {result.returncode}; inspect {folder/'terminal.log'}; session and results preserved")
            complete=read_json(folder/"completion.json")
            if complete.get("completed") is not True or complete.get("profile")!=args.profile:
                raise ValueError(f"Phase {phase} did not complete its declared profile")
            assert_source_unchanged(source)
            records[phase]={"path":str(folder),"completion":complete,"reused":bool(supplied)}
            if phase=="A":args.audit_dir=folder
            if phase=="B":
                args.synthetic_dir=folder
                check_predecessors(args.audit_dir,args.synthetic_dir,args.profile,source["code_digest"])
            write_json(output/f"phase_{phase}.json",records[phase])
            print(f"[phase complete] {phase} seconds_total={time.perf_counter()-start:.1f}",flush=True)
        write_json(output/"completion.json",{"completed":True,"profile":args.profile,"phases":records,
                                             "seconds":time.perf_counter()-start,
                                             "full_training_run":args.profile=="full","source_digest":source["code_digest"]})
        print(f"[complete] A/B/C profile={args.profile} output={output}",flush=True)
    except Exception as error:
        traceback.print_exc()
        write_json(output/"failure.json",{"phase_records":records,"type":type(error).__name__,"message":str(error),
                                           "results_preserved":True,"recovery":"fix the reported cause; use a NEW output directory"})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=("full","debug"),default="full")
    parser.add_argument("--device",default="cuda",help="cuda dispatches all explicitly allocated visible GPUs")
    parser.add_argument("--hardware-profile",choices=HARDWARE_PROFILES,default="auto",
                        help="allocation policy only; full scientific scale is unchanged")
    parser.add_argument("--source-dir",required=True,help="original completed local-energy audit")
    parser.add_argument("--data-root",default="data/wedge-citation")
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--audit-dir",type=Path,help="reuse matching completed A in its original directory")
    parser.add_argument("--synthetic-dir",type=Path,help="reuse matching completed B in its original directory")
    parser.add_argument("--offline",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
