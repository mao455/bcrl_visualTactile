"""Sequential paired first-episode collection, with a durable plan and status.

Never launches training. Each child uses run_cpu.sh, one environment and fresh
output. Data/audit failure stops the next stage; failed attempts are retained.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

from .trial_utils import code_identity, write_json
from .analyze_rl_trials import CONDITIONS, analyze, load_trial, compare_pair


def make_trials(seeds, stage):
    trials=[]
    for i,seed in enumerate(seeds):
        for condition in (CONDITIONS if i%2==0 else CONDITIONS[::-1]):
            trials.append(dict(name=f"pair{i+1:02d}_{condition}",seed=seed,condition=condition,
                               pair_id=f"{stage}_pair{i+1:02d}"))
    return trials


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_completion(path, entry=None, returncode=0):
    """Identical fresh/resume gate; exit zero is insufficient (Kit can mask errors)."""
    path=Path(path)
    if returncode != 0 or (path/"FAILED.txt").exists():
        raise RuntimeError("Incomplete/failed collector result: "+str(path))
    if (path/"result.json").is_file():
        result=json.loads((path/"result.json").read_text())
    elif (path/"data.h5").is_file():
        from .compact import EpisodeFile
        with EpisodeFile(path/"data.h5") as f:
            if int(f.attrs.get("storage_version",0)) not in (1,2) or "result_json" not in f.attrs:
                raise RuntimeError("Missing result provenance")
            result=json.loads(f.attrs["result_json"])
    else:
        raise RuntimeError("Missing collector result: "+str(path))
    trial=load_trial(path)
    s,m=trial["summary"],trial["metadata"]
    if s["audit_errors"] or result.get("steps") != s["control_steps"] or result.get("started_episodes") != 1:
        raise RuntimeError("Committed length/result/audit mismatch: "+str(path))
    if result.get("successful_episodes") != int(s["outcome"]=="success"):
        raise RuntimeError("Result success disagrees with native terminal state")
    if result.get("externally_censored") != (s["outcome"]=="censored"):
        raise RuntimeError("Result censor disagrees with committed boundary")
    if entry:
        if "wrench_frame_mode" in entry and (s.get("missing_telemetry") or s.get("wrench_trace_version") != 3):
            raise RuntimeError("Bounded dataset requires complete wrench3 telemetry")
        for k in ("seed","condition","pair_id"):
            if k in entry and m.get(k)!=entry[k]:
                raise RuntimeError("Experiment identity mismatch: "+k)
        if "offset_xy" in entry and m["initial_offset"]["offset_xy_m"]!=entry["offset_xy"]:
            raise RuntimeError("Initialization offset mismatch")
        if "wrench_frame_mode" in entry and m.get("wrench_frame_mode")!=entry["wrench_frame_mode"]:
            raise RuntimeError("Wrench compatibility mode mismatch")
    return trial


def prior_attempt(root,name,entry=None):
    """At most one retry, and only when the simulator never produced an HDF5.

    Successful attempts are reused, never reselected based on task outcome.
    Failed/partial trajectories require investigation instead of auto-retry.
    """
    paths=sorted(root.glob(name+"*.status.json"))
    if not paths:
        return name,None
    statuses=[json.loads(p.read_text()) for p in paths]
    good=[s for s in statuses if s.get("returncode")==0 and s.get("episode_started") and not s.get("audit_errors")]
    if len(good)>1:
        raise RuntimeError(f"Multiple successful software attempts for {name}; cannot select by outcome")
    if good:
        selected=good[0].get("attempt_name",name)
        return selected,validate_completion(root/selected,entry,good[0]["returncode"])
    if len(paths)!=1 or statuses[0].get("returncode")!=-6 or (root/name/"trajectories.h5").exists():
        raise RuntimeError(f"Unsafe or exhausted retry for {name}; inspect retained attempt")
    return name+"_attempt02",None


def run_stage(root,plan,checkpoint,max_steps,wall_limit,resume=False):
    loaded={}
    for entry in plan["trials"]:
        attempt_name,previous=(prior_attempt(root,entry["name"],entry) if resume else (entry["name"],None))
        if previous is not None:
            if previous["summary"]["audit_errors"]:
                raise RuntimeError("Previously completed trial now fails audit")
            loaded[(entry["seed"],entry["condition"])]=previous
            continue
        if shutil.disk_usage(root).free < 15*1024**3:
            raise RuntimeError("Less than 15 GiB free; stop before launching another trial")
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.startswith('MemAvailable:')))*1024
        if available < 6*1024**3:
            raise RuntimeError("Less than 6 GiB available RAM; stop before next trial")
        path=root/attempt_name
        command=["bash","visuotactile/run_cpu.sh","-m","visuotactile.collect",
                 "--checkpoint",str(checkpoint),"--condition",entry["condition"],
                 "--pair-id",entry["pair_id"],"--seed",str(entry["seed"]),
                 "--teacher-group","0","--episodes","1","--max-steps",str(max_steps),
                 "--num-envs","1","--output",str(path),"--perception-version","2","--camera","oblique"]
        status=dict(**entry,attempt_name=attempt_name,state="launching",command=command,launch_utc=utc(),episode_started=False)
        status["resources_before"] = dict(available_ram_bytes=available,free_disk_bytes=shutil.disk_usage(root).free,
            gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip())
        status_path=root/(attempt_name+".status.json")
        write_json(status_path,status)
        with (root/(attempt_name+".launcher.log")).open("x") as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            status.update(state="running",pid=proc.pid);write_json(status_path,status)
            started=time.monotonic()
            try:
                code=proc.wait(timeout=wall_limit)
            except subprocess.TimeoutExpired:
                # This handle belongs only to our newly launched child.
                proc.terminate()
                try:
                    code=proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill();code=proc.wait()
                status["error"]="external wall-clock resource limit; software/resource censor, not robot failure"
        status.update(state="finished",returncode=code,finish_utc=utc(),wall_seconds=time.monotonic()-started)
        if (path/"trajectories.h5").exists():
            try:
                trial=validate_completion(path,entry,code);loaded[(entry["seed"],entry["condition"])]=trial
                status.update(episode_started=True,audit_errors=trial["summary"]["audit_errors"],
                              recorded_outcome=trial["summary"]["outcome"])
            except Exception as exc:
                status["audit_errors"]=[repr(exc)]
        write_json(status_path,status)
        print(json.dumps(status),flush=True)
        if code or status.get("audit_errors") or not status["episode_started"]:
            raise RuntimeError(f"Trial failed runtime/data acceptance: {entry['name']}; retained for investigation")
        a,b=loaded.get((entry["seed"],"original")),loaded.get((entry["seed"],"force_1p5"))
        if a and b and not compare_pair(a,b)["initial_pass"]:
            raise RuntimeError(f"Initial/config pairing failed: seed {entry['seed']}")
    analysis_path=root/"analysis"
    suffix=0
    while analysis_path.exists():
        suffix+=1
        analysis_path=root/f"analysis_resumed{suffix:02d}"
    report=analyze(root,analysis_path)
    if any(r.get("audit_errors") or r.get("runtime_error") or not r.get("started") for r in report["trials"]):
        raise RuntimeError("Stage analysis failed; subsequent stages blocked")
    if any(not pair.get("initial_pass",False) for pair in report["pairs"]):
        raise RuntimeError("Stage pairing failed; subsequent stages blocked")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,default=Path("pretrained_assembly/tight_insertion/model.pth"))
    p.add_argument("--acceptance-seeds",type=int,nargs=3,default=[910101,910102,910103])
    p.add_argument("--pilot-seeds",type=int,nargs=20,default=list(range(910201,910221)))
    p.add_argument("--exposure-evidence",type=Path,required=True,
                   help="Independent diagnostic with real native force exposure; never counted as cohort")
    p.add_argument("--max-steps",type=int,default=2400)
    p.add_argument("--wall-limit-seconds",type=int,default=480)
    p.add_argument("--resume-startup-errors",action="store_true",
                   help="Reuse valid attempts; retry a pre-HDF5 SIGABRT once in a new directory with the same seed")
    args=p.parse_args()
    if len(set(args.acceptance_seeds+args.pilot_seeds))!=23:
        p.error("Acceptance and pilot require disjoint predeclared seeds")
    if not 1<=args.max_steps<=2400:
        p.error("Bounded experiment supports at most 2400 native steps")
    if not 1<=args.wall_limit_seconds<=480:
        p.error("Bounded experiment supports at most 480 wall-clock seconds per trial")
    evidence=load_trial(args.exposure_evidence)
    if (evidence["summary"]["audit_errors"] or not evidence["summary"]["applied_force"]
            or not evidence["summary"]["applied_force"]["nonzero_steps"]):
        p.error("Exposure diagnostic must pass audit and contain nonzero native external force")
    if not args.resume_startup_errors:
        args.output.mkdir(parents=True,exist_ok=False)
    identity=code_identity()
    plans={stage:dict(stage=stage,created_utc=utc(),code_identity=identity,
                     checkpoint=str(args.checkpoint),exposure_evidence=str(args.exposure_evidence),
                     max_steps=args.max_steps,wall_limit_seconds=args.wall_limit_seconds,
                     exposure_evidence_in_cohort=False,
                     design="Original native DR versus force_scale * 1.5 only; no gate/probability changes",
                     trials=make_trials(seeds,stage))
           for stage,seeds in (("acceptance",args.acceptance_seeds),("pilot",args.pilot_seeds))}
    for stage,plan in plans.items():
        if args.resume_startup_errors:
            original=json.loads((args.output/stage/"plan.json").read_text())
            if original["trials"]!=plan["trials"] or original["checkpoint"]!=plan["checkpoint"]:
                p.error("Resume must use the same predeclared seeds, order and checkpoint")
            if (original.get("max_steps",2400)!=args.max_steps
                    or original.get("wall_limit_seconds",480)!=args.wall_limit_seconds):
                p.error("Resume must preserve the declared control-step and wall-clock budgets")
            plans[stage]=original
        else:
            (args.output/stage).mkdir();write_json(args.output/stage/"plan.json",plan)
    if args.resume_startup_errors:
        # Retain the earlier stop as immutable evidence before updating live status.
        prior=args.output/"initial_stop.json"
        if not prior.exists() and (args.output/"status.json").exists():
            with prior.open("x") as stream:
                stream.write((args.output/"status.json").read_text())
    try:
        write_json(args.output/"status.json",dict(state="running",resumed=args.resume_startup_errors,started_utc=utc()))
        for stage,plan in plans.items():
            run_stage(args.output/stage,plan,args.checkpoint,args.max_steps,args.wall_limit_seconds,args.resume_startup_errors)
        write_json(args.output/"status.json",dict(state="complete",finished_utc=utc()))
    except BaseException as exc:
        write_json(args.output/"status.json",dict(state="stopped",error=repr(exc),finished_utc=utc()))
        raise


if __name__=="__main__":
    main()
