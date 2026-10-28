"""Bounded serial expert collection. No optimizer, retries or outcome selection."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .compact import convert_episode, sha256
from .run_rl_trials import validate_completion, utc
from .trial_utils import write_json, code_identity

MAX_BYTES = 10*1024**3
CHECKPOINT = "pretrained_assembly/tight_insertion/model.pth"
CHECKPOINT_SHA = "7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990"


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def design():
    conditions = [("baseline",[0.,0.])]
    for cm in (1,3,5):
        for axis in (0,1):
            for sign in (1,-1):
                offset=[0.,0.];offset[axis]=sign*cm/100
                conditions.append((f"{'x' if axis==0 else 'y'}{'plus' if sign==1 else 'minus'}{cm:02d}cm",offset))
    rows=[]
    for i,seed in enumerate((920301,920302,920303,920304)):
        ordered=conditions if i%2==0 else conditions[:1]+conditions[:0:-1]
        for condition,offset in ordered:
            rows.append(dict(id=f"s{seed}_{condition}",seed=seed,condition=condition,offset_xy=offset,
                             pair_id=f"initial_{seed}",wrench_frame_mode="world_hold",stage="scan"))
    acceptance=[]
    for label,seed,mode,xy in (("legacy",17,"legacy",[0.,0.]),("world",17,"world_hold",[0.,0.]),
                               ("baseline",920101,"world_hold",[0.,0.]),("offset",920101,"world_hold",[0.,.05])):
        acceptance.append(dict(id="accept_"+label,seed=seed,condition="accept_"+label,offset_xy=xy,
            pair_id=f"accept_initial_{seed}",wrench_frame_mode=mode,stage="acceptance"))
    return dict(plan_version=1,created_utc=utc(),identity=code_identity(),checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA,max_attempts=60,max_acceptance=8,max_scan=52,max_bytes=MAX_BYTES,
        max_steps=2400,wall_limit_s=480,seeds=[920301,920302,920303,920304],
        frame="fixed table plane aligned with world XY; original random pose + offset, no new RNG draw",
        range_rationale="native centers +/-0.10m; added +/-0.05m gives +/-0.15m centers inside table half-widths .2375/.20m; native orientation/z retained, object may overhang as in native resets",
        acceptance=acceptance,scan=rows)


def check_pairing(root, manifest, row, trial):
    candidates=[r for r in manifest['episodes'] if r['pair_id']==row['pair_id']]
    if not candidates:
        return
    from .analyze_rl_trials import load_trial
    baseline=load_trial(root/candidates[0]['directory'])
    a,b=baseline['metadata'],trial['metadata']
    for key in ('expert_model_sha256','initial_torch_rng_sha256','config','camera_ids','joint_limits_rad'):
        if a[key]!=b[key]:
            raise RuntimeError('Paired metadata changed: '+key)
    if not np.array_equal(a['initial_offset']['raw_pose_w'],b['initial_offset']['raw_pose_w']):
        raise RuntimeError('Native initial pose changed between paired offsets')
    for key,value in baseline['arrays']['initial'].items():
        if key=='object_pose_w':
            continue
        if not np.array_equal(value,trial['arrays']['initial'][key]):
            raise RuntimeError('Paired non-object initial state changed: '+key)


def run(root, stage, continue_unaffected=False):
    plan_path=root/'run_status/plan.json'
    if not plan_path.exists():
        (root/'run_status').mkdir(parents=True,exist_ok=True)
        write_json(plan_path,design())
    plan=json.loads(plan_path.read_text())
    if sha256(CHECKPOINT)!=plan['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(manifest_version=1,episodes=[])
    if stage=='scan':
        gate=json.loads((root/'run_status/acceptance_gate.json').read_text())
        if not gate.get('passed'):
            raise RuntimeError('Acceptance analysis did not pass')
    invalid_groups=set()
    if continue_unaffected:
        for p in (root/'run_status').glob('*.json'):
            s=json.loads(p.read_text())
            if s.get('initialization_rejected'):
                invalid_groups.add(s['pair_id'])
        write_json(root/'run_status/continuation.json',dict(created_utc=utc(),
            rule='Skip all unrun variants of a verified invalid initial state; preserve all predeclared seeds, never replace them',
            invalid_groups=sorted(invalid_groups)))
    for row in plan[stage]:
        existing=[e for e in manifest['episodes'] if e['id']==row['id']]
        status_path=root/'run_status'/(row['id']+'.json')
        if row['pair_id'] in invalid_groups:
            if not status_path.exists():
                write_json(status_path,dict(**row,attempt=False,state='skipped_invalid_initial_group'))
            continue
        if existing:
            from .manifest import ManifestReader
            ManifestReader(manifest_path,verify=True)
            status=json.loads(status_path.read_text())
            if status.get('state')!='committed':
                raise RuntimeError('Manifest/status commit mismatch; inspect retained attempt')
            validate_completion(root/existing[0]['directory'],row,status.get('returncode'))
            continue
        if status_path.exists():
            raise RuntimeError('Retained incomplete attempt; no automatic retry: '+row['id'])
        statuses=list((root/'run_status').glob('s*.json'))+list((root/'run_status').glob('accept_*.json'))
        statuses=[p for p in statuses if json.loads(p.read_text()).get('attempt')]
        if len(statuses)>=60 or sum(json.loads(p.read_text())['stage']==stage for p in statuses)>=(8 if stage=='acceptance' else 52):
            raise RuntimeError('Attempt budget reached')
        used=tree_bytes(root)
        # Reserve worst-case raw2400 + compact + videos + logs (2GiB), not just average.
        available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        if used+2*1024**3>MAX_BYTES or shutil.disk_usage(root).free<15*1024**3 or available<6*1024**3:
            raise RuntimeError('Storage/RAM preflight failed before next attempt')
        temporary=root/'temporary'/row['id']
        command=['bash','visuotactile/run_cpu.sh','-m','visuotactile.collect','--output',str(temporary),
            '--checkpoint',CHECKPOINT,'--episodes','1','--max-steps','2400','--seed',str(row['seed']),
            '--pair-id',row['pair_id'],'--experiment-condition',row['condition'],
            '--wrench-frame-mode',row['wrench_frame_mode'],'--offset-xy',*[str(x) for x in row['offset_xy']]]
        status=dict(**row,attempt=True,state='running',command=command,launch_utc=utc(),bytes_before=used,
            capacity_reserve_bytes=2*1024**3,available_ram_bytes=available,
            gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
        write_json(status_path,status)
        start=time.monotonic()
        with (root/'run_status'/(row['id']+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                code=child.wait(timeout=480)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:code=child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();code=child.wait()
        status.update(returncode=code,wall_seconds=time.monotonic()-start,state='auditing')
        write_json(status_path,status)
        try:
            trial=validate_completion(temporary,row,code)
            check_pairing(root,manifest,row,trial)
            if row['wrench_frame_mode']=='world_hold' and trial['summary']['requested_vs_applied_peak_vector_error_N']>1e-5:
                raise RuntimeError('World wrench compatibility verification failed')
            result=json.loads((temporary/'result.json').read_text())
            summary=trial['summary']
            if summary['hdf5_bytes']>800000*summary['control_steps']+2000000:
                raise RuntimeError('Unexplained raw storage growth')
            directory='episodes/'+row['id']
            conversion=convert_episode(temporary/'trajectories.h5',root/directory,result=result)
            from .analyze_rl_trials import load_trial
            compact_trial=validate_completion(root/directory,row,code)
            if compact_trial['summary']['audit_errors']:
                raise RuntimeError('Compact full audit failed')
            if tree_bytes(root)>MAX_BYTES:
                raise RuntimeError('Output budget exceeded; preserve temporary and stop')
            record=dict(**row,**conversion,directory=directory,outcome=summary['outcome'],valid=True,
                schema_version=2,trace_version=2,storage_version=1,wrench_trace_version=3,
                summary=summary,metadata_json_file_sha256=sha256(temporary/'metadata.json'),
                parameter_updates=0,video_role='viewing_derivative',nominal_predictions_available=False)
            manifest['episodes'].append(record);write_json(manifest_path,manifest)
            status.update(state='committed',outcome=summary['outcome'],length=summary['control_steps'],
                exact_roundtrip_fields=conversion['exact_fields'],bytes_before_cleanup=tree_bytes(root),
                result=result,finish_utc=utc())
            write_json(status_path,status)
            # Only this new attempt's redundant temporary files; originals never touched.
            shutil.rmtree(temporary)
            status['bytes_after_cleanup']=tree_bytes(root);write_json(status_path,status)
            print(json.dumps({k:status[k] for k in ('id','outcome','length','wall_seconds','bytes_after_cleanup')}),flush=True)
        except BaseException as exc:
            status.update(state='stopped',error=repr(exc));write_json(status_path,status)
            # Only an explicit geometry rejection BEFORE any recorded action
            # allows the other predeclared independent groups to continue.
            meta_path=temporary/'metadata.json'
            if meta_path.exists() and not (temporary/'trajectories.h5').exists():
                meta=json.loads(meta_path.read_text())
                geometry=meta.get('initial_geometry',{})
                if geometry.get('valid_box_initialization') is False:
                    status.update(initialization_rejected=True,geometry=geometry,
                                  state='invalid_initialization_no_action')
                    write_json(status_path,status)
                    if continue_unaffected:
                        invalid_groups.add(row['pair_id'])
                        continue
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=('acceptance','scan'),required=True)
    p.add_argument('--continue-unaffected-initializations',action='store_true')
    a=p.parse_args();run(a.root,a.stage,a.continue_unaffected_initializations)


if __name__=='__main__':main()
