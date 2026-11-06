"""Fixed stratified experts and source-target replays; no student or collection."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np

from tools.runtime import CODE,read,write,sha
from tools.narrow_repair import execute,beat,status,protected


def plan(root):
    assert read(root/'static_summary.json')['passed']
    assert not (root/'replay_plan.json').exists(),'Do not replace the registered plan'
    s=read(root/'session.json');parent=Path(s['parent']);entries=read(parent/'manifest.json')['episodes']
    audit={p.stem:read(p) for p in (root/'episodes').glob('*.json')};families={f['identity']:f for f in read(parent/'families.json')}
    center=read(parent/'session.json')['domain']['center'];selected=[];used=set();missing=[]
    def feature(e):
        a=audit[e['id']]['initial'];return np.r_[(np.array(a['object_xyz'][:2])-center['object_xyz'][:2])/.025,
            (np.array(a['relative_xyz'][:2])-center['relative_xyz'][:2])/.035,
            audit[e['id']]['initial_rotation_deg']/20]
    for origin in ('old200','new75'):
        for split in ('train','validation','test'):
            for alpha in ('native','slow','mid') if origin=='old200' else ('native','slow'):
                pool=[e for e in entries if e['family'] not in used and audit[e['id']]['origin']==origin and e['split']==split and
                    (e['alpha']==1 if alpha=='native' else e['alpha']<=.65 if alpha=='slow' else .65<e['alpha']<1)]
                if not pool:missing.append(dict(origin=origin,split=split,alpha=alpha));continue
                # Alternate center/edge coverage, with max-min separation from earlier choices.
                if not selected:score=lambda e:-float(np.linalg.norm(feature(e)))
                else:score=lambda e:min(float(np.linalg.norm(feature(e)-feature(x))) for x in selected)
                e=max(sorted(pool,key=lambda e:e['id']),key=score);selected.append(e);used.add(e['family'])
    jobs=[]
    for i,e in enumerate(selected):
        spec=dict(families[e['family']]['spec'],identity=e['family'],family=e['family'],split=e['split'])
        modes=['native_expert','tick_replay','h24_replay']+(['source_slow_expert'] if e['alpha']<1 else [])
        for mode in modes:
            jobs.append(dict(identity=e['id']+'__'+mode,source_id=e['id'],source=e['path'],source_sha256=e['sha256'],
                alpha=e['alpha'],mode=mode,spec=spec,source_success=True,video=i in (0,len(selected)-1)))
    for e in read(root/'failures.json'):
        f=families[e['family']];jobs.append(dict(identity=e['identity']+'__failed_source_expert',source_id=e['identity'],source=e['source'],
            source_sha256=sha(e['source']),alpha=e['alpha'],mode='source_slow_expert' if e['alpha']<1 else 'native_expert',
            spec=dict(f['spec'],identity=e['family'],family=e['family'],split=f['split']),source_success=False,video=True))
    assert len(jobs)<=s['maximum_kit_starts']
    write(root/'replay_plan.json',dict(jobs=jobs,source_ids=[e['id'] for e in selected],source_families=len(used),
        modes=dict(Counter(x['mode'] for x in jobs)),missing_strata=missing,
        selection='Frozen AFTER full static audit, old/new x split x alpha strata; deterministic center then farthest positional/orientation coverage; no student outcomes.',
        limitation='Stratified coverage cohort, not iid population success estimation; source dataset conditioned on success. Separate two failed-source repeats.',
        source_precision='Current unifiedFP32 and corrected cameras. Legacy200 did not record a full runtime arithmetic snapshot; do not claim bitwise original arithmetic.'))


def physical_config(cfg):
    result={k:cfg[k] for k in ('sim','decimation','assets','action','reset','domain_randomization','termination','precise_assembly','reward','obs','observation_noise_model','action_noise_model','episode_length_s','is_finite_horizon')}
    result=json.loads(json.dumps(result));result['sim'].pop('log_dir',None)
    for k in ('object_urdf','receptive_urdf'):
        result['assets'][k]=sha(Path(result['assets'][k]))
    return result


def worker(root,mode):
    from tools.narrow_collection import environment
    from visuotactile.state_benchmark import trial
    from visuotactile.video import verify_video
    from visuotactile.recording import model_fingerprint
    s=read(root/'session.json');app,env,player,lease=environment(root,evaluation=mode=='replay')
    fingerprint=model_fingerprint(player.model);jobs=read(root/'replay_plan.json')['jobs'];rows=[]
    folder=root/'physical';folder.mkdir(exist_ok=True)
    for job in jobs:
        if (job['mode'].endswith('expert')) != (mode=='expert'):continue
        assert sha(job['source'])==job['source_sha256'];out=folder/job['identity'];beat(root,stage=mode,identity=job['identity'])
        with h5py.File(job['source']) as f:
            g=next(iter(f['episodes'].values()));meta=json.loads(f.attrs['metadata_json']);targets=g['transition/target_after_rad'][:]
            init={k:v[0] for k,v in g['transition/diagnostics_before'].items()}
            source_rng=g['transition/torch_rng_before_step'][:];source_q=g['transition/diagnostics_after/joint_position_rad'][:]
        kwargs=dict(video=job['video'],video_title=job['mode']+' | '+job['source_id']+' | 1x')
        if mode=='expert':
            alpha=1. if job['mode']=='native_expert' else job['alpha']
            speed=dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all');env.speed_config=speed;kwargs['expert_speed']=speed
        else:kwargs.update(replay=targets,position=True,replay_horizon=24 if job['mode']=='h24_replay' else None)
        result=trial(env,lease,out,job['spec'],{},player,s['expert_checkpoint'],**kwargs)
        row=dict(source_id=job['source_id'],source_success=job['source_success'],source_sha256=job['source_sha256'],
            mode=job['mode'],source_alpha=job['alpha'],result=result,identity=job['identity'])
        if result['legal']:
            actual=read(out/'initial_observable.json');keys=('joint_position_rad','joint_velocity_rad_s','object_pose_w','hole_pose_w','table_pose_w','palm_body_pose_w','finger_position_w','finger_quat_wxyz')
            row['initial_errors']={k:float(abs(np.asarray(actual[k])[0]-init[k]).max()) for k in keys}
            row['physical_config_exact']=physical_config(meta['config'])==physical_config(read(out/'metadata.json')['config'])
            write(out/'audit.json',row)
            assert max(row['initial_errors'].values())<1e-6 and row['physical_config_exact'],'Initial/config mismatch: retain result and stop'
            with h5py.File(out/'numeric.h5') as f:
                g=f['episode'];n=min(result['steps'],len(targets));rng=g['transition/torch_rng_before_step'][:n]
                row['first_event_rng_equal']=bool(np.array_equal(rng[0],source_rng[0]))
                row['event_rng_equal_ticks']=int(np.all(rng==source_rng[:n],axis=1).sum())
                row['actual_q_source_max_rad']=float(abs(g['transition/diagnostics_after/joint_position_rad'][:n]-source_q[:n]).max())
                if mode=='replay':
                    err=float(abs(g['transition/target_after_rad'][:n]-targets[:n]).max())
                    row.update(requested_source_max_rad=err,actuator_source_max_rad=float(abs(g['transition/actuator_substep_target_rad'][:n]-targets[:n,None,:]).max()),
                        source_ticks=len(targets),executed_source_ticks=n,source_exhausted=n==len(targets),extra_hold_ticks=0,
                        early_native_termination=n<len(targets),h24_padding_executed=False)
                    assert err==0 and row['actuator_source_max_rad']==0,'Replay target contract error'
            if job['video']:row['video']=verify_video(out/'rollout.mp4',result['steps'])
        rows.append(row);write(root/(mode+'_results.json'),rows)
    assert model_fingerprint(player.model)==fingerprint,'Expert weights changed'
    write(root/(mode+'_complete.json'),dict(complete=True,starts=len(rows),parameter_updates=0,expert_fingerprint=fingerprint))


def run(root):
    assert read(root/'static_summary.json')['passed'];s=read(root/'session.json')
    assert time.time()<s['deadline']-1800
    def jobs(start):
        start('expert',['-m','tools.dataset_trust_replay','expert','--root',str(root)],kit=True)
        start('replay',['-m','tools.dataset_trust_replay','replay','--root',str(root)],kit=True)
    execute(root,'physical_acceptance',jobs,s['deadline']);protected(root)
    status(root,state='physical_complete',stage='await_review',formal_updates=0,student_rollouts=0,formal_collection=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('plan','run','expert','replay'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();kit=a.mode in ('expert','replay');code=0
    try:
        if kit:worker(root,a.mode)
        else:globals()[a.mode](root)
    except BaseException:
        write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc(),utc=time.time()));code=1
        if not kit:raise
    if kit:sys.stdout.flush();sys.stderr.flush();os._exit(code)
