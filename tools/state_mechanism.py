"""One finite A/B/C diagnosis; never expands data or resumes formal training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE, read, write, sha
from tools.narrow_repair import execute, beat, status, protected


def prepare(root):
    parent=(CODE/'outputs/visuotactile/run/narrow_repair_20261106_01').resolve()
    assert read(parent/'status.json')['state']=='complete' and read(parent/'status.json')['owned_workers_exited']
    root.mkdir(parents=True,exist_ok=False);previous=read(parent/'session.json')
    entries=read(parent/'manifest.json')['episodes'];families={f['identity']:f for f in read(parent/'families.json')}
    candidates=sorted([e for e in entries if e['split']=='train' and e['multi_view_valid'] and e['alpha']<1],
        key=lambda e:(abs(e['steps']/60-5),e['id']))
    selected=candidates[0];other=next(e for e in candidates if e['family']!=selected['family'])
    specs=[dict(families[e['family']]['spec'],identity=e['family'],family=e['family'],split='train') for e in (selected,other)]
    now=time.time();session=dict(parent=str(parent),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        started=now,deadline=now+4*3600,collection_deadline=now+4*3600,maximum_hours=4,
        maximum_output_bytes=12*1024**3,reserve_bytes=60*1024**3,resource_policy='shared_recent_v1',resource_target_percent=40.,
        expert_checkpoint=previous['expert_checkpoint'],expert_sha256=previous['expert_sha256'],precision_policy='unified_fp32',camera_content_validation=True,
        loss_weights=previous['loss_weights'],maximum_kit_starts=15,tiny_max_updates=4000,tiny_training_seconds=90*60,
        tiny_batch=16,microbatch=4,random_seed=2026110607,inference_seeds=[7091,8091],automatic_retries=False,
        owned_worker_roots=[str(root/'tiny_training')],
        gates=dict(gradient_relative_l2=2e-4,parameter_update_max_abs=2e-6,loss_relative_error=1e-5,target_mapping_max_abs=1e-6),
        tiny_fit=dict(arm_rmse=.003,hand_rmse=.02,fk_mm=3.,first_arm_rmse=.003,minimum_updates=400))
    write(root/'session.json',session);write(root/'budget.json',session)
    write(root/'plan.json',dict(tiny_entry=selected,developer_specs=specs,
        selection='Closest5s accepted multimodal TRAIN slow trajectory, tie identity; second distinct family by same rule. No student/test outcomes used.',
        oracle=5,tiny_closed_loop=2,overlap_closed_loop=8,formal_data_collection=0))
    guards=read(parent/'protected.json')
    for name in ('manifest.json','families.json','state_data/data.json','state_data/numeric.pt','state_data/statistics.pt','expanded_training/best.pt','expanded_training/last.pt','final_evaluation/result.json'):
        p=parent/name;guards[str(p)]=sha(p)
    write(root/'protected.json',guards);status(root,state='prepared',stage='cpu_audit',formal_updates=0,owned_workers_exited=True)
    print('Prepared',root,'tiny',selected['id'],'fixed deadline',session['deadline'],flush=True)


def oracle(root):
    import numpy as np
    import h5py
    from tools.narrow_collection import environment
    from visuotactile.state_benchmark import trial
    from visuotactile.video import verify_video
    from visuotactile.narrow_data import admission_conflict
    s=read(root/'session.json');plan=read(root/'plan.json');app,env,player,lease=environment(root,evaluation=True)
    rows=[];comparisons=[];folder=root/'oracle';folder.mkdir(exist_ok=False)
    def run(label,spec,**kwargs):
        beat(root,stage='oracle',identity=label);out=folder/label
        result=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],**kwargs)
        rows.append(dict(result,condition=label));write(root/'oracle_results.json',rows)
        if not result['legal']:raise RuntimeError('Preregistered oracle start rejected; no replacement')
        if kwargs.get('video'):verify_video(out/'rollout.mp4',result['steps'])
        return result,out
    for index,spec in enumerate(plan['developer_specs']):
        native,out=run(f'native_{index}',spec,video=index==0)
        trace=np.load(out/'trace.npz');targets=trace['target']
        replay,replayed=run(f'h24_target_replay_{index}',spec,replay=targets,position=True,replay_horizon=24,video=index==0)
        actual=np.load(replayed/'trace.npz');n=replay['steps']
        initial=admission_conflict(read(out/'initial_observable.json'),read(replayed/'initial_observable.json'),True,True)
        check=dict(identity=spec['identity'],native_success=native['success'],open_loop_success=replay['success'],
            censored=replay['action_exhausted'],ticks=n,
            source_to_requested_max_rad=float(abs(actual['command']-targets[:n]).max()),
            submitted_to_actuator_max_rad=replay['actuator_target_max_error'],
            requested_to_submitted_max_rad=replay['submitted_target_max_error'],
            actual_joint_difference_max_rad=float(abs(actual['joint']-trace['joint'][:n]).max()),
            boundary_max_rad=float(abs(actual['command'][::12]-targets[:n:12]).max()),initial_observable_errors=initial['observable_max_abs'])
        comparisons.append(check);write(root/'oracle_mapping.json',comparisons)
        if max(check[k] for k in ('source_to_requested_max_rad','submitted_to_actuator_max_rad','requested_to_submitted_max_rad','boundary_max_rad'))>s['gates']['target_mapping_max_abs']:
            raise RuntimeError('Oracle target mapping contract failure; B/C blocked')
    entry=plan['tiny_entry']
    with h5py.File(entry['path']) as f:
        targets=f['episodes'][next(iter(f['episodes']))]['transition/target_after_rad'][:]
    result,out=run('recorded_slow',plan['developer_specs'][0],replay=targets,position=True,replay_horizon=24,video=True)
    actual=np.load(out/'trace.npz');err=float(abs(actual['command']-targets[:result['steps']]).max())
    assert max(err,result['actuator_target_max_error'],result['submitted_target_max_error'])<=s['gates']['target_mapping_max_abs']
    write(root/'oracle_result.json',dict(passed=True,comparisons=comparisons,recorded_slow=dict(success=result['success'],censored=result['action_exhausted'],target_max_rad=err,
        actuator_max_rad=result['actuator_target_max_error'],submitted_max_rad=result['submitted_target_max_error']),
        starts=len(rows),scope='Native RL is closed loop; both H24 target cases are OPEN LOOP, not an online expert. Exact physical replay is not an acceptance requirement.'))


def closed_loop(root):
    import torch
    import numpy as np
    from tools.narrow_collection import environment
    from visuotactile.state_benchmark import trial
    from visuotactile.narrow_decision import failure_stage
    from visuotactile.video import verify_video
    from tools.state_mechanism_train import rollout_motion
    s=read(root/'session.json');plan=read(root/'plan.json');app,env,player,lease=environment(root,evaluation=True)
    jobs=[]
    for seed in s['inference_seeds']:jobs.append(('B_tiny',root/'tiny_training/best.pt',plan['developer_specs'][0],seed,0))
    for spec in plan['developer_specs']:
        for seed in s['inference_seeds']:
            for overlap in (0,4):jobs.append(('C'+str(overlap),Path(s['parent'])/'expanded_training/best.pt',spec,seed,overlap))
    results=[];initials={};folder=root/'closed_loop';folder.mkdir(exist_ok=False);saved=None;loaded=None
    for group,path,spec,seed,overlap in jobs:
        if loaded!=path:
            saved=torch.load(path,map_location='cpu',weights_only=False);loaded=path
        identity=f'{group}__{spec["family"]}__noise{seed}';out=folder/identity;beat(root,stage='closed_loop',identity=identity)
        r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],saved=saved,position=True,video=seed==7091,
            inference=dict(inference_steps=16,clip_sample=False,overlap_steps=overlap,sampler_seed=seed),
            video_title=f'{group} | update {saved["updates"]} | {spec["family"]} | noise {seed} | overlap {overlap} | 1x')
        metrics=rollout_motion(out) if r['legal'] else None
        if r['legal'] and seed==7091:r['video']=verify_video(out/'rollout.mp4',r['steps'])
        initial=read(out/'initial_observable.json');key=(spec['family'],seed)
        pair=max(float(abs(np.asarray(initial[k])-initials[key][k]).max()) for k in ('joint_position_rad','object_pose_w','hole_pose_w')) if key in initials else None
        initials[key]=initial
        results.append(dict(r,experiment=group,noise_seed=seed,overlap_steps=overlap,model_update=saved['updates'],
            motion=metrics,failure_stage=failure_stage(r),observable_pair_max=pair));write(root/'closed_loop_results.json',results)
    write(root/'closed_loop_result.json',dict(complete=True,starts=len(results),groups={g:dict(attempts=len(rr),
        legal=sum(r['legal'] for r in rr),successes=sum(r['success'] for r in rr)) for g in ('B_tiny','C0','C4') for rr in [[r for r in results if r['experiment']==g]]},
        generalization_claim=False,formal_default_overlap=0,hidden_physics_exact=False))


def run(root):
    s=read(root/'session.json');assert read(root/'data_audit.json')['passed']
    assert time.time()<s['deadline']-7200,'Preparation consumed diagnostic window; do not reset deadline'
    def jobs(worker):
        worker('gradient_gate',['-m','tools.state_mechanism_train','gradient','--root',str(root)],limit=min(s['deadline'],time.time()+1800))
        assert read(root/'gradient_gate.json')['passed']
        worker('oracle',['-m','tools.state_mechanism','oracle','--root',str(root)],kit=True,limit=min(s['deadline'],time.time()+2400))
        assert read(root/'oracle_result.json')['passed']
        worker('stochasticity',['-m','tools.state_mechanism_train','stochasticity','--root',str(root)],limit=min(s['deadline'],time.time()+1800))
        write(root/'tiny_budget.json',dict(deadline=min(s['deadline']-2700,time.time()+s['tiny_training_seconds']),max_updates=s['tiny_max_updates']))
        worker('tiny_training',['-m','tools.state_mechanism_train','train','--root',str(root)],limit=read(root/'tiny_budget.json')['deadline'])
        worker('closed_loop',['-m','tools.state_mechanism','closed-loop','--root',str(root)],kit=True)
        protected(root)
    try:
        execute(root,'diagnosis',jobs,s['deadline']);status(root,state='complete',stage='complete',formal_updates=0,
            diagnostic_updates=read(root/'tiny_training/finished.json')['updates'])
    finally:
        if read(root/'status.json').get('owned_workers_exited'):
            try:
                from tools.state_mechanism_delivery import finish
                finish(root)
            except BaseException:write(root/'publication_error.json',dict(traceback=traceback.format_exc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','oracle','closed-loop'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();kit=a.mode in ('oracle','closed-loop');code=0
    try:{'prepare':prepare,'run':run,'oracle':oracle,'closed-loop':closed_loop}[a.mode](root)
    except BaseException:
        write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc(),utc=time.time()))
        if not kit:raise
        code=1
    if kit:sys.stdout.flush();sys.stderr.flush();os._exit(code)
