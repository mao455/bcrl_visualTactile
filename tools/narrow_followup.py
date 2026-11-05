"""One bounded continuation, at most one coverage expansion and one fresh StateDP.

This owner is isolated from its predecessor. It never edits the predecessor,
retries a failed worker, uses final-test outcomes for selection, or loops back
from expanded training to collection. STOP preserves all evidence/checkpoints.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, proc, kill_owned
from tools.narrow_collection import freeze, check_freeze, environment, GATE
from visuotactile.narrow_decision import curve_summary, decide, failure_stage


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def update(root, **fields):
    old=read(root/'status.json') if (root/'status.json').exists() else {}
    write(root/'status.json',dict(old,**fields,utc=time.time()))


def beat(root, **fields):
    write(root/'heartbeat.json',dict(utc=time.time(),**fields))


def prepare(root, parent):
    from visuotactile.narrow_data import direct_configuration
    parent=parent.resolve();old=read(parent/'session.json')
    root.mkdir(parents=True,exist_ok=False)
    session=dict(version='narrow_followup_v1',parent=str(parent),
        parent_deadline=read(parent/'budget.json')['deadline'],
        waiting_deadline=read(parent/'budget.json')['deadline']+900,
        maximum_hours=8,collection_hours=3,extension_hours=1,expanded_training_hours=2.5,
        final_evaluation_hours=1.5,maximum_extra_episodes=96,maximum_evaluation_starts=52,
        maximum_output_bytes=45*1024**3,reserve_bytes=60*1024**3,
        expert_checkpoint=old['expert_checkpoint'],expert_sha256=old['expert_sha256'],
        domain=old['domain'],loss_weights=old['loss_weights'],resource_policy='shared_recent_v1',
        resource_target_percent=40.,validation_interval=500,original_manifest_sha256=old['old_manifest_sha256'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        maximum_worker_starts_per_stage=1,automatic_retries=False,
        fresh_seed=2026110601,publication_branch='feature/narrow-state-data')
    write(root/'session.json',session)
    def plan(prefix, seed, count):
        return [dict(identity=f'{prefix}_{i:02d}',seed=seed+i,cohort='fresh',split='unseen',collision_gate=GATE,
                     narrow_reset=direct_configuration(old['domain'],seed+i)) for i in range(count)]
    # Frozen before any student result. These 16 never enter a decision function.
    write(root/'final_unseen_plan.json',plan('final_unseen',26161000,16))
    write(root/'extension_unseen_plan.json',plan('extension_unseen',26162000,8))
    write(root/'candidate_plan.json',plan('extra_family',26163000,80))
    update(root,state='prepared',stage='dependency',formal_updates=0)


def predecessor_released(parent):
    status=read(parent/'status.json')
    if status['state'] in ('blocked','paused'):
        raise RuntimeError('Predecessor did not complete normally: '+str(status.get('last_error')))
    if not (parent/'PIPELINE_COMPLETE.json').exists() or not status.get('owned_workers_exited'):
        return False
    if status['state']!='complete':raise RuntimeError('Contradictory predecessor completion')
    owner=read(parent/'owner.json')
    for name in ('owner','worker','monitor'):
        expected=owner.get(name)
        if expected:
            actual=proc(expected['pid'])
            if actual and actual['start_ticks']==expected['start_ticks']:return False
    return True


def dataset_union(original, additions):
    old={e['id']:e for e in original}
    if len(old)!=len(original):raise ValueError('Duplicate original episode')
    split={e['family']:e['split'] for e in original}
    for e in additions:
        if e['id'] in old:raise ValueError('Duplicate/overwritten episode')
        if e['family'] in split and split[e['family']]!=e['split']:raise ValueError('Family leakage')
        old[e['id']]=e;split[e['family']]=e['split']
    return list(old.values())


def collect(root):
    import numpy as np
    from visuotactile.narrow_data import initial_row, near, quantiles, record, build_cache, admission_conflict
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.state_benchmark import prepare as initialize, trial
    from visuotactile.recording import model_fingerprint
    s=read(root/'session.json');parent=Path(s['parent']);old=read(parent/'families.json')
    check_freeze(root);beat(root,stage='collect_start');app,env,player,lease=environment(root)
    fingerprint=model_fingerprint(player.model);initials=root/'initials';initials.mkdir()
    pool=[];rejections=0
    for spec in read(root/'candidate_plan.json'):
        out=initials/spec['identity'];out.mkdir()
        with lease.block():
            _,legal=initialize(env,IsolatedEnvironment(env,spec['seed']),spec,{},out)
            row=initial_row(spec['identity'],{k:v[0].cpu().numpy() for k,v in env._diagnostics().items()},env.scene.env_origins[0].cpu().numpy())
        duplicate=any(near(row,f) for f in old+pool)
        write(out/'admission.json',dict(legal=legal['legal'],duplicate=duplicate))
        if legal['legal'] and not duplicate:pool.append(dict(row,spec=spec))
        else:rejections+=1
        beat(root,stage='family_admission',accepted=len(pool),rejected=rejections)
    if len(pool)<32:raise RuntimeError('Fewer than 32 distinct legal new families; no widening or replacement')
    # Geometric max-min coverage only; each feature group has equal aggregate
    # weight. No student result, speed outcome or final-test state is consulted.
    keys=('object_xyz','relative_xyz','quaternion','arm','hand','velocity')
    def features(row):
        return np.concatenate([np.asarray(row[k])/scales[k]/math.sqrt(len(np.asarray(row[k]))) for k in keys])
    old_train=[f for f in old if f['split']=='train']
    scales={k:np.maximum(np.std([f[k] for f in old_train],axis=0),1e-3) for k in keys}
    reference=np.stack([features(f) for f in old_train]);remaining=list(pool);chosen=[];distances=[]
    while len(chosen)<32:
        distances_to_old=[float(np.linalg.norm(reference-features(f),axis=1).min()) for f in remaining]
        index=int(np.argmax(distances_to_old));f=remaining.pop(index);chosen.append(f);distances.append(distances_to_old[index])
        reference=np.vstack([reference,features(f)])
    rng=np.random.default_rng(2026110602);order=rng.permutation(32)
    split={int(j):('validation' if i<4 else 'test' if i<8 else 'train') for i,j in enumerate(order)}
    for i,f in enumerate(chosen):f['split']=split[i]
    write(root/'families.json',old+chosen)
    write(root/'coverage.json',dict(new_families=32,candidates=80,legal_distinct=len(pool),initial_rejections=rejections,
        new_splits=dict(Counter(f['split'] for f in chosen)),maxmin_distances=distances,
        old_quantiles=quantiles(old,s['domain']['center']),new_quantiles=quantiles(chosen,s['domain']['center']),
        selection='geometry-only max-min within unchanged narrow distribution; no student outcomes'))
    jobs=[]
    for wave in range(3):
        for i in rng.permutation(32):
            f=chosen[int(i)];alpha=1. if wave==0 else float(rng.uniform(.58,.95))
            # Include the validated slow endpoint region without imposing a
            # discrete speed label or synthetic duration requirement.
            if wave==2 and int(i)%3==0:alpha=float(rng.uniform(.58,.65))
            jobs.append(dict(family=f['identity'],alpha=alpha,identity=f['identity']+f'_v{wave}'))
    write(root/'production_plan.json',jobs);byid={f['identity']:f for f in chosen}
    original=read(parent/'manifest.json')['episodes'];accepted=[];results=[];failures=0
    for job in jobs:
        if (root/'STOP_AFTER_EPISODE').exists():raise InterruptedError('Stopped after complete new episode')
        if time.time()>s['collection_deadline']-900 and len(accepted)>=60 and len({e['family'] for e in accepted})==32:break
        if time.time()>s['collection_deadline']-600:raise TimeoutError('Insufficient expanded data within collection budget')
        f=byid[job['family']];spec=dict(f['spec'],**job,split=f['split']);alpha=job['alpha']
        speed=dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all');env.speed_config=speed
        beat(root,stage='extra_episode',identity=spec['identity']);out=root/'episodes'/spec['identity']
        r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],expert_speed=speed,capture_dataset=True)
        results.append(r)
        write(root/'collection_counts.json',dict(attempts=len(results),successes=len(accepted),
            task_failures=sum(x['legal'] and not x['success'] for x in results),
            initial_rejections=sum(not x['legal'] for x in results),family_admission_rejections=rejections))
        if not r['legal']:raise RuntimeError('Admitted family became illegal; stop for contract review')
        mismatch=admission_conflict(read(initials/f['identity']/'initial_observable.json'),read(out/'initial_observable.json'),True,True)
        if max(mismatch['observable_max_abs'].values())>1e-5:raise RuntimeError('Family observable initialization changed')
        path=Path(r['dataset_capture']['path']);record(path);r['data_sha256']=sha(path);write(out/'result.json',r)
        failures=0 if r['success'] else failures+1
        if r['success']:accepted.append(dict(id=spec['identity'],family=f['identity'],split=f['split'],alpha=alpha,
            path=str(path),steps=r['steps'],sha256=r['data_sha256'],size=path.stat().st_size))
        write(root/'manifest.json',dict(version=2,collection_finished=False,episodes=dataset_union(original,accepted),
            parent_manifest_sha256=sha(parent/'manifest.json'),original400_included=False,quarantined=False))
        write(root/'collection_counts.json',dict(attempts=len(results),successes=len(accepted),total_successes=200+len(accepted),
            task_failures=sum(x['legal'] and not x['success'] for x in results),
            initial_rejections=sum(not x['legal'] for x in results),family_admission_rejections=rejections,
            new_successful_families=len({e['family'] for e in accepted})))
        if failures>=5:raise RuntimeError('Five consecutive real task failures')
    if len(accepted)<60 or len({e['family'] for e in accepted})<28:raise RuntimeError('Expanded data coverage/count gate failed')
    if model_fingerprint(player.model)!=fingerprint:raise RuntimeError('Expert changed')
    manifest=read(root/'manifest.json');manifest['collection_finished']=True;write(root/'manifest.json',manifest)
    beat(root,stage='expanded_numeric_acceptance');cache=build_cache(root)
    write(root/'DATA_READY.json',dict(data_ready=True,successes=len(manifest['episodes']),new_successes=len(accepted),
        new_families=len({e['family'] for e in accepted}),state_data=str(cache)))
    write(root/'COMPLETE.json',read(root/'DATA_READY.json'))


def audit(root):
    import numpy as np
    import torch
    import h5py
    from visuotactile.narrow_data import record
    from visuotactile.state_history import PersistentState, HistoryBank, HistoryController
    from visuotactile.state_resources import lease_for
    s=read(root/'session.json');parent=Path(s['parent']);lease=lease_for(root,deadline=s['deadline'])
    data=read(parent/'state_data/data.json');manifest=read(parent/'manifest.json')
    cache=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    assert cache['data_sha256']==sha(parent/'state_data/data.json')
    families={};checked=0
    for e in manifest['episodes']:
        beat(root,stage='audit_data',identity=e['id'])
        assert sha(e['path'])==e['sha256'] and Path(e['path']).stat().st_size==e['size']
        _,meta,rec=record(e['path'])
        assert meta['dataset_split']==e['split'] and meta['initial_family']==e['family']
        assert families.setdefault(e['family'],e['split'])==e['split']
        assert all(torch.equal(v,cache['records'][e['id']][k]) for k,v in rec.items())
        checked+=1
    stats=torch.load(parent/'state_data/statistics.pt',map_location='cpu',weights_only=False)
    for key in ('robot','geometry'):
        x=torch.cat([cache['records'][i][key][::12] if key=='geometry' else cache['records'][i][key] for i in data['splits']['train']])
        assert torch.equal(stats[key+'_mean'],x.mean(0)) and torch.equal(stats[key+'_std'],x.std(0).clamp_min(1e-4))
    from visuotactile.native_data import limits_affine
    scale,offset=limits_affine(torch.cat([cache['records'][i]['command'] for i in data['splits']['train']]))
    assert torch.equal(scale,stats['action_scale']) and torch.equal(offset,stats['action_offset'])
    with lease.block():
        saved=torch.load(parent/'training/best.pt',map_location='cpu',weights_only=False)
        assert read(parent/'evaluation/result.json')['checkpoint_sha256']==sha(parent/'training/best.pt')
        assert saved['configuration']['data_sha256']==sha(parent/'state_data/data.json')
        assert saved['configuration']['statistics_sha256']==sha(parent/'state_data/statistics.pt')
        model=PersistentState('dp',saved['contract']).cuda().eval();model.load_state_dict(saved['ema'])
        history_error=0.
        for identity in (data['splits']['train'][0],data['splits']['validation'][0]):
            rec={k:v.cuda() for k,v in cache['records'][identity].items()};bank=HistoryBank()
            with torch.no_grad():
                full,_,_=model.encode(rec)
                for t in range(len(rec['time'])):
                    c=bank.observe(0,1,t,model,rec['actor'][t],rec['robot'][t],rec['geometry'][t],rec['time'][t])
                    if c is not None:history_error=max(history_error,float((c-full[t//12:t//12+1]).abs().max()))
        assert history_error<2e-5
        del model
    # Reconstruct the deployed input path on recorded student observations. This
    # checks first and subsequent replan blocks, actual history and execution,
    # without any simulator state forcing or new autonomous start.
    online_error=0.;actuator_error=0.;input_error=0.;episodes=0
    candidates=[p for p in sorted((parent/'evaluation').glob('*/result.json')) if read(p).get('legal')]
    selected={p for split in ('train','unseen') for p in [p for p in candidates if read(p)['split']==split][:2]}
    for rp in candidates:
        result=read(rp);trace=np.load(rp.parent/'trace.npz');plans=np.load(rp.parent/'plans.npz')['plans']
        assert result['simulation_step_hz']==60 and result['physics_hz']==120 and result['plan_hz']==5
        assert result['post_success_actions']==0
        assert result['actor_normalization_max_error']<2e-5
        limits=np.asarray(read(rp.parent/'metadata.json')['joint_limits_rad'][0])
        assert plans.shape[1:]==(24,29)
        requested=np.stack([np.clip(plans[t//12,t%12],limits[:,0],limits[:,1]) for t in range(result['steps'])])
        assert np.max(np.abs(requested-trace['command']))<1e-6,'Plan/commit execution alignment'
        actuator_error=max(actuator_error,float(abs(trace['actuator_targets']-trace['target'][:,None,:]).max()),
                           float(abs(trace['command']-trace['target']).max()))
        assert actuator_error<1e-6
        if rp in selected:
            with lease.block():
                controller=HistoryController(saved,read(rp.parent/'metadata.json'),inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091))
            with h5py.File(rp.parent/'numeric.h5') as f:
                g=f['episode']
                for t in range(min(96,result['steps'])):
                    def tree(group):
                        return {k:tree(v) if isinstance(v,h5py.Group) else torch.from_numpy(np.array(v[t]))[None] for k,v in group.items()}
                    packet=tree(g['obs']);transition=tree(g['transition'])
                    with lease.block():
                        action=controller.act(torch.from_numpy(trace['actor'][t:t+1]),packet,transition['diagnostics_before'])
                        online_error=max(online_error,float(abs(action.numpy()[0]-trace['command'][t]).max()))
                        from visuotactile.native_data import robot_rows
                        diag=transition['diagnostics_before'];past=torch.zeros(61)
                        stamp=packet['time_s'][0]
                        if t:
                            age=float(g['obs/time_s'][t]-g['obs/time_s'][t-1])
                            past=torch.cat((torch.from_numpy(trace['command'][t-1]),
                                torch.from_numpy(np.array(g['transition/position_applied_rad'][t-1])),torch.tensor([age,age,1.])))
                        expected=robot_rows(diag['joint_position_rad'][0],diag['joint_velocity_rad_s'][0],
                            transition['target_before_rad'][0],past,stamp-float(g['obs/time_s'][0]))
                        input_error=max(input_error,float((expected-controller.last_inputs['robot']).abs().max()))
                        controller.observe_execution(transition)
                    beat(root,stage='audit_online',identity=rp.parent.name,tick=t)
            del controller;episodes+=1
        trace.close()
    assert online_error<2e-5,('Recorded online controller mismatch',online_error)
    assert input_error<2e-5,('Training/live proprio/history mismatch',input_error)
    assert episodes>=3
    write(root/'audit.json',dict(passed=True,data_episodes=checked,train_statistics_recomputed=True,
        labels_and_native_clock=True,history_max_abs=history_error,online_command_max_abs=online_error,
        training_live_robot_max_abs=input_error,
        actuator_target_max_abs=actuator_error,online_reconstructed_episodes=episodes,optimizer_updates=0,
        limitation='Observable/controller audit; does not establish exact hidden PhysX replay or causal data insufficiency'))


def evaluate(root, job):
    import torch
    import numpy as np
    from visuotactile.state_benchmark import trial
    s=read(root/'session.json');j=read(job);check_freeze(root);beat(root,stage='evaluation_start')
    app,env,player,lease=environment(root,True);output=Path(j['output']);output.mkdir(exist_ok=False)
    all_results=[];initials={};pair_errors={}
    for model in j['models']:
        path=Path(model['checkpoint']);assert sha(path)==model['sha256']
        with lease.block():saved=torch.load(path,map_location='cpu',weights_only=False)
        results=[]
        for spec in j['plan']:
            identity=model['name']+'__'+spec['identity'];beat(root,stage='evaluation',identity=identity)
            r=trial(env,lease,output/identity,spec,{},player,s['expert_checkpoint'],saved=saved,position=True,
                inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091))
            initial=read(output/identity/'initial_observable.json')
            if spec['identity'] in initials:
                keys=('joint_position_rad','joint_velocity_rad_s','object_pose_w','hole_pose_w','table_pose_w',
                      'palm_body_pose_w','finger_position_w','finger_quat_wxyz')
                pair_errors[identity]={k:float(np.max(np.abs(np.asarray(initial[k])-np.asarray(initials[spec['identity']][k])))) for k in keys}
            else:initials[spec['identity']]=initial
            results.append(r);all_results.append(dict(r,model=model['name'],failure_stage=failure_stage(r)))
            write(output/'progress.json',dict(completed=len(all_results),total=len(j['plan'])*len(j['models']),utc=time.time()))
            write(output/'results.json',all_results)
        del saved
    cohorts={}
    for model in j['models']:
        cohorts[model['name']]={}
        for split in ('train','unseen'):
            rr=[r for r in all_results if r['model']==model['name'] and r['split']==split]
            cohorts[model['name']][split]=dict(attempts=len(rr),valid=sum(r['legal'] for r in rr),
                successes=sum(r['success'] for r in rr),initial_rejections=sum(not r['legal'] for r in rr),
                failures=dict(Counter(failure_stage(r) for r in rr if not r['success'])))
    write(output/'result.json',dict(complete=True,models=cohorts,job_sha256=sha(job),
        final_test=j['final_test'],selection_allowed=not j['final_test'],software_missing=0,
        observable_pair_max_errors=pair_errors,exact_hidden_simulator_pairing_claimed=False))


def offline(root, job):
    """Equal original validation cohort for comparisons; never final test."""
    import torch
    import h5py
    from visuotactile.state_resources import lease_for
    from visuotactile.state_history import PersistentState
    from visuotactile.state_image_loss import GeometryObjective
    from visuotactile.state_image_offline import evaluate
    from visuotactile.anchor_continue import deterministic_loss, fresh_early_metrics
    s=read(root/'session.json');parent=Path(s['parent']);j=read(job);lease=lease_for(root,deadline=s['deadline'])
    data=read(parent/'state_data/data.json');cache=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    ids=data['splits']['validation'];train=data['splits']['train'][:8]
    with h5py.File(data['entries'][train[0]]['path']) as f:meta=json.loads(f.attrs['metadata_json'])
    objective=GeometryObjective(meta);objective.weights=s['loss_weights'];reports={}
    with lease.block():records={i:{k:v.cuda() for k,v in cache['records'][i].items()} for i in train+ids}
    for entry in j['models']:
        beat(root,stage='offline',model=entry['name'])
        assert sha(entry['checkpoint'])==entry['sha256']
        with lease.block():
            saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
            model=PersistentState('dp',saved['contract']).cuda().eval();model.load_state_dict(saved['ema'])
        reports[entry['name']]=dict(updates=saved['updates'],checkpoint_sha256=entry['sha256'],
            validation=evaluate(model,records,ids,objective,lease),train8=evaluate(model,records,train,objective,lease),
            validation_loss=deterministic_loss(model,records,ids,objective,lease),train8_loss=deterministic_loss(model,records,train,objective,lease),
            early=fresh_early_metrics(model,records,dict(train=train,validation=ids),objective,lease))
        del model,saved
    write(root/'offline_comparison.json',dict(models=reports,cohort='unchanged original 26 validation episodes and train8; no test',
        old_test_families_excluded=True))


def train_session(root, directory, data_root, target, deadline, continuation=None, minimum=None):
    s=read(root/'session.json');directory.mkdir(exist_ok=False)
    runtime=dict(resource_policy='shared_recent_v1',resource_target_percent=40.,external_monitor_root=str(root),
        training_deadline=deadline,validation_interval=500,random_seed=s['fresh_seed'],
        deterministic_validation_loss=True,early_stop_minimum_update=minimum or target)
    if continuation:runtime['continuation']=continuation
    write(directory/'session.json',runtime)
    write(directory/'budget.json',dict(target_updates=target,deadline=deadline,data=str(data_root),
        basis='measured predecessor convergence/anchors/throughput; fixed once before launch'))
    return ['-m','visuotactile.anchor_continue','fresh-train','--job','state','--root',str(directory),
            '--data',str(data_root),'--target',str(target),'--output',str(directory/'launch_1')]


def owner(root):
    s=read(root/'session.json');parent=Path(s['parent']);child=None;monitor=None
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'budget.json').exists():raise RuntimeError('Owner was already admitted; explicit recovery review required')
        if (root/'STOP').exists():raise InterruptedError('STOP present')
        freeze(root)
        update(root,state='waiting',stage='dependency',owner=proc(os.getpid()),owned_workers_exited=False)
        while not predecessor_released(parent):
            if time.time()>s['waiting_deadline'] or (root/'STOP').exists():raise TimeoutError('Bounded dependency wait ended')
            beat(root,stage='waiting_predecessor');time.sleep(15)
        # Hold the released predecessor lock read-only throughout this session.
        with (parent/'owner.lock').open('r') as parent_lock:
            fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            now=time.time();budget=dict(started=now,deadline=now+s['maximum_hours']*3600,
                parent_deadline_unchanged=read(parent/'budget.json')['deadline'])
            write(root/'budget.json',budget);s.update(budget,collection_deadline=budget['deadline'],
                owned_worker_roots=[str(root/'continuation'),str(root/'expanded_training')]);write(root/'session.json',s)
            write(root/'dependency_release.json',dict(parent_complete_sha256=sha(parent/'PIPELINE_COMPLETE.json'),
                parent_budget_sha256=sha(parent/'budget.json'),parent_manifest_sha256=sha(parent/'manifest.json'),utc=now))
            with (root/'monitor.log').open('w') as log:
                monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            def run(stage,args,seconds,kit=False):
                nonlocal child
                check_freeze(root);attempt=root/'workers'/stage;attempt.mkdir(parents=True,exist_ok=False)
                if kit:
                    from tools.kit_resource_preflight import inspect
                    pre=inspect(parent=root);write(attempt/'inotify.json',pre)
                    if not pre['resource_admitted']:raise RuntimeError('Inotify admission failed')
                deadline=min(s['deadline'],time.time()+seconds);beat(root,stage=stage)
                with (attempt/'worker.log').open('w') as log:
                    child=subprocess.Popen(RUN+args,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update(root,state='running',stage=stage,worker=proc(child.pid),monitor=proc(monitor.pid))
                last_disk=0
                while child.poll() is None:
                    if time.time()>deadline or (root/'STOP').exists():raise TimeoutError(stage+' deadline/STOP')
                    if monitor.poll() is not None:raise RuntimeError('Resource monitor exited')
                    if time.time()-last_disk>30:
                        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                        if size>s['maximum_output_bytes'] or shutil.disk_usage(root).free<s['reserve_bytes']:raise RuntimeError('Disk budget/reserve')
                        last_disk=time.time()
                    progress=[root/'heartbeat.json',root/'continuation/progress.json',root/'continuation/validation.json',
                        root/'expanded_training/progress.json',root/'expanded_training/validation.json']
                    progress+=list(root.glob('episodes/*/progress.json'))+list(root.glob('*evaluation/*/progress.json'))
                    if time.time()-max(p.stat().st_mtime for p in progress if p.exists())>900:raise TimeoutError('No durable progress for 900s')
                    time.sleep(2)
                write(attempt/'exit.json',dict(returncode=child.returncode,utc=time.time(),worker_absent=proc(child.pid) is None))
                if child.returncode:
                    update(root,software_worker_failures=1)
                    raise RuntimeError(stage+' failed; no automatic retry')
            def evaluation_job(name, models, plan, final):
                allocated=read(root/'evaluation_allocation.json') if (root/'evaluation_allocation.json').exists() else {'starts':0}
                allocated['starts']+=len(models)*len(plan)
                if allocated['starts']>s['maximum_evaluation_starts']:raise RuntimeError('Evaluation start budget')
                write(root/'evaluation_allocation.json',allocated)
                job=root/(name+'_job.json');write(job,dict(models=models,plan=plan,output=str(root/name),final_test=final))
                run(name,['-m','tools.narrow_followup','evaluate','--root',str(root),'--job',str(job)],s['final_evaluation_hours']*3600,True)
                return read(root/name/'result.json')
            try:
                run('audit',['-m','tools.narrow_followup','audit','--root',str(root)],2700)
                original_rows=rows(parent/'training/validation.jsonl');closed=read(parent/'evaluation/result.json')['cohorts']
                decision=decide(original_rows,closed,audit_passed=True);write(root/'decision_initial.json',decision)
                def model(name,path):return dict(name=name,checkpoint=str(path),sha256=sha(path))
                baseline=model('state200',parent/'training/best.pt');selected=baseline
                train_plan=[x for x in read(parent/'evaluation_plan.json') if x['split']=='train']
                current_rows=original_rows
                if decision['action']=='extend':
                    previous=read(parent/'training/progress.json');rate=previous['updates_per_hour'];step=previous['update']
                    extra=max(1000,min(int(rate*.65)//500*500,int(step*.5)//500*500))
                    directory=root/'continuation';donor=parent/'training/last.pt'
                    args=train_session(root,directory,parent/'state_data',step+extra,min(s['deadline']-7200,time.time()+3600),
                        dict(path=str(donor),sha256=sha(donor),configuration=read(parent/'training/configuration.json')),
                        minimum=step+min(1500,extra))
                    shutil.copy2(parent/'training/best.pt',directory/'best.pt')
                    run('continuation',args,3600)
                    selected=model('state200_continued',directory/'best.pt')
                    result=evaluation_job('extension_evaluation',[selected],train_plan+read(root/'extension_unseen_plan.json'),False)
                    current_rows=original_rows+rows(directory/'validation.jsonl')
                    decision=decide(current_rows,result['models'][selected['name']],extension_used=True,audit_passed=True)
                    write(root/'decision_after_extension.json',decision)
                models=[baseline] if selected==baseline else [baseline,selected]
                if decision['action']=='expand':
                    # No outcome-based retries or second expansion can enter here.
                    s['collection_deadline']=min(s['deadline']-4*3600,time.time()+s['collection_hours']*3600)
                    write(root/'session.json',s)
                    run('collection',['-m','tools.narrow_followup','collect','--root',str(root)],s['collection_deadline']-time.time(),True)
                    data=read(root/'state_data/data.json');ids=data['splits']['train']
                    anchors=sum(math.ceil(data['entries'][i]['length']/12) for i in ids);average=len(ids)/math.ceil(len(ids)/64)
                    old_budget=read(parent/'training/budget.json');last=read(parent/'training/progress.json')
                    # Exposure at measured plateau, with one validation interval
                    # headroom; throughput and fixed wall window also cap it.
                    plateau=next((r['update'] for i,r in enumerate(original_rows) if i>=3 and
                        curve_summary(original_rows[:i+1])['validation_gain']<.02),last['update'])
                    exposure=(plateau+500)*old_budget['mean_batch']/old_budget['anchors']
                    target=min(12000,math.ceil(exposure*anchors/average),int(last['updates_per_hour']*1.8))
                    deadline=min(s['deadline']-s['final_evaluation_hours']*3600,time.time()+s['expanded_training_hours']*3600)
                    args=train_session(root,root/'expanded_training',root/'state_data',target,deadline,minimum=max(2000,int(target*.6)))
                    write(root/'expanded_training/exposure_budget.json',dict(anchors=anchors,mean_batch=average,
                        predecessor_plateau_update=plateau,planned_anchor_epochs=exposure,target=target,fresh_random=True))
                    run('expanded_training',args,deadline-time.time())
                    expanded=model('state_expanded',root/'expanded_training/best.pt');models.append(expanded)
                    # The original200 is the primary comparator. Continued200,
                    # if any, retains its separate diagnostic and offline report.
                    final_models=[baseline,expanded]
                else:final_models=[selected]
                offline_job=root/'offline_job.json';write(offline_job,dict(models=models))
                run('offline_comparison',['-m','tools.narrow_followup','offline','--root',str(root),'--job',str(offline_job)],1800)
                final=None
                if decision['action']!='stop':
                    final=evaluation_job('final_evaluation',final_models,train_plan+read(root/'final_unseen_plan.json'),True)
                # Final test outcomes are reported only: no return to any stage.
                recommendation='State闭环仍不可靠，先审查失败时序/表示与训练在线差异；暂不启动ImageDP或触觉训练。'
                if final:
                    candidate=final['models'][final_models[-1]['name']]
                    u=candidate['unseen'];t=candidate['train']
                    if u['valid']>=12 and u['successes']/u['valid']>=.5 and t['successes']>=2:
                        recommendation='State已有有意义的小样本完整成功，可作为ImageDP参考；先通过严格live Image输入配对。触觉收益尚未验证，本轮不自动训练ImageDP或触觉。'
                release=read(root/'dependency_release.json')
                assert sha(parent/'budget.json')==release['parent_budget_sha256']
                assert sha(parent/'manifest.json')==release['parent_manifest_sha256']
                assert sha(CODE/'outputs/visuotactile/data/close_v4/manifest.json')==s['original_manifest_sha256']
                report=dict(complete=True,decision=decision,original_closed_loop=closed,
                    original_failure_stages={split:dict(Counter(failure_stage(read(p)) for p in
                        (parent/'evaluation').glob('*/result.json') if read(p)['split']==split)) for split in ('train','unseen')},
                    original_curve=curve_summary(original_rows),models=models,final_evaluation=final,
                    extension=read(root/'continuation/finished.json') if (root/'continuation/finished.json').exists() else None,
                    extension_evaluation=read(root/'extension_evaluation/result.json') if (root/'extension_evaluation/result.json').exists() else None,
                    expansion=read(root/'DATA_READY.json') if (root/'DATA_READY.json').exists() else None,
                    audit=read(root/'audit.json'),next_stage=recommendation,
                    utc=time.time())
                write(root/'RESULTS.json',report);update(root,state='complete',stage='complete')
            except BaseException as error:
                write(root/'owner_error.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=time.time()))
                update(root,state='blocked',last_error=repr(error))
                raise
            finally:
                if child is not None:kill_owned(child)
                (root/'STOP_MONITOR').touch()
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
                update(root,owned_workers_exited=True,worker=None)
            try:
                from tools.narrow_delivery import finish
                finish(root)
            except BaseException as error:
                write(root/'publication.json',dict(state='pending_review',error=repr(error),traceback=traceback.format_exc()))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','status','audit','collect','evaluate','offline'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--job',type=Path)
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='prepare':prepare(root,a.parent)
    elif a.mode=='run':
        try:owner(root)
        except BaseException as error:
            if not (root/'owner_error.json').exists():
                write(root/'owner_error.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=time.time()))
            update(root,state='blocked',last_error=repr(error),owned_workers_exited=True)
            raise
    elif a.mode=='status':
        result=read(root/'status.json')
        for stage in ('continuation','expanded_training'):
            if (root/stage/'progress.json').exists():result[stage]=read(root/stage/'progress.json')
        print(json.dumps(result,indent=2))
    else:
        try:
            if a.mode in ('evaluate','offline'):globals()[a.mode](root,a.job)
            else:globals()[a.mode](root)
        except BaseException:
            write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc(),utc=time.time()))
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__':main()
