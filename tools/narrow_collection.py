"""Frozen direct families -> real expert speeds -> accepted data -> fresh StateDP.

The ordinary owner enforces one fixed deadline, durable stage boundaries and no
implicit retries. Old quarantined v1 roots are never resumed by this pipeline.
"""
import argparse
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
from tools.runtime import CODE,RUN,read,write,sha,kill_owned,proc

GATE='live_collision_mesh_bounds_v1'
VERSION='narrow_state_pipeline_v2'


def freeze(root):
    write(root/'source_freeze.json',{str(p):sha(p) for folder in ('visuotactile','tools','isaacsimenvs','evaluation','rl_games')
        for p in sorted((CODE/folder).rglob('*.py'))})


def check_freeze(root):
    for p,digest in read(root/'source_freeze.json').items():
        if sha(p)!=digest:raise RuntimeError('Frozen source drift: '+p)


def prepare(root):
    from visuotactile.narrow_data import direct_configuration
    old=read(CODE/'outputs/visuotactile/native_dp_v2/anchor_continue_20261104_01/session.json')
    domain=read(CODE/'outputs/visuotactile/run/narrow_state_20261105_01/session.json')['domain']
    root.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for i in range(80):
        seed=25110000+i
        candidates.append(dict(identity=f'family_{i:03d}',seed=seed,cohort='fresh',collision_gate=GATE,
            narrow_reset=direct_configuration(domain,seed)))
    evaluation=[dict(identity=f'unseen_{i:02d}',seed=25120000+i,cohort='fresh',collision_gate=GATE,
        narrow_reset=direct_configuration(domain,25120000+i)) for i in range(12)]
    write(root/'candidate_plan.json',candidates);write(root/'unseen_plan.json',evaluation)
    session=dict(version=VERSION,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        expert_checkpoint=old['expert_checkpoint'],expert_sha256=old['expert_sha256'],domain=domain,
        distribution=dict(object_xy_halfwidth_m=[.025,.025],relative_xy_halfwidth_m=[.035,.035],rotation_radius_deg=20.,
            object_height_above_table_origin_m=[.244,.256],table_z_m=[.376,.384],arm_hand_native_lerp=.03,joint_velocity_halfwidth=.15),
        resource_policy='shared_recent_v1',resource_target_percent=40.,reserve_bytes=60*1024**3,maximum_output_bytes=75*1024**3,
        maximum_hours=16,collection_hours=3,training_hours=12,maximum_worker_starts=2,maximum_no_progress_s=600,
        target_families=60,minimum_families=50,target_successes=200,minimum_successes=150,maximum_collection_attempts=244,
        validation_interval=500,maximum_training_updates=12000,anchor_epochs=100,
        loss_weights=read(Path(old['old_root'])/'state_audit/calibration.json')['weights'],
        old_manifest_sha256=sha(CODE/'outputs/visuotactile/data/close_v4/manifest.json'),
        training_authorized=True,automatic_next_stage=True,initial_collision_gate=GATE)
    write(root/'session.json',session);write(root/'status.json',dict(state='prepared',stage='families',attempts=0,successes=0,data_ready=False,formal_updates=0))


def status_update(root,**fields):
    value=read(root/'status.json');value.update(fields,utc=time.time());write(root/'status.json',value)


def owner(root):
    s=read(root/'session.json')
    if s['version']!=VERSION:raise RuntimeError('Historical/quarantined root cannot enter the new pipeline')
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'PIPELINE_COMPLETE.json').exists():raise RuntimeError('Pipeline already complete')
        if (root/'STOP').exists() or (root/'STOP_AFTER_EPISODE').exists():raise RuntimeError('Stop marker present; explicit resume required')
        if not (root/'budget.json').exists():
            now=time.time();write(root/'budget.json',dict(started=now,deadline=now+16*3600,collection_deadline=now+3*3600));freeze(root)
        budget=read(root/'budget.json');s.update(budget,training_deadline=budget['deadline']);write(root/'session.json',s)
        check_freeze(root)
        if time.time()>=s['deadline']:raise TimeoutError('Original pipeline deadline exhausted')
        (root/'STOP_MONITOR').unlink(missing_ok=True)
        with (root/'monitor.log').open('a') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=None
        def run(stage,args,deadline,kit=False):
            nonlocal child
            directory=root/'workers'/stage;directory.mkdir(parents=True,exist_ok=True)
            index=len(list(directory.glob('attempt_*')))+1
            if index>s['maximum_worker_starts']:raise RuntimeError('Stage start budget exhausted: '+stage)
            attempt=directory/f'attempt_{index}';attempt.mkdir();check_freeze(root)
            if kit:
                from tools.kit_resource_preflight import inspect
                pre=inspect(parent=root);write(attempt/'inotify.json',pre)
                if not pre['resource_admitted']:raise RuntimeError('Inotify admission failed')
            with (attempt/'worker.log').open('w') as log:
                child=subprocess.Popen(RUN+args,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'owner.json',dict(owner=proc(os.getpid()),worker=proc(child.pid),monitor=proc(monitor.pid),stage=stage,attempt=str(attempt)))
            status_update(root,state='running',stage=stage,worker_pid=child.pid,owned_workers_exited=False)
            while child.poll() is None:
                if monitor.poll() is not None:raise RuntimeError('Resource monitor exited')
                if time.time()>deadline or (root/'STOP').exists():raise TimeoutError('Fixed deadline/STOP reached')
                progress=[attempt,root/'heartbeat.json',root/'training/progress.json',root/'training/validation.json']
                progress+=list(root.glob('episodes/*/progress.json'))+list(root.glob('evaluation/*/progress.json'))
                recent=max(p.stat().st_mtime for p in progress if p.exists())
                if time.time()-recent>s['maximum_no_progress_s']:raise TimeoutError('No effective stage progress for 600s')
                time.sleep(2)
            write(attempt/'exit.json',dict(returncode=child.returncode,ended=time.time(),worker_absent=proc(child.pid) is None))
            if child.returncode:raise RuntimeError(f'{stage} worker exited {child.returncode}; no automatic retry')
            if (root/'PAUSED.json').exists():raise InterruptedError('Paused at complete episode boundary')
        try:
            if not (root/'DATA_READY.json').exists():
                run('collection',['-m','tools.narrow_collection','worker','--root',str(root)],s['collection_deadline'],True)
            if not read(root/'DATA_READY.json')['data_ready']:raise RuntimeError('Dataset gate not passed')
            data=read(root/'state_data/data.json');lengths=[data['entries'][i]['length'] for i in data['splits']['train']]
            anchors=sum(math.ceil(n/12) for n in lengths);episodes=len(lengths)
            average_batch=episodes/math.ceil(episodes/64)
            target=min(s['maximum_training_updates'],math.ceil(s['anchor_epochs']*anchors/average_batch))
            train=root/'training';train.mkdir(exist_ok=True)
            if not (train/'session.json').exists():
                write(train/'session.json',dict(resource_policy='shared_recent_v1',resource_target_percent=40.,
                    training_deadline=min(s['deadline']-3600,time.time()+s['training_hours']*3600),
                    validation_interval=s['validation_interval'],reserve_bytes=s['reserve_bytes'],external_monitor_root=str(root)))
                write(train/'budget.json',dict(target_updates=target,anchors=anchors,train_episodes=episodes,
                    mean_batch=average_batch,target_anchor_epochs=s['anchor_epochs'],counts_from_accepted_data=True))
            if not (root/'disposable_gpu/check/result.json').exists():
                run('gpu_smoke',['-m','visuotactile.anchor_continue','fresh-smoke','--smoke-device','cuda:0','--job','state',
                    '--data',str(root/'state_data'),'--root',str(root/'disposable_gpu'),'--target','1','--output',str(root/'disposable_gpu/check')],s['deadline'])
            if not (train/'finished.json').exists():
                launch=train/f'launch_{len(list(train.glob("launch_*")))+1}'
                run('training',['-m','visuotactile.anchor_continue','fresh-train','--job','state','--data',str(root/'state_data'),
                    '--root',str(train),'--target',str(target),'--output',str(launch)],read(train/'session.json')['training_deadline'])
            status_update(root,formal_updates=read(train/'progress.json')['update'])
            if not (root/'evaluation/result.json').exists():
                run('evaluation',['-m','tools.narrow_collection','evaluate','--root',str(root)],s['deadline'],True)
            status_update(root,state='complete',stage='complete',closed_loop=read(root/'evaluation/result.json'))
            write(root/'PIPELINE_COMPLETE.json',read(root/'status.json'))
        except BaseException as error:
            write(root/'owner_error.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=time.time()))
            status_update(root,state='paused' if (root/'PAUSED.json').exists() else 'blocked',last_error=repr(error))
            raise
        finally:
            if child is not None:kill_owned(child)
            (root/'STOP_MONITOR').touch()
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(monitor)
            status_update(root,worker_pid=None,owned_workers_exited=True)


def environment(root,evaluation=False):
    from visuotactile.native_evaluation import client_startup_controls,make_env
    from visuotactile.state_resources import lease_for
    from visuotactile.expert import create_expert
    s=read(root/'session.json');lease=lease_for(root,deadline=s['deadline'] if evaluation else s['collection_deadline'])
    write(root/('eval_startup.json' if evaluation else 'startup.json'),client_startup_controls('kit_managed'))
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cpu',fast_shutdown=False,kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    import isaacsimenvs
    env=make_env(True,lease,scene_queries=True,expert_speed=None if evaluation else dict(kind='closed_loop',arm=1.,hand=1.,phase='all'))
    if sha(s['expert_checkpoint'])!=s['expert_sha256']:raise RuntimeError('Expert source drift')
    with lease.block():player,_=create_expert(env,s['expert_checkpoint'])
    return app,env,player,lease


def worker(root):
    import numpy as np
    from visuotactile.narrow_data import initial_row,near,quantiles,record,build_cache,admission_conflict
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.state_benchmark import prepare as initialize,trial
    from visuotactile.recording import model_fingerprint
    s=read(root/'session.json');check_freeze(root);started=time.time()
    def beat(stage,**more):write(root/'heartbeat.json',dict(stage=stage,utc=time.time(),**more))
    beat('startup');app,env,player,lease=environment(root);fingerprint=model_fingerprint(player.model)
    families=[];initials=root/'initials';initials.mkdir(exist_ok=True)
    if (root/'families.json').exists():families=read(root/'families.json')
    else:
        for spec in read(root/'candidate_plan.json'):
            out=initials/spec['identity'];out.mkdir(exist_ok=True)
            with lease.block():
                _,legal=initialize(env,IsolatedEnvironment(env,spec['seed']),spec,{},out)
                row=initial_row(spec['identity'],{k:v[0].cpu().numpy() for k,v in env._diagnostics().items()},env.scene.env_origins[0].cpu().numpy())
            duplicate=any(near(row,f) for f in families)
            write(out/'admission.json',dict(legal=legal['legal'],duplicate=duplicate))
            if legal['legal'] and not duplicate:families.append(dict(row,spec=spec))
            beat('families',accepted=len(families),examined=len(list(initials.iterdir())))
            if len(families)>=s['target_families']:break
        if len(families)<s['minimum_families']:raise RuntimeError('Direct domain lacks sufficient valid families; no broad sampling or widening')
        rng=np.random.default_rng(20261105);order=rng.permutation(len(families));split={int(j):('validation' if i<8 else 'test' if i<16 else 'train') for i,j in enumerate(order)}
        for i,f in enumerate(families):f['split']=split[i]
        write(root/'families.json',families)
        write(root/'initial_coverage.json',dict(families=len(families),examined=len(list(initials.iterdir())),
            split_families={k:sum(f['split']==k for f in families) for k in ('train','validation','test')},
            quantiles=quantiles(families,s['domain']['center']),direct_sampling=True))
        write(root/'evaluation_plan.json',[dict(f['spec'],identity='train_'+f['identity'],family=f['identity'],split='train')
            for f in families if f['split']=='train'][:4]+[dict(x,split='unseen') for x in read(root/'unseen_plan.json')])
    byid={f['identity']:f for f in families};episodes=root/'episodes';episodes.mkdir(exist_ok=True)
    results={p.parent.name:read(p) for p in episodes.glob('*/result.json')}
    def publish(finished=False):
        accepted=[]
        for identity,r in results.items():
            if r.get('success') and r.get('data_sha256'):
                p=Path(r['dataset_capture']['path']);accepted.append(dict(id=identity,family=r['family'],split=r['split'],alpha=r['alpha'],
                    path=str(p),steps=r['steps'],sha256=r['data_sha256'],size=p.stat().st_size))
        write(root/'manifest.json',dict(version=2,collection_finished=finished,episodes=accepted,families_sha256=sha(root/'families.json'),
            original400_included=False,quarantined=False))
        status_update(root,stage='collection',families=len(families),attempts=len(results),successes=len(accepted),
            task_failures=sum(r.get('legal',False) and not r.get('success',False) for r in results.values()),
            initial_rejections=sum(not r.get('legal',False) for r in results.values()),data_ready=False,
            total_control_steps=sum(e['steps'] for e in accepted),bytes=sum(e['size'] for e in accepted),
            collection_wall_s=time.time()-started,episodes_per_hour=len(results)*3600/max(1,time.time()-started))
        return accepted
    def one(f,tag,alpha):
        identity=f['identity']+'_'+tag
        if identity in results:
            if results[identity].get('legal') and not results[identity].get('data_sha256'):raise RuntimeError('Unvalidated prior attempt; preserve for review')
            return results[identity]
        if (root/'STOP_AFTER_EPISODE').exists():
            write(root/'PAUSED.json',dict(next_identity=identity,utc=time.time()));return None
        if shutil.disk_usage(root).free<s['reserve_bytes']:raise RuntimeError('60GiB disk reserve reached')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>s['maximum_output_bytes']:raise RuntimeError('75GiB output budget reached')
        if len(results)>=s['maximum_collection_attempts']:raise RuntimeError('Fixed attempt budget exhausted')
        check_freeze(root);beat('episode',identity=identity)
        speed=dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all');env.speed_config=speed
        spec=dict(f['spec'],identity=identity,family=f['identity'],split=f['split'],alpha=alpha)
        wall=time.time();r=trial(env,lease,episodes/identity,spec,{},player,s['expert_checkpoint'],expert_speed=speed,capture_dataset=True)
        reference=read(initials/f['identity']/'initial_observable.json');actual=read(episodes/identity/'initial_observable.json')
        comparison=admission_conflict(reference,actual,True,r['legal'])
        if max(comparison['observable_max_abs'].values())>1e-5 or not r['legal']:
            write(episodes/identity/'family_mismatch.json',comparison);raise RuntimeError('Native family reset changed observable geometry or clearance')
        path=Path(r['dataset_capture']['path']);record(path);r.update(data_sha256=sha(path),wall_s=time.time()-wall)
        write(episodes/identity/'result.json',r);results[identity]=r;publish();beat('saved',identity=identity)
        return r
    pilots=[f for f in families if f['split']=='train'][:2]
    for f in pilots:
        for tag,alpha in [('native',1.),('cal058',.58)]:
            r=one(f,tag,alpha)
            if r is None:return
            if tag=='native' and not r['success']:raise RuntimeError('Native pilot failure before production')
    lower=.58 if all(results[f['identity']+'_cal058']['success'] for f in pilots) else .65
    if lower==.65:
        for f in pilots:
            r=one(f,'cal065',.65)
            if r is None:return
            if not r['success']:raise RuntimeError('Single bounded slow endpoint adjustment failed')
    ratios=[results[f['identity']+('_cal058' if lower==.58 else '_cal065')]['steps']/results[f['identity']+'_native']['steps'] for f in pilots]
    write(root/'calibration.json',dict(lower_alpha=lower,duration_ratios=ratios,passed=True))
    if not (root/'production_plan.json').exists():
        rng=np.random.default_rng(20261106);jobs=[]
        for i,f in enumerate(families):
            jobs.extend([dict(family=f['identity'],tag='native',alpha=1.),
                dict(family=f['identity'],tag='mid_a',alpha=float(rng.uniform(max(lower,.67),.95))),
                dict(family=f['identity'],tag='mid_b',alpha=float(rng.uniform(max(lower,.67),.95)))])
            if i%3!=2:jobs.append(dict(family=f['identity'],tag='slow',alpha=float(rng.uniform(lower,min(.72,lower+.07)))))
        # Mix families/splits; all are predetermined, never selected by outcome.
        rng.shuffle(jobs);write(root/'production_plan.json',dict(jobs=jobs,maximum_attempts=len(jobs)+4,lower_alpha=lower))
    write(root/'SHORT_SUPERVISION_READY.json',dict(passed=True,successful_pilots=sum(r['success'] for r in results.values()),
        actual_duration_ratios=ratios,families_sha256=sha(root/'families.json'),utc=time.time()))
    # Human/model startup verification is the only manual boundary. Authorized
    # data/training/evaluation transitions below are entirely automatic.
    until=time.time()+600
    while not (root/'SUPERVISION_ACCEPTED.json').exists():
        if time.time()>until:raise TimeoutError('Startup verification absent after 10min')
        lease.check();beat('short_supervision');time.sleep(2)
    failures=0
    for job in read(root/'production_plan.json')['jobs']:
        r=one(byid[job['family']],job['tag'],job['alpha'])
        if r is None:return
        failures=0 if r['success'] else failures+1
        if failures>=5:raise RuntimeError('Five consecutive task failures; stop without replacement')
        accepted=read(root/'manifest.json')['episodes']
        covered={e['family'] for e in accepted}
        if len(accepted)>=s['target_successes'] and len(covered)==len(families):break
    accepted=publish(True);counts={k:len({e['family'] for e in accepted if e['split']==k}) for k in ('train','validation','test')}
    if len(accepted)<s['minimum_successes'] or counts['train']<32 or min(counts['validation'],counts['test'])<6:
        raise RuntimeError('Insufficient successful data/family coverage; formal training not admitted')
    if model_fingerprint(player.model)!=fingerprint:raise RuntimeError('Expert parameters changed')
    if sha(CODE/'outputs/visuotactile/data/close_v4/manifest.json')!=s['old_manifest_sha256']:raise RuntimeError('Original manifest changed')
    beat('numeric_acceptance');cache=build_cache(root)
    status_update(root,data_ready=True,stage='data_ready',split_families=counts,state_data=str(cache))
    write(root/'DATA_READY.json',read(root/'status.json'));write(root/'COMPLETE.json',read(root/'status.json'))


def evaluate(root):
    import torch
    from visuotactile.state_benchmark import trial
    s=read(root/'session.json');check_freeze(root);app,env,player,lease=environment(root,True)
    path=root/'training/best.pt'
    with lease.block():saved=torch.load(path,map_location='cpu',weights_only=False)
    output=root/'evaluation';output.mkdir(exist_ok=True);results=[]
    for spec in read(root/'evaluation_plan.json'):
        out=output/spec['identity']
        if (out/'result.json').exists():r=read(out/'result.json')
        else:
            write(root/'heartbeat.json',dict(stage='evaluation',identity=spec['identity'],utc=time.time()))
            r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],saved=saved,position=True,
                inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091))
        results.append(r)
    cohorts={k:dict(attempts=sum(r['split']==k for r in results),valid=sum(r['split']==k and r['legal'] for r in results),
        successes=sum(r['split']==k and r['success'] for r in results),initial_rejections=sum(r['split']==k and not r['legal'] for r in results)) for k in ('train','unseen')}
    write(output/'result.json',dict(complete=True,cohorts=cohorts,checkpoint_sha256=sha(path),updates=saved['updates'],
        success_definition='unchanged ordered native full success',unseen_plan_sha256=sha(root/'evaluation_plan.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','worker','evaluate','status','stop','resume']);p.add_argument('--root',required=True,type=Path)
    a=p.parse_args();r=a.root.resolve()
    if a.mode=='prepare':prepare(r)
    elif a.mode=='status':
        value=read(r/'status.json')
        if (r/'training/progress.json').exists():value['training']=read(r/'training/progress.json')
        print(json.dumps(value,ensure_ascii=False,indent=2))
    elif a.mode=='stop':(r/'STOP_AFTER_EPISODE').touch();(r/'training/STOP').touch() if (r/'training').exists() else None
    elif a.mode in ('run','resume'):
        if a.mode=='resume':
            for name in ('STOP','STOP_AFTER_EPISODE','PAUSED.json','training/STOP'):(r/name).unlink(missing_ok=True)
        owner(r)
    else:
        try:(worker if a.mode=='worker' else evaluate)(r)
        except BaseException:
            write(r/f'{a.mode}_error.json',dict(traceback=traceback.format_exc(),utc=time.time()));sys.stdout.flush();sys.stderr.flush();os._exit(1)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__':main()
