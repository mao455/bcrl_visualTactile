"""Bounded persistent-Kit collection; no training or evaluation stage transition."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, kill_owned, proc


def prepare(root):
    from visuotactile.narrow_data import select
    source=CODE/'outputs/visuotactile/native_dp_3way/native_20261101_02'
    domain,families,report=select(source)
    root.mkdir(parents=True,exist_ok=False)
    source_data=read(source/'data.json')
    for i,f in enumerate(families):
        f.update(entry=source_data['entries'][f['identity']],seed=24110000+i,source='original_observable_initial')
    # Fill held-out coverage with nine native draws, never with successful-policy
    # selection. IDs/splits/seeds are fixed before any speed expansion.
    extra=[dict(identity=f'new_{i:02d}',split=s,seed=24120000+i*10000,source='conditional_native_reset')
        for i,s in enumerate(['train']*4+['validation']*3+['test']*2)]
    old=read(CODE/'outputs/visuotactile/native_dp_v2/anchor_continue_20261104_01/session.json')
    session=dict(version='narrow_state_collection_v1',source_root=str(source.resolve()),domain=domain,
        historical=families,supplemental=extra,maximum_reset_draws_per_family=500,
        expert_checkpoint=old['expert_checkpoint'],expert_sha256=old['expert_sha256'],
        resource_policy='shared_recent_v1',resource_target_percent=40.,reserve_bytes=60*1024**3,
        maximum_output_bytes=50*1024**3,maximum_hours=4,maximum_worker_starts=2,
        maximum_no_progress_s=240,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        velocity_interval='one alpha per episode, same arm/hand, continuous; calibration .58 then at most .60',
        loss_weights=read(Path(old['old_root'])/'state_audit/calibration.json')['weights'],
        old_manifest_sha256=sha(CODE/'outputs/visuotactile/data/close_v4/manifest.json'),
        training_authorized=False,automatic_next_stage=False)
    write(root/'session.json',session);write(root/'selection.json',dict(domain=domain,families=families,**report))
    write(root/'status.json',dict(state='prepared',successes=0,attempts=0,data_ready=False))


def freeze(root):
    paths=[]
    for folder in ('visuotactile','tools','isaacsimenvs','evaluation','rl_games'):
        paths.extend(sorted((CODE/folder).rglob('*.py')))
    write(root/'source_freeze.json',{str(p):sha(p) for p in paths})


def check_freeze(root):
    for p,digest in read(root/'source_freeze.json').items():
        if sha(p)!=digest: raise RuntimeError('Frozen source drift: '+p)


def owner(root):
    root=root.resolve();session=read(root/'session.json')
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'COMPLETE.json').exists(): raise RuntimeError('Already complete; no duplicate starts')
        if (root/'manifest.json').exists() and read(root/'manifest.json').get('quarantined'):
            raise RuntimeError('Quarantined admission conflict requires reviewed correction, not automatic resume')
        if (root/'STOP').exists(): raise RuntimeError('STOP present; explicit resume required')
        if not (root/'budget.json').exists():
            write(root/'budget.json',dict(started=time.time(),deadline=time.time()+session['maximum_hours']*3600))
            freeze(root)
        check_freeze(root)
        budget=read(root/'budget.json'); session.update(deadline=budget['deadline'],training_deadline=budget['deadline'])
        write(root/'session.json',session)
        if time.time()>=session['deadline']: raise TimeoutError('Original collection deadline exhausted')
        attempts=root/'workers';attempts.mkdir(exist_ok=True)
        ordinal=len(list(attempts.iterdir()))+1
        if ordinal>session['maximum_worker_starts']: raise RuntimeError('Worker-start budget exhausted')
        attempt=attempts/f'attempt_{ordinal}';attempt.mkdir()
        (root/'STOP_MONITOR').unlink(missing_ok=True)
        monitor=None;worker=None
        try:
            from tools.kit_resource_preflight import inspect
            admission=inspect(parent=root);write(attempt/'inotify.json',admission)
            if not admission['resource_admitted']: raise RuntimeError('Inotify admission failed')
            with (attempt/'monitor.log').open('w') as log:
                monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with (attempt/'worker.log').open('w') as log:
                worker=subprocess.Popen(RUN+['-m','tools.narrow_collection','worker','--root',str(root),'--attempt',str(attempt)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'owner.json',dict(owner=proc(os.getpid()),worker=proc(worker.pid),monitor=proc(monitor.pid),attempt=str(attempt)))
            while worker.poll() is None:
                if monitor.poll() is not None: raise RuntimeError('Resource monitor exited')
                if time.time()>session['deadline']: raise TimeoutError('Fixed collection deadline reached')
                heartbeat=root/'heartbeat.json'
                # progress.json is updated during a real physical episode.
                recent=max([attempt.stat().st_mtime]+[p.stat().st_mtime for p in root.glob('episodes/*/progress.json')]
                    +([heartbeat.stat().st_mtime] if heartbeat.exists() else []))
                if time.time()-recent>session['maximum_no_progress_s']: raise TimeoutError('No physics/save/reset progress for 240s')
                time.sleep(1)
            write(attempt/'exit.json',dict(returncode=worker.returncode,ended=time.time(),worker_absent=proc(worker.pid) is None))
            if worker.returncode: raise RuntimeError(f'Worker exited {worker.returncode}; no automatic retry')
            if not (root/'COMPLETE.json').exists() and not (root/'PAUSED.json').exists(): raise RuntimeError('Missing durable completion/pause')
        except BaseException as error:
            write(attempt/'owner_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            status=read(root/'status.json');status.update(state='blocked',last_error=repr(error),data_ready=False)
            write(root/'status.json',status)
            raise
        finally:
            if worker is not None:kill_owned(worker)
            (root/'STOP_MONITOR').touch()
            if monitor is not None:
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
            status=read(root/'status.json');status.update(worker_pid=None,owned_workers_exited=True)
            write(root/'status.json',status)


def snapshot_initial(env,path,stream):
    import h5py
    import torch
    metadata=env.research_metadata();metadata['config']=env.cfg.to_dict()
    metadata['env_origins_w']=env.scene.env_origins.cpu().tolist()
    with h5py.File(path,'x') as f:
        f.attrs['metadata_json']=json.dumps(metadata,default=str)
        g=f.create_group('episodes/initial')
        for name,value in env._diagnostics().items():g.create_dataset('transition/diagnostics_before/'+name,data=value.cpu().numpy())
        g.create_dataset('obs/actuator_target_rad',data=env.research_packet['actuator_target_rad'].cpu().numpy())
        g.create_dataset('transition/external_force_probability',data=env._random_force_prob.cpu().numpy())
        g.create_dataset('transition/external_torque_probability',data=env._random_torque_prob.cpu().numpy())
        g.create_dataset('transition/torch_rng_before_step',data=stream.state[2][None].numpy())
    return dict(path=str(path),episode='initial',source_sha256=sha(path))


def worker(root,attempt):
    import numpy as np
    import torch
    from visuotactile.narrow_data import initial_row,inside,near,record,build_cache,quantiles
    from visuotactile.native_evaluation import client_startup_controls,make_env
    from visuotactile.expert import create_expert
    from visuotactile.recording import model_fingerprint
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.state_resources import lease_for
    root=root.resolve();s=read(root/'session.json');check_freeze(root)
    assert sha(s['expert_checkpoint'])==s['expert_sha256']
    lease=lease_for(root,deadline=s['deadline'])
    def beat(stage,**more):write(root/'heartbeat.json',dict(stage=stage,utc=time.time(),**more))
    beat('kit_startup')
    write(attempt/'startup_controls.json',client_startup_controls('kit_managed'))
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cpu',fast_shutdown=False,kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    import isaacsimenvs
    from visuotactile.state_benchmark import trial,collision_evidence,prepare as prepare_initial
    env=make_env(True,lease,scene_queries=True,expert_speed=dict(kind='closed_loop',arm=1.,hand=1.,phase='all'))
    with lease.block():player,_=create_expert(env,s['expert_checkpoint'])
    fingerprint=model_fingerprint(player.model)
    beat('initial_families')
    families_path=root/'families.json'
    if families_path.exists():families=read(families_path)
    else:
        families=list(s['historical']);snapshot_dir=root/'initials';snapshot_dir.mkdir(exist_ok=True)
        for extra in s['supplemental']:
            marker=snapshot_dir/(extra['identity']+'.json')
            if marker.exists():families.append(read(marker));continue
            accepted=None
            for k in range(s['maximum_reset_draws_per_family']):
                seed=extra['seed']+k;isolated=IsolatedEnvironment(env,seed)
                with lease.block():
                    isolated.reset(seed)
                    d={name:value[0].cpu().numpy() for name,value in env._diagnostics().items()}
                    row=initial_row(extra['identity'],d,env.scene.env_origins[0].cpu().numpy())
                    candidate=inside(row,s['domain']) and not any(near(row,f) for f in families)
                    legal=collision_evidence(env) if candidate else None
                beat('initial_families',family=extra['identity'],draw=k+1)
                with (root/'reset_draws.jsonl').open('a') as log:
                    log.write(json.dumps(dict(family=extra['identity'],seed=seed,inside=candidate,
                        legal=None if legal is None else legal['legal']))+'\n')
                if candidate and legal['legal']:
                    path=snapshot_dir/(extra['identity']+'.h5')
                    if path.exists():raise RuntimeError('Uncommitted initial snapshot; preserve and review')
                    accepted=dict(row,**{k:v for k,v in extra.items() if k!='identity'},reset_seed=seed,
                        reset_draws=k+1,entry=snapshot_initial(env,path,isolated.stream))
                    write(marker,accepted);families.append(accepted);break
            if accepted is None:raise RuntimeError('Fixed reset-draw budget exhausted; do not widen domain')
        write(families_path,families)
        write(root/'initial_coverage.json',dict(families=len(families),splits={k:sum(f['split']==k for f in families)
            for k in ('train','validation','test')},quantiles=quantiles(families,s['domain']['center'])))
    entries={f['identity']:f['entry'] for f in families}
    for f in families:
        if sha(f['entry']['path'])!=f['entry']['source_sha256']:raise RuntimeError('Initial source drift')
    (root/'episodes').mkdir(exist_ok=True)
    results={p.parent.name:read(p) for p in (root/'episodes').glob('*/result.json')}
    for identity,r in results.items():
        if r.get('legal') and 'data_sha256' not in r:
            path=Path(r['dataset_capture']['path']);record(path)
            r['data_sha256']=sha(path);write(root/'episodes'/identity/'result.json',r)
    # Historical collection checked object/table/hole boxes, not hand meshes.
    # Admit every restored family before choosing calibration cases or generating
    # any speed jobs. Exclusion is pre-action geometry, never task performance.
    preflight=root/'family_preflight';preflight.mkdir(exist_ok=True)
    eligible=[];excluded=[]
    for f in families:
        output=preflight/f['identity'];receipt=output/'admission.json'
        if receipt.exists():admission=read(receipt)
        else:
            prior=results.get(f['identity']+'_native')
            if prior is not None:
                admission=dict(legal=prior['legal'],source='existing pre-action live collision evidence',
                    evidence=str(root/'episodes'/(f['identity']+'_native')/'initial_collision.json'))
                output.mkdir(exist_ok=True)
            else:
                output.mkdir(exist_ok=False);isolated=IsolatedEnvironment(env,f['seed'])
                with lease.block():
                    _,legal=prepare_initial(env,isolated,dict(identity=f['identity'],cohort='historical',seed=f['seed']),
                        dict(entries=entries),output)
                admission=dict(legal=legal['legal'],source='live restored PhysX collision before any action')
            write(receipt,admission)
        (eligible if admission['legal'] else excluded).append(f)
        beat('family_preflight',family=f['identity'])
    families=eligible
    write(root/'eligible_families.json',dict(families=families,excluded=[f['identity'] for f in excluded],
        rule='unchanged live collision check before actions; no outcome selection, no replacement families'))
    if min(sum(f['split']==k for f in families) for k in ('train','validation','test'))<2:
        raise RuntimeError('Insufficient legal families after mesh admission; no automatic supplementation')
    def publish(finished=False):
        episodes=[]
        for identity,r in results.items():
            if r.get('success') and r.get('dataset_capture',{}).get('accepted_success'):
                p=Path(r['dataset_capture']['path'])
                episodes.append(dict(id=identity,family=r['family'],split=r['split'],alpha=r['alpha'],
                    path=str(p),steps=r['steps'],sha256=r['data_sha256'],size=p.stat().st_size))
        write(root/'manifest.json',dict(version=1,collection_finished=finished,episodes=episodes,families_sha256=sha(families_path)))
        state=dict(state='collecting',attempts=len(results),successes=len(episodes),task_failures=sum(r.get('legal',False) and not r.get('success',False) for r in results.values()),
            illegal_initials=sum(not r.get('legal',False) for r in results.values()),data_ready=False,
            covered_families=len({e['family'] for e in episodes}),bytes=sum(e['size'] for e in episodes),
            free_disk_bytes=shutil.disk_usage(root).free,utc=time.time(),worker_pid=os.getpid(),
            admitted_families=len(families),excluded_initial_families=len(excluded))
        write(root/'status.json',state)
        return state
    def one(f,tag,alpha):
        identity=f['identity']+'_'+tag
        if identity in results:return results[identity]
        if (root/'STOP_AFTER_EPISODE').exists():
            write(root/'PAUSED.json',dict(next_identity=identity,utc=time.time()));return None
        if shutil.disk_usage(root).free<s['reserve_bytes']:raise RuntimeError('60GiB disk reserve reached')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>s['maximum_output_bytes']:raise RuntimeError('50GiB collection budget reached')
        check_freeze(root);beat('episode',identity=identity)
        output=root/'episodes'/identity
        if output.exists():raise RuntimeError('Uncommitted attempt exists; no silent overwrite/retry: '+identity)
        speed=dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all');env.speed_config=speed
        spec=dict(identity=identity,family=f['identity'],split=f['split'],cohort='historical',seed=f['seed'],alpha=alpha)
        result=trial(env,lease,output,spec,dict(entries={identity:entries[f['identity']]}),player,s['expert_checkpoint'],
            expert_speed=speed,capture_dataset=True)
        from visuotactile.narrow_data import admission_conflict
        admission=read(preflight/f['identity']/'admission.json')
        reference=(Path(admission['evidence']).parent if 'evidence' in admission else preflight/f['identity'])
        conflict=admission_conflict(read(reference/'initial_observable.json'),read(output/'initial_observable.json'),
            admission['legal'],result['legal'])
        if conflict['conflict']:
            results[identity]=result;publish()
            write(output/'admission_conflict.json',conflict)
            raise RuntimeError('Identical observable initial state has contradictory collision admission; stop before further attempts')
        if result.get('legal'):
            path=Path(result['dataset_capture']['path']);record(path)
            result['data_sha256']=sha(path);write(output/'result.json',result)
        results[identity]=result;publish();beat('saved',identity=identity)
        return result
    training=[f for f in families if f['split']=='train']
    pilots=sorted(training,key=lambda f:sum((np.array(f['object_xyz'])-s['domain']['center']['object_xyz'])**2))[:2]
    # Outcomes select only the already bounded speed interval, never families/split.
    for f in pilots:
        r=one(f,'native',1.)
        if r is None:return
        if not r.get('success'):raise RuntimeError('Native calibration failure; stop before production')
    for f in pilots:
        if one(f,'cal058',.58) is None:return
    lower=.58 if all(results[f['identity']+'_cal058'].get('success') for f in pilots) else .60
    if lower==.60:
        for f in pilots:
            r=one(f,'cal060',.60)
            if r is None:return
            if not r.get('success'):raise RuntimeError('Bounded slow calibration failed')
    gate=dict(passed=True,lower_alpha=lower,pilot_families=[f['identity'] for f in pilots],formal_training_updates=0)
    write(root/'calibration.json',gate)
    if not (root/'production_plan.json').exists():
        rng=np.random.default_rng(20261105);jobs=[]
        for f in families:
            jobs.extend([dict(family=f['identity'],tag='native',alpha=1.),
                dict(family=f['identity'],tag='continuous',alpha=float(rng.uniform(lower,.95)))])
        # Balanced 2/3 multiplicity, no family contributes more than three accepted
        # production variants. Calibration remains separately identified.
        slow=[]
        for split in ('train','validation','test'):
            fs=[f for f in families if f['split']==split]
            slow.extend(fs[:max(1,round(2*len(fs)/3))])
        for f in slow:jobs.append(dict(family=f['identity'],tag='slow',alpha=float(rng.uniform(lower,min(.64,lower+.06)))))
        # Pilot successes are calibration evidence, excluded from production
        # except their native reference, which is precisely the same contract.
        write(root/'production_plan.json',dict(jobs=jobs,lower_alpha=lower,maximum_attempts=len(jobs)+4,
            maximum_hours=s['maximum_hours'],families_sha256=sha(families_path),
            eligible_families_sha256=sha(root/'eligible_families.json'),source_sha256=sha(root/'source_freeze.json')))
    beat('awaiting_short_supervision')
    write(root/'SHORT_SUPERVISION_READY.json',dict(**gate,manifest_sha256=sha(root/'manifest.json')))
    until=time.time()+180
    while not (root/'SUPERVISION_ACCEPTED.json').exists():
        if time.time()>until:raise TimeoutError('Short supervision not accepted within 180s; no unattended release')
        lease.check();beat('awaiting_short_supervision');time.sleep(1)
    planned=read(root/'production_plan.json');family_map={f['identity']:f for f in families}
    consecutive=0
    for job in planned['jobs']:
        r=one(family_map[job['family']],job['tag'],job['alpha'])
        if r is None:return
        consecutive=0 if r.get('success') else consecutive+1
        if consecutive>=3:raise RuntimeError('Three consecutive failed attempts; safe stop, no replacement')
    # Remove calibration-only entries from the publication, retaining all payloads
    # and complete attempt counts. No deletion or outcome-conditioned substitution.
    state=publish(True);manifest=read(root/'manifest.json')
    valid_ids={j['family']+'_'+j['tag'] for j in planned['jobs']}
    manifest['episodes']=[e for e in manifest['episodes'] if e['id'] in valid_ids]
    write(root/'manifest.json',manifest)
    counts={k:len({e['family'] for e in manifest['episodes'] if e['split']==k}) for k in ('train','validation','test')}
    if min(counts.values())<2:raise RuntimeError('Insufficient successful family coverage; not data ready')
    assert model_fingerprint(player.model)==fingerprint
    assert sha(CODE/'outputs/visuotactile/data/close_v4/manifest.json')==s['old_manifest_sha256']
    beat('numeric_acceptance');cache=build_cache(root)
    # Complete is emitted only after all actual attempts and State data acceptance.
    state.update(state='complete',successes=len(manifest['episodes']),data_ready=True,worker_pid=None,
        split_families=counts,state_data=str(cache),finished=time.time())
    write(root/'status.json',state);write(root/'COMPLETE.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','worker','status','stop','resume'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--attempt',type=Path)
    a=p.parse_args();r=a.root.resolve()
    if a.mode=='prepare':prepare(r)
    elif a.mode=='status':print(json.dumps(read(r/'status.json'),ensure_ascii=False,indent=2))
    elif a.mode=='stop':(r/'STOP_AFTER_EPISODE').touch()
    elif a.mode in ('run','resume'):
        if a.mode=='resume':
            (r/'STOP_AFTER_EPISODE').unlink(missing_ok=True);(r/'PAUSED.json').unlink(missing_ok=True)
        owner(r)
    else:
        try:
            worker(r,a.attempt)
        except BaseException as error:
            write(a.attempt/'worker_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        # All episode HDF5 writers are closed/fsynced. Existing measured Kit
        # teardown failures justify process-isolated OS cleanup, not a claim of
        # graceful SDK shutdown. Owner reaps the process and monitor.
        write(a.attempt/'exit_requested.json',dict(utc=time.time(),sdk_graceful_shutdown=False))
        sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__':main()
