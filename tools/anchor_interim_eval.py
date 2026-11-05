"""Bounded evaluation between complete training stages, without editing the trainer.

Only the verified training owner is suspended. Its current child finishes normally;
the monitor keeps running. Cohorts and post-evaluation hold are explicit. A ready
checkpoint can also be assessed independently once its training worker has exited.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from tools.anchor_continue_queue import Owner
from tools.runtime import CODE, RUN, read, write, sha, kill_owned
from tools.runtime import proc, descendants, gpu_pids


def same_process(record):
    current = proc(record['pid'])
    return current if current and current['start_ticks'] == record['start_ticks'] else None


def counts(items):
    return dict(valid=len(items),success=sum(bool(t['success']) for t in items),
        contact=sum(bool(t['contact']['object_contact']) for t in items),
        lift=sum(bool(t['native_lift']) for t in items),
        first_subgoal=sum(bool(t['native_subgoals']) for t in items),
        fall=sum(bool(t['reasons'].get('fall')) for t in items),
        dropped=sum(bool(t['reasons'].get('dropped')) for t in items),
        timeout=sum(bool(t['reasons'].get('timeout')) for t in items))


def resume(root, force=False):
    path = root/'parent_pause.json'
    if not path.exists():
        return
    record = read(path)
    if record.get('resumed'):
        return
    session = read(root/'session.json') if (root/'session.json').exists() else {}
    if session.get('hold_parent_for_decision') and not force:
        write(Path(record['parent_root'])/'evaluation_interlude.json', dict(
            state='held_for_evidence_review', evaluation_root=str(root),
            reason='User requires 150-epoch offline and closed-loop decision before more training',
            parent_owner_suspended=True, utc=time.time()))
        return
    current = same_process(record)
    if current:
        command = Path('/proc', str(record['pid']), 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'tools.anchor_continue_queue' in command and record['parent_root'] in command
        os.kill(record['pid'], signal.SIGCONT)
    record.update(resumed=True, resumed_utc=time.time(), exact_parent_alive=bool(current))
    write(path, record)
    write(Path(record['parent_root'])/'evaluation_interlude.json', dict(
        state='released', evaluation_root=str(root), resume=record))


def prepare(root, parent, epoch=100, ordinals=None, hold_parent=False, boundary_seconds=3600, checkpoint=None):
    parent = parent.resolve()
    status = read(parent/'status.json')
    owner=worker=None
    if checkpoint is None:
        assert status['state'] == 'running' and status['stage'] == f'state_train_epoch_{epoch}'
        owner, worker = proc(status['pid']), proc(status['worker_pid'])
        assert owner and worker and worker['ppid'] == owner['pid'] and owner['state'] not in ('T', 't')
    else:
        assert not hold_parent and checkpoint.is_file()
        assert not status.get('worker_pid'), 'Standalone evaluation cannot overlap a parent worker'
    root.mkdir(parents=True, exist_ok=False)
    session = read(parent/'session.json')
    ordinals = [4,5,6,7] if ordinals is None else ordinals
    assert ordinals and len(set(ordinals)) == len(ordinals)
    assert all(session['specs'][i]['cohort'] in ('train','validation') for i in ordinals)
    expected_updates = 10000 + session['targets']['state'][str(epoch)]
    now = time.time()
    session.update(run_id=root.name, parent_root=str(parent), started_epoch=now,
        deadline=min(session['deadline'], now+boundary_seconds+4*3600), training_deadline=session['deadline'],
        owned_worker_roots=[str(root)], kit_maximum_starts=len(ordinals)+1, kit_recovery_limit=1,
        evaluation_only=True, expected_updates=expected_updates, closed_loop_ordinals=ordinals,
        epoch=epoch, boundary_seconds=boundary_seconds, hold_parent_for_decision=hold_parent,
        standalone=checkpoint is not None,
        expected_stage=status['stage'], parent_owner=owner, parent_worker=worker,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=CODE, text=True).strip())
    write(root/'session.json', session)
    shutil.copytree(parent/'freeze', root/'freeze')
    frozen=read(parent/'runtime_freeze.json')
    frozen['tools/anchor_interim_eval.py']=sha(CODE/'tools/anchor_interim_eval.py')
    write(root/'runtime_freeze.json', frozen)
    write(root/'status.json', dict(state='pending',stage=f'wait_complete_{epoch}_epoch_checkpoint',pid=None))
    # Both queues use the same GPU lease even though the parent owner is suspended.
    (root/'gpu.lock').symlink_to(parent/'gpu.lock')
    if checkpoint is not None:
        import torch
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['updates']==expected_updates and saved['contract']['horizon']==24
        assert saved['contract']['action']=='absolute_joint_position_target_rad'
        os.link(checkpoint,root/'checkpoint.pt')
        write(root/'checkpoint.json',dict(path=str(root/'checkpoint.pt'),original=str(checkpoint),
            sha256=sha(checkpoint),updates=expected_updates,
            cumulative_anchor_epochs=saved['cumulative_sampled_anchors']/session['anchor_population']))


class Interim(Owner):
    def status(self, **kw):
        state=read(self.root/'status.json')
        state.update(pid=os.getpid(), heartbeat=time.time(), **kw)
        write(self.root/'status.json', state)

    def stage(self, name, module, args, kit=False):
        failures=sum(not x.get('valid') for x in self.attempts if x['kit'] and 'exit_code' in x)
        if kit and failures>=2:
            raise RuntimeError('One replacement-start budget exhausted; no new Kit start')
        return super().stage(name,module,args,kit=kit)

    def report(self):
        attempts=[read(self.root/'stages'/x['name']/'committed.json')
            if (self.root/'stages'/x['name']/'committed.json').exists() else x for x in self.attempts]
        rows=[x for x in attempts if x.get('native_outcome_valid')]
        completed=[x for x in self.attempts if 'exit_code' in x]
        trials=[x['result']['trials'][0] for x in rows]
        originals=[x for x in read(Path(self.s['baseline_root'])/'attempts.json')
            if x.get('valid') and x['model']=='StateDP' and x['ordinal'] in self.s['closed_loop_ordinals']]
        old=[x['trial'] for x in originals]
        planned=len(self.s['closed_loop_ordinals'])
        by_cohort={cohort:counts([x['result']['trials'][0] for x in rows
            if any(spec['identity']==x['result']['trials'][0]['identity'] and spec['cohort']==cohort
                for spec in self.s['specs'])]) for cohort in ('train','validation')}
        summary=dict(state=read(self.root/'status.json'),planned_episodes=planned,maximum_starts=self.s['kit_maximum_starts'],
            actual_starts=len(self.attempts),software_missing=sum(not x.get('valid') for x in completed),
            acceptance_failed=sum(x.get('valid',False) and not x.get('native_outcome_valid',False) for x in completed),
            pending_initials=planned-len(rows),old=counts(old),new=counts(trials),by_cohort=by_cohort,rows=rows,
            stable_grasp='N/A: no native discrete stable-grasp criterion',test40_used=False,
            training_checkpoint_unchanged=True,source_commit=self.s['source_commit'])
        if (self.root/'checkpoint.json').exists(): summary['checkpoint']=read(self.root/'checkpoint.json')
        write(self.root/'summary.json',summary)
        lines=[f"# StateDP {self.s.get('epoch',100)}-anchor-epoch evaluation",'',
            'Preregistered initial states; train fitting diagnostics and validation remain separate. No test40.',
            'Autonomous unchanged H24/commit12 absolute-position controller. No expert takeover, smoothing or loss change.','',
            '|Model|Success/valid|Contact|Native lift|First subgoal|','|---|---:|---:|---:|---:|']
        for name,c in [('StateDP10000 reference (all selected cohorts)',summary['old']),
            *[(f"StateDP{self.s['expected_updates']} {cohort}",values) for cohort,values in by_cohort.items()]]:
            lines.append(f"|{name}|{c['success']}/{c['valid']}|{c['contact']}|{c['lift']}|{c['first_subgoal']}|")
        lines += ['', f"Planned{planned}; starts{summary['actual_starts']}/{summary['maximum_starts']}; software missing{summary['software_missing']}; acceptance failed{summary['acceptance_failed']}; pending{summary['pending_initials']}.",
            'Stable grasp N/A. Contact is the existing force diagnostic, not a native stable-grasp success.',
            'Task outcome and SDK shutdown are separate; SDK fast exit is not graceful full teardown.','',
            '|Seed|Steps|Native outcome|Video|','|---|---:|---|---|']
        for x in rows:
            t=x['result']['trials'][0];video=Path(x['output'])/t['identity']/'rollout.mp4'
            lines.append(f"|{t['seed']}|{t['steps']}|{t['reasons']}|[local video]({video})|")
        (self.root/'EVALUATION_SUMMARY.md').write_text('\n'.join(lines)+'\n')

    def pause_and_wait(self):
        record=dict(self.s['parent_owner'],parent_root=self.s['parent_root'],resumed=False,utc=time.time())
        current=same_process(record)
        assert current and current['state'] not in ('T','t')
        command=Path('/proc',str(record['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'tools.anchor_continue_queue' in command and self.s['parent_root'] in command
        write(self.root/'parent_pause.json',record)
        os.kill(record['pid'],signal.SIGSTOP)
        parent=Path(self.s['parent_root'])
        write(parent/'evaluation_interlude.json',dict(state='waiting_current_stage',evaluation_root=str(self.root),
            parent_owner_suspended=True,trainer_and_monitor_suspended=False,utc=time.time()))
        self.status(state='waiting_training_boundary',stage=self.s['expected_stage'])
        while True:
            worker=same_process(self.s['parent_worker'])
            if not worker or worker['state']=='Z':break
            if time.time()>min(self.s['deadline'],self.s['started_epoch']+self.s.get('boundary_seconds',3600)):
                raise TimeoutError('Training boundary wait budget exhausted; checkpoint preserved')
            self.status(state='waiting_training_boundary',stage=self.s['expected_stage'])
            time.sleep(2)
        worker_pid=self.s['parent_worker']['pid']
        assert worker_pid not in gpu_pids() and not descendants(worker_pid), 'Trainer resources not released'
        stage=parent/'stages'/self.s['expected_stage']
        result_paths=sorted(stage.glob('attempt_*/output/result.json'))
        assert result_paths, 'No completed milestone result'
        result=read(result_paths[-1])
        assert result.get('passed') and result['update']==self.s['expected_updates']
        cp=Path(result['checkpoint']);assert sha(cp)==result['checkpoint_sha256']
        (self.root/'checkpoint.pt').unlink(missing_ok=True)
        os.link(cp,self.root/'checkpoint.pt')
        write(self.root/'checkpoint.json',dict(path=str(self.root/'checkpoint.pt'),original=str(cp),
            sha256=result['checkpoint_sha256'],updates=result['update'],cumulative_anchor_epochs=result['cumulative_anchor_epochs']))
        write(parent/'evaluation_interlude.json',dict(state='evaluating',evaluation_root=str(self.root),
            parent_owner_suspended=True,trainer_finished=True,utc=time.time()))

    def run(self):
        root=self.root
        with (root/'owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for path,digest in read(root/'runtime_freeze.json').items():
                assert sha(CODE/path)==digest, 'Frozen source drift: '+path
            try:
                if not self.s.get('standalone'):self.pause_and_wait()
                else:
                    checkpoint=read(root/'checkpoint.json')
                    assert sha(checkpoint['path'])==checkpoint['sha256']
                with (root/'monitor.log').open('a') as log:
                    self.monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],
                        cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                cp=read(root/'checkpoint.json')
                for ordinal in self.s['closed_loop_ordinals']:
                    if time.time()>self.s['deadline']:raise TimeoutError('Interim evaluation deadline')
                    # At most one replacement start across the four episodes.
                    failed=sum(not x.get('valid') for x in self.attempts if 'exit_code' in x)
                    if failed>=2:raise RuntimeError('Interim software recovery budget exhausted')
                    payload=dict(model='StateDP',updates=cp['updates'],mode='dp',specs=[self.s['specs'][ordinal]],
                        checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],video=True,capture_pair=False,
                        inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091,overlap_steps=0),
                        lifecycle=dict(fast_shutdown=True,disarm_stop=True))
                    self.kit(f'state_interim_{ordinal:02d}',payload)
                self.status(state='completed',stage='frozen_cohorts_completed',worker_pid=None)
            except BaseException:
                write(root/'owner_error.json',dict(error=traceback.format_exc(),utc=time.time()))
                self.status(state='blocked',worker_pid=None)
                raise
            finally:
                if self.worker:kill_owned(self.worker)
                if self.monitor:
                    (root/'STOP_MONITOR').touch()
                    try:self.monitor.wait(timeout=15)
                    except subprocess.TimeoutExpired:kill_owned(self.monitor)
                resume(root)
                state=read(root/'status.json');state.update(pid=None,worker_pid=None,finished=time.time())
                write(root/'status.json',state);self.report()
                write(root/'owner_exit.json',state)


def wilson(successes, n):
    if not n:return None
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [max(0.,center-half),min(1.,center+half)]


def random_specs(data,seed):
    import random
    rng=random.Random(seed)
    pool=sorted(data['splits']['train']+data['splits']['validation'])
    assert len(pool)==360 and not set(pool)&set(data['splits']['test'])
    historical=rng.sample(pool,120);seeds=rng.sample(range(11000000,21000000),240)
    return [dict(identity=identity,cohort='train' if identity in data['splits']['train'] else 'validation',
        seed=seeds[i],evaluation_group='historical',selection_index=i,reserve=i>=100)
        for i,identity in enumerate(historical)]+[
        dict(identity=f'fresh_{v}',cohort='fresh',seed=v,evaluation_group='fresh',selection_index=i,reserve=i>=100)
        for i,v in enumerate(seeds[120:])]


def initial_signature(values):
    import hashlib
    import numpy as np
    base=np.asarray(values['base_pose_w']).reshape(-1)
    pieces=[np.asarray(values['joint_position_rad']).reshape(-1)]
    for key in ('object_pose_w','hole_pose_w'):
        pose=np.asarray(values[key]).reshape(-1).copy();pose[:3]-=base[:3];pieces.append(pose)
    return hashlib.sha256(np.round(np.concatenate(pieces),6).astype('<f8').tobytes()).hexdigest()


def prepare_random(root, assessment, seed=20261105):
    """Freeze outcome-independent cohorts. Reserve entries replace only illegal starts."""
    import h5py
    import json
    assessment=assessment.resolve();s=read(assessment/'session.json');cp=read(assessment/'checkpoint.json')
    assert cp['updates']==20918 and sha(cp['path'])==cp['sha256']
    data=read(Path(s['source_root'])/'data.json');specs=random_specs(data,seed)
    root.mkdir(parents=True,exist_ok=False)
    now=time.time()
    s.update(run_id=root.name,evaluation_kind='random_state150',started_epoch=now,deadline=now+24*3600,
        training_deadline=now+24*3600,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        assessment_root=str(assessment),specs=specs,sampling_seed=seed,cohort_targets=dict(historical=100,fresh=100),
        kit_maximum_starts=243,kit_auto_retry=False,kit_gate=dict(shutdown='isolated_process_exit',exit_on_error=True),
        syscall_diagnostic_stages=['smoke_state_0','smoke_state_1','smoke_expert'],
        owned_worker_roots=[str(root)],num_envs=1,
        isolation='One independent Kit process and new HistoryController per episode; no parallel/reset equivalence claim',
        smoke_specs=[read(assessment/'session.json')['specs'][4],read(assessment/'session.json')['specs'][0]],
        sampling='Uniform without replacement from combined train320+validation40; 100 primary plus20 preregistered illegal-initialization-only reserves per group')
    write(root/'session.json',s);write(root/'sampling_manifest.json',dict(seed=seed,specs=specs,
        outcome_selected=False,test40_used=False,targets=s['cohort_targets']))
    shutil.copytree(assessment/'freeze',root/'freeze')
    # A new private runtime, not a symlink to the failed worker's caches. Only
    # the frozen persistent settings above are copied; old evidence is retained.
    (root/'private_runtime').mkdir()
    signatures={};distribution=None
    for identity in data['splits']['train']+data['splits']['validation']:
        entry=data['entries'][identity]
        with h5py.File(entry['path']) as f:
            g=f['episodes'][entry['episode']]['transition/diagnostics_before']
            signatures[identity]=initial_signature({k:g[k][0] for k in ('joint_position_rad','base_pose_w','object_pose_w','hole_pose_w')})
            cfg=json.loads(f.attrs['metadata_json'])['config']
            fields={k:cfg[k] for k in ('reset','domain_randomization','precise_assembly','termination','reward','assets','decimation')}
            fields['physics_dt']=cfg['sim']['dt']
            if distribution is None:distribution=fields
            assert distribution==fields,'Original initialization/physics configuration heterogeneity requires explicit stratification'
    write(root/'original_initial_signatures.json',dict(signatures=signatures,precision=1e-6,
        scope='train320+validation40 only; robot-relative object/hole pose plus canonical joint positions'))
    write(root/'initialization_distribution.json',distribution)
    (root/'gpu.lock').symlink_to(assessment/'gpu.lock')
    os.link(cp['path'],root/'checkpoint.pt')
    write(root/'checkpoint.json',dict(cp,path=str(root/'checkpoint.pt')))
    paths=[p for base in ('visuotactile','tools','isaacsimenvs','evaluation') for p in (CODE/base).rglob('*.py')]
    paths.append(CODE/'visuotactile/run_cpu.sh')
    write(root/'runtime_freeze.json',{str(p.relative_to(CODE)):sha(p) for p in paths})
    write(root/'status.json',dict(state='pending',stage='supervised_smokes',pid=None))


class RandomEvaluation(Owner):
    def kit(self,name,job):
        record=super().kit(name,job)
        self.attempts=[record if x['name']==record['name'] and x['number']==record['number'] else x for x in self.attempts]
        write(self.root/'attempts.json',self.attempts)
        return record

    def status(self,**kw):
        value=read(self.root/'status.json');value.update(pid=os.getpid(),heartbeat=time.time());value.update(kw)
        write(self.root/'status.json',value)

    def validate_kit(self,record):
        import h5py
        import numpy as np
        import torch
        from visuotactile.success_audit import analyze
        from visuotactile.video import verify_video
        result=record['result'];assert len(result['trials'])==1
        trial=result['trials'][0];directory=Path(record['output'])/trial['identity']
        metadata=read(directory/'metadata.json') if (directory/'metadata.json').exists() else None
        if not trial['legal']:
            assert trial.get('not_started')
            record.update(initialization_rejected=True,native_outcome_valid=False)
            return
        assert not trial['action_exhausted'] and trial['actuator_target_max_error']==0
        job=read(self.root/'jobs'/(record['name']+'.json'))
        cfg=metadata['config'];expected=read(self.root/'initialization_distribution.json')
        actual={k:cfg[k] for k in expected if k!='physics_dt'};actual['physics_dt']=cfg['sim']['dt']
        assert actual==expected,'Environment initialization/dynamics contract changed'
        signature=initial_signature(read(directory/'initial_observable.json'))
        if trial['cohort']=='fresh':
            assert signature not in read(self.root/'original_initial_signatures.json')['signatures'].values(), 'Fresh initial state matches training/validation pool'
            assert signature not in [x.get('observable_initial_signature') for x in self.attempts if x is not record], 'Duplicate fresh initialization'
        record['observable_initial_signature']=signature
        if job['mode']=='dp':
            assert trial['submitted_target_max_error']==0
            audit=read(Path(record['output'])/'controller_audit.json')
            assert audit['completed'] and audit['history_initially_empty'] and audit['model_output_only']
        with np.load(directory/'trace.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files)
        with h5py.File(directory/'numeric.h5') as f:
            goal_state=torch.load(directory/'initial_export.pt',map_location='cpu',weights_only=False)['env_tensors']
            audit,_=analyze(f['episode'],metadata,directory/'success_audit.npz',goal_state=goal_state)
        write(directory/'success_audit.json',audit)
        assert audit['passed'], 'Native success/termination audit disagreement'
        assert trial['first_success_tick']==audit['first_complete_tick']
        assert trial['termination_tick']==audit['termination_tick'] and trial['post_success_actions']==0
        record['success_audit']=audit
        record['video']=verify_video(directory/'rollout.mp4',trial['steps'])
        record['trace_sha256']=sha(directory/'trace.npz')
        record['initial_state_sha256']=sha(directory/'initial_observable.json')
        record['sdk_exit']=dict(code=record['exit_code'],graceful=(Path(record['output'])/'closed.json').exists(),
            isolated_exit=(Path(record['output'])/'process_exit_requested.json').exists())
        if record['name']=='smoke_state_0':
            previous=next(r for r in read(Path(self.s['assessment_root'])/'summary.json')['rows']
                if r['result']['trials'][0]['identity']==trial['identity'])
            from tools.trajectory_metrics import compare_initial
            old=Path(previous['output'])/trial['identity']
            initial=compare_initial(directory,old)
            assert initial['observable_equal'] and initial['environment_rng_equal'] and not initial['exported_differences']
            with np.load(directory/'plans.npz') as a,np.load(old/'plans.npz') as b:
                error=float(np.abs(a['plans'][0]-b['plans'][0]).max())
            assert error<2e-5, ('First plan after source consolidation',error)
            record['postcleanup_first_plan_max_rad']=error
        record['native_outcome_valid']=True

    def payload(self,spec,expert=False):
        cp=read(self.root/'checkpoint.json')
        return dict(model='RL expert reference' if expert else 'StateDP150',updates=0 if expert else cp['updates'],
            mode='expert' if expert else 'dp',specs=[spec],checkpoint=self.s['expert_checkpoint'] if expert else cp['path'],
            checkpoint_sha256=sha(self.s['expert_checkpoint']) if expert else cp['sha256'],video=True,capture_pair=False,
            **({} if expert else dict(inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091,overlap_steps=0))),
            lifecycle=dict(fast_shutdown=True,disarm_stop=True))

    def report(self):
        completed=[x for x in self.attempts if 'exit_code' in x]
        formal=[x for x in completed if x['name'].startswith('random_')]
        valid=[x for x in formal if x.get('native_outcome_valid')]
        rows=[]
        for group in ('historical','train','validation','fresh'):
            selected=[x for x in valid if (x['result']['trials'][0].get('evaluation_group')==group if group in ('historical','fresh')
                else x['result']['trials'][0]['cohort']==group)]
            trials=[x['result']['trials'][0] for x in selected];n=len(trials);k=sum(t['success'] for t in trials)
            c=counts(trials);c.update(group=group,success_rate=k/n if n else None,ci95=wilson(k,n),
                approach_diagnostic=sum(t.get('approach_proxy',False) for t in trials),
                stable_grasp_diagnostic=sum(t.get('contact',{}).get('stable_grasp_diagnostic',False) for t in trials),
                native_alignment=sum(t['native_alignment'] for t in trials),
                ever_complete=sum(t['ever_complete_success'] for t in trials),
                terminal_geometry_diagnostic=sum(x['success_audit'].get('terminal_geometry',{}).get('ever_satisfied',False) for x in selected),
                other_failure=sum(not t['success'] and not any(t['reasons'].get(k) for k in ('fall','dropped','timeout')) for t in trials))
            rows.append(c)
        summary=dict(status=read(self.root/'status.json'),planned_valid=200,primary_planned_starts=200,
            maximum_starts=self.s['kit_maximum_starts'],actual_starts=len(self.attempts),formal_attempts=sum(x['name'].startswith('random_') for x in self.attempts),
            valid=len(valid),pending_valid=200-len(valid),software_missing=sum(not x.get('valid') for x in formal),
            acceptance_failed=sum(x.get('valid',False) and not x.get('native_outcome_valid',False) and not x.get('initialization_rejected') for x in formal),
            initialization_rejections=sum(x.get('initialization_rejected',False) for x in formal),
            shutdown_faults=sum(x.get('native_outcome_valid',False) and x['exit_code']!=0 for x in formal),
            smoke_attempts=sum(not x['name'].startswith('random_') for x in self.attempts),smoke_valid=sum(x.get('native_outcome_valid',False) for x in completed if not x['name'].startswith('random_')),
            smoke_software_missing=sum(not x.get('valid') for x in completed if not x['name'].startswith('random_')),
            smoke_acceptance_failed=sum(bool(x.get('acceptance_error')) for x in completed if not x['name'].startswith('random_')),
            groups=rows,checkpoint=read(self.root/'checkpoint.json'),sampling_seed=self.s['sampling_seed'],
            source_commit=self.s['source_commit'],num_envs=1,test40_used=False,
            limitation='Historical observable reconstruction, not exact hidden PhysX state. Wilson intervals describe conditional binomial uncertainty; historical pool sampled without replacement.',
            trials=[dict(name=x['name'],output=x['output'],trial=x['result']['trials'][0],success_audit=x['success_audit'],
                video=x.get('retained_video'),sdk_exit=x['sdk_exit']) for x in valid])
        write(self.root/'summary.json',summary)
        lines=['# StateDP150 random closed-loop evaluation','',
            '|Cohort|Success / valid|Rate|95% Wilson CI|Contact|Grasp diagnostic|Lift|First subgoal|',
            '|---|---:|---:|---|---:|---:|---:|---:|']
        for r in rows:
            ci='N/A' if r['ci95'] is None else '–'.join(f'{v:.1%}' for v in r['ci95'])
            rate='N/A' if r['success_rate'] is None else f"{r['success_rate']:.1%}"
            lines.append(f"|{r['group']}|{r['success']}/{r['valid']}|{rate}|{ci}|{r['contact']}|{r['stable_grasp_diagnostic']}|{r['lift']}|{r['first_subgoal']}|")
        lines += ['',f"Planned valid 200; formal attempts {summary['formal_attempts']}; valid {len(valid)}; pending {200-len(valid)}; software missing {summary['software_missing']}; initial rejections {summary['initialization_rejections']}; shutdown faults {summary['shutdown_faults']}.",
            f"Acceptance failures {summary['acceptance_failed']}. Separate smoke attempts {summary['smoke_attempts']}, valid {summary['smoke_valid']}, software missing {summary['smoke_software_missing']}.",
            'Contact, approach and stable grasp are diagnostics, not native success. A legal task failure remains in the denominator.',
            'Final-pose plus fingers-clear diagnostic (ordered native success unchanged): '+', '.join(f"{r['group']} {r['terminal_geometry_diagnostic']}/{r['valid']}" for r in rows)+'.',
            summary['limitation'],f"State: {summary['status'].get('state')}; stage: {summary['status'].get('stage')}."]
        (self.root/'EVALUATION_SUMMARY.md').write_text('\n'.join(lines)+'\n')

        public=CODE/'research_docs/evidence/state_dp'
        light={k:v for k,v in summary.items() if k!='trials'}
        light['trials']=[dict(name=r['name'],output=r['output'],trial=r['trial'],video=r['video'],sdk_exit=r['sdk_exit'],
            success_audit={k:v for k,v in r['success_audit'].items() if k!='motion'}) for r in summary['trials']]
        write(public/'random_evaluation.json',light)
        block=['<!-- RANDOM_EVALUATION_START -->','## 本轮随机评估（实时保存）','',*lines[2:],
            '',f"固定抽样seed `{self.s['sampling_seed']}`，历史池360中无放回随机抽取；前100为主清单，另20仅替换动作前非法初态。全新组同样预注册100+20。不得按成功/失败替换。",
            '每次独立单环境Kit；未采用并行或进程内reset复用，因此不声称已验证并行等价。所有模型动作来自150 epoch checkpoint。',
            f"运行目录：`{self.root}`。冻结源码 `{self.s['source_commit']}`；checkpoint SHA `{summary['checkpoint']['sha256']}`。",
            '[机器统计与每次结果](evidence/state_dp/random_evaluation.json)。软件故障、非法初始化和任务失败分别记录；未完成时不得据此声称100+100成功率。']
        if valid:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig,ax=plt.subplots(figsize=(9,4))
            stages=['approach_diagnostic','contact','stable_grasp_diagnostic','lift','first_subgoal','success']
            for index,g in enumerate(('train','validation','fresh')):
                row=next(r for r in rows if r['group']==g)
                ax.bar(np.arange(len(stages))+(index-1)*.25,[row[k]/row['valid']*100 if row['valid'] else 0 for k in stages],
                    width=.25,label=f"{g} n={row['valid']}"+(' (pending)' if not row['valid'] else ''))
            ax.set_xticks(range(6),['Approach*','Contact*','Stable grasp*','Native lift','Subgoal 1','Full success'])
            ax.set_ylim(0,105);ax.set_ylabel('Percent of valid rollouts');ax.legend();ax.set_title('Stages are separate indicators, not a required monotonic funnel')
            fig.tight_layout();fig.savefig(public/'random_stages.png',dpi=150);plt.close(fig)
            block += ['','![随机闭环阶段对比](evidence/state_dp/random_stages.png)',
                '带*为既有诊断定义。Native lift不是完整任务的必要条件；原400成功中只有153条触发该抬升阈值。',
                '', '|类别|seed|结果|视频|','|---|---:|---|---|']
            for r in summary['trials']:
                if r['video']:
                    t=r['trial'];label='完整成功' if t['success'] else '+'.join(k for k,v in t['reasons'].items() if v)
                    block.append(f"|{t['cohort']}|{t['seed']}|{label}|[本地视频]({r['video']})|")
        block += ['<!-- RANDOM_EVALUATION_END -->','']
        report=CODE/'research_docs/STATE_DP_RESULTS.md';text=report.read_text();start='<!-- RANDOM_EVALUATION_START -->';end='<!-- RANDOM_EVALUATION_END -->'
        if start in text:
            before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+'\n'.join(block)+after.lstrip('\n')
        else:text=text.split('\n',1)[0]+'\n\n'+'\n'.join(block)+text.split('\n',1)[1]
        report.write_text(text)

    def retain_video(self,record):
        trial=record['result']['trials'][0];video=Path(record['output'])/trial['identity']/'rollout.mp4'
        category=[trial['cohort'],'success' if trial['success'] else '+'.join(k for k,v in trial['reasons'].items() if v)]
        prior=[x for x in self.attempts if x.get('video_category')==category and x.get('retained_video')]
        # Keep two per split/outcome and every smoke. Only this new run's validated
        # temporary videos are discarded; numeric evidence is always retained.
        keep=record['name'].startswith('smoke_') or len(prior)<2
        record['video_category']=category
        record['retained_video']=str(video) if keep else None
        if not keep:
            record['discarded_video']=dict(path=str(video),sha256=sha(video),reason='additional same-category viewing encoding; full numeric rollout retained')
            video.unlink()
        write(self.root/'stages'/record['name']/'committed.json',record)
        write(self.root/'attempts.json',self.attempts)

    def run(self):
        root=self.root
        with (root/'owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for p,digest in read(root/'runtime_freeze.json').items():assert sha(CODE/p)==digest,'Source drift: '+p
            assert sha(root/'checkpoint.pt')==read(root/'checkpoint.json')['sha256']
            (root/'STOP_MONITOR').unlink(missing_ok=True)
            with (root/'monitor.log').open('a') as log:
                self.monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                smoke_jobs=[('smoke_state_0',self.s['smoke_specs'][0],False),
                    ('smoke_state_1',self.s['smoke_specs'][1],False),('smoke_expert',self.s['smoke_specs'][1],True)]
                if self.s.get('smoke_fresh'):
                    smoke_jobs.append(('smoke_fresh',self.s['smoke_fresh'],False))
                for name,spec,expert in smoke_jobs:
                    record=self.kit(name,self.payload(spec,expert))
                    assert record.get('native_outcome_valid') and record['resources_released']
                    if expert:assert record['result']['trials'][0]['success'],'Expert success acceptance failed'
                    if not record.get('retained_video'):self.retain_video(record)
                write(root/'smoke_gate.json',dict(passed=True,utc=time.time(),full_episodes=len(smoke_jobs),
                    purpose='Runtime correctness, complete native expert success, frozen first-plan parity and OS resource reclamation',
                    sdk_graceful_shutdown_claimed=False,parallel_execution_used=False))
                self.status(state='supervised_ready',stage='awaiting_recorded_smoke_review',worker_pid=None);self.report()
                while not (root/'SUPERVISION_ACCEPTED.json').exists():
                    if time.time()>self.s['deadline'] or (root/'STOP').exists():raise RuntimeError('Deadline/STOP before formal release')
                    time.sleep(2)
                assert read(root/'SUPERVISION_ACCEPTED.json')['passed']
                for group in ('historical','fresh'):
                    for spec in [x for x in self.s['specs'] if x['evaluation_group']==group]:
                        committed=[x for x in self.attempts if x['name'].startswith('random_') and x.get('native_outcome_valid')
                            and x['result']['trials'][0]['evaluation_group']==group]
                        if len(committed)>=100:break
                        name=f"random_{group}_{spec['selection_index']:03d}"
                        if (root/'stages'/name/'committed.json').exists():continue
                        record=self.kit(name,self.payload(spec))
                        if record.get('native_outcome_valid'):self.retain_video(record)
                        self.report()
                    assert sum(x['name'].startswith('random_') and x.get('native_outcome_valid',False)
                        and x['result']['trials'][0]['evaluation_group']==group for x in self.attempts)==100, 'Preregistered legal-start reserve exhausted'
                self.status(state='completed',stage='historical100_fresh100_complete',worker_pid=None)
            except BaseException:
                write(root/'owner_error.json',dict(error=traceback.format_exc(),utc=time.time()))
                self.status(state='blocked',worker_pid=None)
                raise
            finally:
                if self.worker:kill_owned(self.worker)
                (root/'STOP_MONITOR').touch()
                try:self.monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(self.monitor)
                self.status(pid=None,worker_pid=None,finished=time.time());self.report()
                write(root/'owner_exit.json',read(root/'status.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--epoch',type=int,default=100)
    p.add_argument('--ordinals',type=int,nargs='+');p.add_argument('--hold-parent',action='store_true')
    p.add_argument('--boundary-seconds',type=int,default=3600)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--sampling-seed',type=int,default=20261105)
    p.add_argument('mode',choices=['prepare','random-prepare','run','resume','release','report'])
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='prepare':prepare(root,a.parent,a.epoch,a.ordinals,a.hold_parent,a.boundary_seconds,a.checkpoint)
    elif a.mode=='random-prepare':prepare_random(root,a.parent,a.sampling_seed)
    elif a.mode in ('resume','release'):resume(root,force=a.mode=='release')
    else:
        owner=RandomEvaluation(root) if read(root/'session.json').get('evaluation_kind')=='random_state150' else Interim(root)
        if a.mode=='report':owner.report()
        else:owner.run()
