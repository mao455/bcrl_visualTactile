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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--epoch',type=int,default=100)
    p.add_argument('--ordinals',type=int,nargs='+');p.add_argument('--hold-parent',action='store_true')
    p.add_argument('--boundary-seconds',type=int,default=3600)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('mode',choices=['prepare','run','resume','release','report'])
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='prepare':prepare(root,a.parent,a.epoch,a.ordinals,a.hold_parent,a.boundary_seconds,a.checkpoint)
    elif a.mode in ('resume','release'):resume(root,force=a.mode=='release')
    elif a.mode=='report':Interim(root).report()
    else:Interim(root).run()
