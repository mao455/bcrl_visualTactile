"""Persistent bounded state-first protocol; old run is a read-only predecessor."""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .compact import sha256
from .native_v2_gates import Gate, replay_gate, rollout_gate, visual_gate, plateau
from .training_resources import atomic_json


def validate_config(config):
    fixed={'horizon':64,'commit':12,'noise_steps':100,'ddim_steps':16,'prediction_type':'sample','loss':'B','gradient_accumulation':1}
    for key,value in fixed.items():
        if config['state_training'].get(key)!=value:raise ValueError('Unsupported inference/training contract: '+key)
    checks={'models':['S-NativeDP'],'train_initials':8,'validation_initials':8,'max_replay_starts':8,
            'repeat_initials_per_split':2,'replay_required_successes':8,'state_gate_min_successes':2,
            'diffusion_seeds':[7091,17091,27091],'parallel_envs':1}
    for key,value in checks.items():
        if config.get(key)!=value:raise ValueError('Registered protocol implementation requires '+key+'='+str(value))
    if config['diagnostic_bc']!={'hidden':[256,256],'max_updates':1000,'train_episodes':4}:
        raise ValueError('BC diagnostic implementation/configuration mismatch')
    if config['state_training']['milestones']!=[1000,3000,6000,10000,19000]:
        raise ValueError('Milestone protocol change needs an explicit implementation review')
    for key,maximum in [('diagnostic_window_hours',4),('training_window_hours',16),('waiting_predecessor_hours',10)]:
        if not 0<config[key]<=maximum:raise ValueError('Budget outside reviewed bound: '+key)


def gpu_deadlines(session,started):
    absolute=min(session['session_deadline'],started+(session['config']['diagnostic_window_hours']+session['config']['training_window_hours'])*3600)
    diagnostic=min(absolute,started+session['config']['diagnostic_window_hours']*3600)
    return diagnostic,absolute


def source_hashes():
    paths=list(Path('visuotactile').rglob('*.py'))+list(Path('isaacsimenvs/tasks/play').rglob('*.py'))
    paths += [Path('configs/native_dp_v2.json')]
    return {str(p):sha256(p) for p in sorted(paths)}


def predecessor_alive(source):
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().decode().split('\0')
            if '--root' not in args or Path(args[args.index('--root')+1]).resolve()!=source.resolve():continue
            if any(module in args for module in ('visuotactile.native_pipeline','visuotactile.native_resources',
                    'visuotactile.native_training','visuotactile.native_evaluation')):found.append(int(p.parent.name))
        except (OSError,ValueError,IndexError):pass
    return found


def checkpoint_inventory(source):
    manifest=json.loads((source/'manifest.json').read_text());row=manifest['stages']['S-NativeDP_formal']
    if row['state']!='complete':raise RuntimeError('Old State formal training is not complete')
    import torch
    result={}
    for label in ('best','last'):
        p=Path(row['output'])/(label+'.pt');saved=torch.load(p,map_location='cpu',weights_only=False)
        weight_hash=hashlib.sha256()
        for key,value in sorted(saved['ema'].items()):
            weight_hash.update(key.encode());weight_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[label]=dict(ema_sha256=weight_hash.hexdigest(),path=str(p.resolve()),sha256=sha256(p),updates=saved['updates'],epoch=saved['epoch'],
            best=saved['best'],ema_step=saved['ema_step'],configuration=saved['configuration'],
            state_dict_elements_including_buffers=sum(v.numel() for v in saved['model'].values()),
            parameter_count=json.loads((Path(manifest['stages']['S-NativeDP_network']['output'])/'result.json').read_text())['parameters'])
        del saved
    result['milestones_missing']=True
    return result


def initialize(root,config_path):
    if root.exists():raise FileExistsError('New root required')
    config=json.loads(config_path.read_text());validate_config(config);source=Path(config['source_root']).resolve()
    audit_path=Path(config['data_audit']);audit=json.loads(audit_path.read_text())
    if not audit['infrastructure_passed']:raise RuntimeError('CPU data audit failed')
    if sha256(source/'data.json')!=audit['source_data_sha256'] or sha256(source/'statistics.pt')!=audit['statistics_sha256']:
        raise ValueError('Audited data/statistics identity changed')
    control_path=audit_path.parent/'control_audit.json'
    control=json.loads(control_path.read_text())
    if not control['passed'] or control['episodes']!=400:raise RuntimeError('All400 native control audit required')
    inventory=checkpoint_inventory(source);missing=inventory.pop('milestones_missing')
    now=time.time();root.mkdir(parents=True);(root/'gates').mkdir()
    shutil.copy2(audit_path,root/'data_audit.json');shutil.copy2(control_path,root/'control_audit.json')
    session=dict(protocol_version=2,created=now,session_deadline=now+24*3600,wait_deadline=now+config['waiting_predecessor_hours']*3600,
        started_epoch=None,deadline=None,training_deadline=None,source_root=str(source),config=config,
        cohorts=audit['cohorts'],checkpoints=inventory,old_milestone_checkpoints_missing=missing,
        implementation_sha256=source_hashes(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='8 independent historical replays, old State train8/val8 plus seed and last sensitivity; new updates only after G0',
        maximum_initial_diagnostic_starts=32,maximum_post_gate_starts=100,automatic_trajectory_retries=0)
    atomic_json(root/'session.json',session);atomic_json(root/'status.json',dict(state='registered',pid=None))
    atomic_json(root/'gpu_schedule.json',dict(target_percent=40,admission_percent=45,min_gpu_free_mib=6144,min_ram_gib=14))
    trials=[]
    for i,identity in enumerate(session['cohorts']['replay']):
        trials.append(dict(name=f'replay_{i:02d}',identity=identity,kind='replay',seed=20261102+i,video=i==0))
    for split in ('train','validation'):
        for i,identity in enumerate(session['cohorts'][split]):
            trials.append(dict(name=f'old_best_{split}_{i:02d}',identity=identity,kind='student',cohort=split,
                seed=7091,checkpoint=inventory['best']['path'],checkpoint_sha256=inventory['best']['sha256'],video=i==0))
        for i,identity in enumerate(session['cohorts'][split][:2]):
            for label,seed in [('best',17091),('last',7091)]:
                if label=='last' and inventory['best']['ema_sha256']==inventory['last']['ema_sha256']:continue
                trials.append(dict(name=f'old_{label}_{split}_sensitivity_{i:02d}',identity=identity,kind='student',
                    cohort=split+'_sensitivity_'+label,seed=seed,checkpoint=inventory[label]['path'],checkpoint_sha256=inventory[label]['sha256']))
    atomic_json(root/'diagnostic_job.json',dict(trials=trials))
    atomic_json(root/'gates/G0.json',Gate('G0',True,None,None,False,'Physical replay pending').to_dict())
    for name in ('G1','G2','G3','G4','G5'):
        atomic_json(root/'gates'/(name+'.json'),Gate(name,False,None,None,False,'Required experiment evidence pending').to_dict())
    atomic_json(root/'cleanup_inventory.json',dict(deleted=[],released_bytes=0,policy='Retain raw400, weights, attempts and logs; no speculative deletion'))


class Runner:
    def __init__(self,root):
        self.root=root;self.session=json.loads((root/'session.json').read_text());self.child=None;self.monitor=None;self.stop=False
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True));signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))

    def status(self,state,**extra):
        atomic_json(self.root/'status.json',dict(state=state,pid=os.getpid(),utc=time.time(),**extra))

    def check(self):
        if self.stop or (self.root/'STOP').exists():raise InterruptedError('Explicit owned stop')
        deadline=self.session['deadline'] or self.session['wait_deadline']
        if time.time()>=deadline:raise TimeoutError('Immutable V2 deadline expired')

    def stage(self,name,module,args,completion='result.json'):
        marker=self.root/(name+'_'+completion.replace('.json','')+'_committed.json');output=self.root/name
        if marker.exists():
            record=json.loads(marker.read_text())
            if sha256(output/completion)!=record['sha256']:raise ValueError('Stage artifact changed')
            return output
        self.check()
        if source_hashes()!=self.session['implementation_sha256']:raise ValueError('Implementation changed; no source migration')
        if output.exists() and module!='visuotactile.native_v2_training':
            raise RuntimeError('Incomplete attempted cohort requires review; no automatic trajectory retry')
        command=['bash','visuotactile/run_cpu.sh','-m',module,'--root',str(self.root),'--output',str(output)]+args
        if module=='visuotactile.native_v2_training' and (output/'last.pt').exists():command+=['--resume']
        atomic_json(self.root/(name+'_launch.json'),dict(command=command,utc=time.time(),implementation_sha256=self.session['implementation_sha256']))
        self.status('running',stage=name)
        with (self.root/(name+'.log')).open('a') as log:
            self.child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.status('running',stage=name,child_pid=self.child.pid)
            while self.child.poll() is None:
                self.check()
                if self.monitor.poll() is not None:raise RuntimeError('Resource monitor exited')
                time.sleep(2)
            code=self.child.returncode;self.child=None
        if code!=0 or not (output/completion).exists():raise RuntimeError(f'{name}: exit {code} or completion missing')
        result=json.loads((output/completion).read_text())
        if not result.get('software_complete'):raise RuntimeError('Incomplete software stage '+name)
        atomic_json(marker,dict(sha256=sha256(output/completion),utc=time.time()))
        return output

    def write_gate(self,gate):atomic_json(self.root/'gates'/(gate.name+'.json'),gate.to_dict())

    def rollout_job(self,checkpoint,label):
        trials=[];digest=sha256(checkpoint)
        for split in ('train','validation'):
            for i,identity in enumerate(self.session['cohorts'][split]):
                trials.append(dict(name=f'{label}_{split}_{i:02d}',identity=identity,kind='student',cohort=split,
                    seed=7091,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest,video=i==0))
        path=self.root/(label+'_job.json');atomic_json(path,dict(trials=trials));return path

    def run(self):
        try:
            # Persistent script owns the wait; no model polling or source edits in old worktree.
            while predecessor_alive(Path(self.session['source_root'])):
                self.status('waiting_predecessor',old_pids=predecessor_alive(Path(self.session['source_root'])))
                self.check();time.sleep(10)
            if self.session['started_epoch'] is None:
                now=time.time();diagnostic,absolute=gpu_deadlines(self.session,now)
                self.session.update(started_epoch=now,deadline=diagnostic,absolute_deadline=absolute)
                self.session['training_deadline']=self.session['deadline']
                atomic_json(self.root/'session.json',self.session)
            self.check()
            # Read test usage only as provenance; never feed its scores into selection.
            source=Path(self.session['source_root'])
            ledger={p.name:json.loads(p.read_text()) for p in source.glob('*_test_started.json')}
            atomic_json(self.root/'test_usage_ledger.json',dict(previous_run=str(source),previous_test_markers=ledger,
                virgin_test=False if ledger else None,new_test_calls=0,policy='Do not describe reused test40 as previously unseen'))
            (self.root/'STOP_MONITOR').unlink(missing_ok=True)
            self.monitor_log=(self.root/'monitor.log').open('a')
            self.monitor=subprocess.Popen(['bash','visuotactile/run_cpu.sh','-m','visuotactile.native_resources','--root',str(self.root)],
                stdin=subprocess.DEVNULL,stdout=self.monitor_log,stderr=subprocess.STDOUT,start_new_session=True)
            path=self.stage('diagnostics','visuotactile.native_v2_evaluation',['--job',str(self.root/'diagnostic_job.json')])
            trials=json.loads((path/'result.json').read_text())['trials']
            audit=json.loads((self.root/'data_audit.json').read_text())
            g0=replay_gate(audit,[r for r in trials if r['kind']=='replay'],self.session['cohorts']['replay']);self.write_gate(g0)
            g2=rollout_gate('G2',[r for r in trials if r.get('cohort')=='train'],self.session['cohorts']['train']);self.write_gate(g2)
            g3=rollout_gate('G3',[r for r in trials if r.get('cohort')=='validation'],self.session['cohorts']['validation']);self.write_gate(g3)
            self.stage('frozen_offline','visuotactile.native_v2_training',['--kind','frozen-offline'])
            if not g0.acceptance_passed:raise RuntimeError('G0 blocked: '+g0.blocked_reason)
            self.session['deadline']=self.session['absolute_deadline']
            self.session['training_deadline']=self.session['deadline']
            atomic_json(self.root/'session.json',self.session)
            # No new State DP before deterministic diagnostic. Fresh optimizer, no old-run resume.
            self.stage('bc','visuotactile.native_v2_training',['--kind','bc','--until','1000'],completion='completed_1000.json')
            job=self.rollout_job(self.root/'bc/best_validation_offline.pt','bc')
            bc_path=self.stage('bc_rollout','visuotactile.native_v2_evaluation',['--job',str(job)])
            bc_rows=json.loads((bc_path/'result.json').read_text())['trials']
            if not any(r['success'] for r in bc_rows if r['cohort']=='train'):
                raise RuntimeError('StateBC and existing DP diagnostic foundation unresolved; no automatic full DP retraining')
            # Old fit1/fit4 remain distinct offline evidence; current vectorized parity separately tested.
            fit_manifest=json.loads((source/'manifest.json').read_text())['stages']
            fit_results=[json.loads((Path(fit_manifest['S-NativeDP_'+s]['output'])/'result.json').read_text()) for s in ('fit1','fit4')]
            g1=Gate('G1',True,all(r['passed'] for r in fit_results),None,all(r['passed'] for r in fit_results),None)
            self.write_gate(g1)
            history=[]
            if not (g2.acceptance_passed and g3.acceptance_passed):
                for milestone in self.session['config']['state_training']['milestones']:
                    self.stage('state','visuotactile.native_v2_training',['--kind','dp','--until',str(milestone)],completion=f'completed_{milestone}.json')
                    ckpt=self.root/'state'/f'milestone_{milestone}.pt';job=self.rollout_job(ckpt,f'state_{milestone}')
                    directory=self.stage(f'rollout_{milestone}','visuotactile.native_v2_evaluation',['--job',str(job)])
                    rows=json.loads((directory/'result.json').read_text())['trials']
                    g2=rollout_gate('G2',[r for r in rows if r['cohort']=='train'],self.session['cohorts']['train']);self.write_gate(g2)
                    g3=rollout_gate('G3',[r for r in rows if r['cohort']=='validation'],self.session['cohorts']['validation']);self.write_gate(g3)
                    validation=[r for r in rows if r['cohort']=='validation']
                    successes=sum(r['success'] for r in validation);stage_score=sum(r['native_subgoals']+r['native_lift']+r['native_alignment'] for r in validation)
                    history.append(dict(updates=milestone,successes=successes,stage_score=stage_score))
                    atomic_json(self.root/'milestones.json',history)
                    selected=self.root/'best_validation_rollout.json';key=(successes,stage_score,-milestone)
                    previous=json.loads(selected.read_text()) if selected.exists() else None
                    if previous is None or key>tuple(previous['key']):atomic_json(selected,dict(path=str(ckpt),sha256=sha256(ckpt),key=key))
                    if g2.acceptance_passed and g3.acceptance_passed:break
                    if plateau(history):raise RuntimeError('Three State milestones without full success or stage progress; stop uninformative expansion')
            repeats=[r for r in trials if r.get('cohort')=='validation_sensitivity_best'] if not history else []
            if history and g2.acceptance_passed and g3.acceptance_passed:
                repeat_specs=[dict(name=f'repeat_{i}',kind='student',identity=identity,cohort='validation_repeat',seed=17091,
                    checkpoint=str(ckpt.resolve()),checkpoint_sha256=sha256(ckpt))
                    for i,identity in enumerate(self.session['cohorts']['validation'][:2])]
                repeat_job=self.root/'state_repeats_job.json';atomic_json(repeat_job,dict(trials=repeat_specs))
                repeat_path=self.stage('state_repeats','visuotactile.native_v2_evaluation',['--job',str(repeat_job)])
                repeats=json.loads((repeat_path/'result.json').read_text())['trials']
            g4=visual_gate(g0,g1,g2,g3,repeats);self.write_gate(g4)
            if not g4.acceptance_passed:raise RuntimeError(g4.blocked_reason)
            self.status('state_accepted_visual_validity_pending',reason='State acceptance complete; visual small-scale validity implementation and verification remain; no formal test launched')
        except BaseException as error:
            atomic_json(self.root/'blocked.json',dict(reason=repr(error),utc=time.time(),automatic_retry=False))
            self.status('blocked',reason=repr(error))
            for path in (self.root/'gates').glob('*.json'):
                value=json.loads(path.read_text())
                if not value['acceptance_passed'] and 'pending' in str(value.get('blocked_reason')).lower():
                    value['blocked_reason']=repr(error);atomic_json(path,value)
        finally:
            if self.child is not None and self.child.poll() is None:
                os.killpg(self.child.pid,signal.SIGTERM)
                try:self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch()
                try:self.monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:self.monitor.terminate();self.monitor.wait(timeout=10)
                self.monitor_log.close()
            atomic_json(self.root/'owner_exit.json',dict(utc=time.time(),owned_children_exited=True))
            from .native_v2_report import report
            report(self.root)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['initialize','run','resume','status','report','stop'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/native_dp_v2.json'));a=p.parse_args()
    a.root=a.root.resolve()
    if a.command=='initialize':initialize(a.root,a.config)
    elif a.command=='status':print((a.root/'status.json').read_text())
    elif a.command=='stop':atomic_json(a.root/'STOP',dict(utc=time.time()))
    elif a.command=='report':
        from .native_v2_report import report
        report(a.root)
    else:
        with (a.root/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if a.command=='resume':
                history=a.root/'recovery_history';history.mkdir(exist_ok=True)
                for name in ('STOP','blocked.json'):
                    path=a.root/name
                    if path.exists():path.rename(history/(str(time.time_ns())+'_'+name))
            Runner(a.root).run()


if __name__=='__main__':main()
