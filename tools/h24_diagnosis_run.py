"""Persistent bounded owner for H24 recordings and diagnosis; never trains."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import numpy as np

from tools.h24_diagnosis_common import OLD, ROOT, TRAIN, THRESHOLDS, INFERENCE, read, sha, events
from tools.h24_runtime_recovery import native_startup_evidence
from tools.state_motion_recovery import shutdown_stall
from visuotactile.training_resources import atomic_json

CODE = Path(__file__).resolve().parents[1]
RUN = ['bash', 'visuotactile/run_cpu.sh']


def prepare(root):
    if root.exists():
        raise FileExistsError('Never overwrite or automatically repeat a registered run')
    if read(OLD/'status.json')['training_state'] != 'completed':
        raise RuntimeError('Original training not completed')
    session = read(OLD/'session.json')
    for name, expected in session['implementation_sha256'].items():
        if sha(CODE/name) != expected:
            raise ValueError('Original frozen code mismatch: '+name)
    manifests = read(OLD/'checkpoint_manifest.json')
    for row in manifests.values():
        if sha(row['path']) != row['sha256']:
            raise ValueError('Checkpoint mismatch: '+row['path'])
    old_sample = Path(read(OLD/'run_plan.json')['old_root'])/'h24_train/milestone_3000.pt'
    protected = {r['path']:r['sha256'] for r in manifests.values()}
    protected[str(old_sample)] = read(OLD/'run_plan.json')['sample_checkpoint_sha256']
    assert sha(old_sample) == protected[str(old_sample)]
    a = read(OLD/'sample_8000_loop/result.json')['trials']
    b = {r['identity']:r for r in read(OLD/'sample_10000_loop/result.json')['trials']}
    candidates = sorted(r['identity'] for r in a if r['cohort']=='validation'
        and r.get('contact',{}).get('object_contact') and not r['native_lift']
        and r['identity'] in b and b[r['identity']].get('contact',{}).get('object_contact')
        and not b[r['identity']]['native_lift'])
    validation = candidates[0]
    specs = {r['identity']:{k:r[k] for k in ('identity','cohort','seed')} for r in a}
    root.mkdir(parents=True)
    for folder in ('stages','jobs','videos','analysis','freeze'):
        (root/folder).mkdir()
    shutil.copy2(OLD/'freeze/benchmark.json', root/'freeze/benchmark.json')
    now = time.time()
    session.update(deadline=now+4*3600, resource_target_percent=40., owned_worker_roots=[str(root)],
        diagnosis_only=True, original_root=str(OLD), new_optimizer_updates=0)
    atomic_json(root/'session.json', session)
    plan = dict(started_epoch=now, deadline=session['deadline'], maximum_kit_starts=15,
        maximum_v1_attempts=2, maximum_takeovers=8, consecutive_identical_pre_scene_cap=2,
        worker_wall_limit_s=900, train=specs[TRAIN], validation=specs[validation],
        validation_selection='Lexicographic first common validation ID with object contact and no native lift at both 8000 and10000; descriptive selection, not a performance cohort',
        eligible_validation_ids=candidates, thresholds=THRESHOLDS, inference=INFERENCE,
        checkpoint_manifest=manifests, protected_checkpoints=protected,
        original_success=dict(path=str(OLD/'sample_8000_loop'/TRAIN), steps=342,
            result_sha256=sha(OLD/'sample_8000_loop'/TRAIN/'result.json'),
            trace_sha256=sha(OLD/'sample_8000_loop'/TRAIN/'trace.npz'), video_found=False),
        existing_videos=[str(p) for p in OLD.rglob('*.mp4')],
        prefix_rule='Fixed native12 boundaries: floor((first sustained object contact -1)/12)*12; ceil(first native lift/12)*12. Shadow student observes every actual expert tick and receives actual postprocessed target; no hidden-state reset or physical-state injection.',
        expansion_rule='Two train precontact probes first; require observable-prefix parity and a stage gain or >=20% lower target joint RMSE against the same online expert over first24 post-switch ticks before further train/lift or validation probes.',
        no_test40=True, no_retries_of_old_runs=True, isolated_singleton=True,
        no_policy_or_control_changes=True, physical_hidden_state_match_unproven=True)
    atomic_json(root/'plan.json', plan)
    sources = dict(session['implementation_sha256'])
    sources.update({str(p.relative_to(CODE)):sha(p) for p in sorted((CODE/'tools').glob('h24_diagnosis_*.py'))})
    sources['tools/h24-diagnosis-20261104-01.service'] = sha(CODE/'tools/h24-diagnosis-20261104-01.service')
    atomic_json(root/'frozen.json', dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources, session_sha256=sha(root/'session.json'), plan_sha256=sha(root/'plan.json')))
    atomic_json(root/'status.json', dict(state='registered', new_optimizer_updates=0, utc=time.time()))
    print(json.dumps(dict(root=str(root), validation=validation, checkpoints_verified=len(protected), deadline=session['deadline'])))


def verify(root):
    frozen = read(root/'frozen.json')
    if sha(root/'session.json') != frozen['session_sha256'] or sha(root/'plan.json') != frozen['plan_sha256']:
        raise ValueError('Registered diagnosis identity changed')
    for name, expected in frozen['source_sha256'].items():
        if sha(CODE/name) != expected:
            raise ValueError('Frozen source changed: '+name)


class Owner:
    def __init__(self, root):
        self.root = root
        self.plan = read(root/'plan.json')
        self.results = read(root/'results.json') if (root/'results.json').exists() else {}
        self.fault_signature = None
        self.fault_count = 0
        self.worker = None
        self.stopped = None

    def check(self):
        if time.time() >= self.plan['deadline']:
            raise TimeoutError('Original diagnosis deadline reached')
        if (self.root/'STOP').exists():
            raise RuntimeError('Requested stop')
        if shutil.disk_usage(self.root).free < 12*1024**3:
            raise RuntimeError('Disk reserve below12GiB')

    def stage(self, name, mode, cohort, updates=None, prefix=0, reference=None):
        if name in self.results:
            return self.results[name]
        if self.stopped:
            return None
        self.check()
        verify(self.root)
        output = self.root/'stages'/name
        if output.exists():
            raise RuntimeError('Existing incomplete attempt retained; cannot repeat '+name)
        count = len(list((self.root/'stages').glob('*_launch.json')))
        if count >= self.plan['maximum_kit_starts']:
            self.stopped = 'preallocated startup budget exhausted'
            return None
        spec = self.plan[cohort]
        job = dict(specs=[spec], video=True, expert_prefix_ticks=prefix)
        if updates:
            ck = self.plan['checkpoint_manifest'][f'sample/milestone_{updates}.pt']
            job.update(checkpoint=ck['path'], checkpoint_sha256=ck['sha256'], inference=INFERENCE)
        if prefix:
            job['prefix_reference'] = str(reference)
        job_path = self.root/'jobs'/(name+'.json')
        atomic_json(job_path, job)
        command = RUN+['-m','tools.h24_diagnosis_worker','--root',str(self.root),'--output',str(output),
            '--mode',mode,'--job',str(job_path)]
        started = time.time()
        with (self.root/'stages'/(name+'.log')).open('x') as log:
            self.worker = subprocess.Popen(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            atomic_json(self.root/'stages'/(name+'_launch.json'), dict(pid=self.worker.pid, command=command, utc=started))
            atomic_json(self.root/'status.json', dict(state='running', stage=name, owner_pid=os.getpid(),
                worker_pid=self.worker.pid, completed=len(self.results), utc=time.time()))
            try:
                while self.worker.poll() is None:
                    self.check()
                    reason = shutdown_stall(output, time.time())
                    if reason or time.time()-started > self.plan['worker_wall_limit_s']:
                        raise TimeoutError(reason or 'Worker wall bound reached')
                    time.sleep(1)
            except BaseException:
                if self.worker.poll() is None:
                    os.killpg(self.worker.pid, signal.SIGTERM)
                    try: self.worker.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.worker.pid, signal.SIGKILL)
                        self.worker.wait(timeout=15)
                raise
            finally:
                if self.worker.poll() is not None:
                    atomic_json(self.root/'stages'/(name+'_process_exit.json'),
                        dict(pid=self.worker.pid, code=self.worker.returncode, reaped=True, utc=time.time()))
        code = self.worker.returncode
        self.worker = None
        result_file = output/'result.json'
        if code == 0 and result_file.exists() and not (output/'worker_error.json').exists():
            result = read(result_file)
            if not result.get('software_complete') or len(result.get('trials',[])) != 1:
                raise RuntimeError('Invalid singleton completion record')
            result = dict(state='completed', mode=mode, updates=updates, cohort=cohort,
                path=str(output/spec['identity']), result=result['trials'][0], result_sha256=sha(result_file))
            self.fault_count = 0
            self.fault_signature = None
        else:
            evidence = native_startup_evidence(self.root/'stages', name)
            result = dict(state='software_missing' if evidence else 'failed', mode=mode, updates=updates,
                cohort=cohort, code=code, evidence=evidence, path=str(output), native_outcome=None)
            if evidence:
                signature = 'pre_scene_FileThread_SIGABRT'
                self.fault_count = self.fault_count+1 if self.fault_signature == signature else 1
                self.fault_signature = signature
                if self.fault_count >= 2:
                    self.stopped = 'two consecutive identical pre-scene FileThread faults'
            else:
                self.stopped = 'unclassified software error; retain evidence for focused repair'
        self.results[name] = result
        atomic_json(self.root/'results.json', self.results)
        return result

    def directory(self, name):
        row = self.results.get(name, {})
        return Path(row['path']) if row.get('state') == 'completed' and row['result'].get('legal') else None

    def takeovers(self):
        reference = self.directory('V3')
        baseline = {8000:self.directory('V1'),10000:self.directory('V2')}
        if not reference or not all(baseline.values()):
            return
        ref = np.load(reference/'trace.npz')
        e = events(ref, read(reference/'result.json'))
        contact = e['contact_tick']
        if contact is None or contact < 13:
            return
        switch = ((contact-1)//12)*12
        probes = []
        for updates in (10000,8000):
            name = f'T_train_{updates}_precontact'
            row = self.stage(name,'dp','train',updates,switch,reference)
            path = self.directory(name)
            if not path:
                return
            result = row['result']
            if not result['prefix_validation'] or not result['prefix_validation']['passed']:
                return
            probe = np.load(path/'trace.npz')
            original = np.load(baseline[updates]/'trace.npz')
            stop = min(switch+24,len(probe['target']),len(original['target']),len(ref['target']))
            old_error = float(np.sqrt(np.mean((original['target'][switch:stop]-ref['target'][switch:stop])**2)))
            new_error = float(np.sqrt(np.mean((probe['target'][switch:stop]-ref['target'][switch:stop])**2)))
            old_result = read(baseline[updates]/'result.json')
            progress = (int(result['success'])>int(old_result['success']) or
                result['native_subgoals']>old_result['native_subgoals'] or
                int(result['native_lift'])>int(old_result['native_lift']))
            probes.append(dict(name=name, baseline_error_rad=old_error, prefix_error_rad=new_error,
                compared_ticks=stop-switch, stage_gain=progress,
                informative=progress or (stop-switch>=12 and new_error <= .8*old_error)))
        informative = any(p['informative'] for p in probes)
        atomic_json(self.root/'takeover_expansion.json',dict(probes=probes, expand=informative,
            rule=self.plan['expansion_rule'], causal_root_proven=False))
        if not informative:
            return
        for cohort, expert_name in [('train','V3'),('validation','V6')]:
            reference = self.directory(expert_name)
            if reference is None:
                continue
            trace = np.load(reference/'trace.npz')
            e = events(trace,read(reference/'result.json'))
            switches = {}
            if cohort == 'validation' and e['contact_tick'] is not None and e['contact_tick'] >= 13:
                switches['precontact'] = ((e['contact_tick']-1)//12)*12
            if e['lift_tick'] is not None:
                tick = ((e['lift_tick']+11)//12)*12
                if tick > 0 and tick < len(trace['joint']):
                    switches['lift'] = tick
            for label, tick in switches.items():
                for updates in (8000,10000):
                    self.stage(f'T_{cohort}_{updates}_{label}','dp',cohort,updates,tick,reference)

    def run(self):
        for name, mode, cohort, updates in [('V1','dp','train',8000),('V2','dp','train',10000),
                ('V3','expert','train',None),('V4','dp','validation',8000),
                ('V5','dp','validation',10000),('V6','expert','validation',None)]:
            self.stage(name,mode,cohort,updates)
            if name == 'V1' and self.directory('V1') and not self.results['V1']['result']['success']:
                self.stage('V1_attempt2','dp','train',8000)
        self.takeovers()


def run(root):
    verify(root)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'owner_exit.json').exists():
            raise RuntimeError('Completed/stopped owner must not restart automatically')
        monitor_log = (root/'monitor.log').open('a')
        monitor = subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],
            stdout=monitor_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=CODE)
        owner = Owner(root)
        error = None
        try:
            owner.run()
        except BaseException as exc:
            error = repr(exc)
            atomic_json(root/'owner_error.json',dict(error=error,traceback=traceback.format_exc(),utc=time.time()))
        finally:
            (root/'STOP_MONITOR').touch()
            try: monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                monitor.terminate();monitor.wait(timeout=10)
            monitor_log.close()
        status = 'failed' if error else 'software_missing' if owner.stopped or any(r['state']!='completed' for r in owner.results.values()) else 'completed'
        atomic_json(root/'status.json',dict(state='postprocessing', simulation_state=status,
            stop_reason=error or owner.stopped, completed=len(owner.results), utc=time.time()))
        command = RUN+['-m','tools.h24_diagnosis_report','--root',str(root)]
        with (root/'analysis.log').open('a') as log:
            completed = subprocess.run(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=1800)
        integrity = {path:sha(path)==expected for path,expected in owner.plan['protected_checkpoints'].items()}
        verify(root)
        atomic_json(root/'final_integrity.json',dict(checkpoints=integrity, passed=all(integrity.values()),
            frozen_sources_verified=True, new_optimizer_updates=0, utc=time.time()))
        final = dict(state=status if completed.returncode==0 and all(integrity.values()) else 'failed',
            simulation_state=status, analysis_exit=completed.returncode, stop_reason=error or owner.stopped,
            completed=len(owner.results), new_optimizer_updates=0, utc=time.time())
        atomic_json(root/'status.json',final)
        atomic_json(root/'owner_exit.json',dict(**final,pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    os.chdir(CODE)
    (prepare if args.command=='prepare' else run)(args.root.resolve())
