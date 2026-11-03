"""Bounded persistent owner for G0-A/B/C repair; no old experiment is resumed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .compact import sha256
from .native_v2_gates import data_gate, historical_replay_gate, infrastructure_gate, online_expert_gate
from .training_resources import atomic_json


def initialize(root):
    previous = Path('outputs/visuotactile/native_dp_v2/state_first_20261102_01').resolve()
    old = json.loads((previous/'session.json').read_text())
    data = json.loads((Path(old['source_root'])/'data.json').read_text())
    expert = Path('../bc+rl_RSS/pretrained_assembly/tight_insertion/model.pth').resolve()
    if sha256(expert) != '7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990':
        raise ValueError('Released expert identity mismatch')
    root.mkdir(parents=True, exist_ok=False); (root/'gates').mkdir()
    started = time.time()
    source_files = list(Path('visuotactile').rglob('*.py')) + list(Path('isaacsimenvs/tasks/play').rglob('*.py'))
    session = dict(protocol_version=3, created=started, started_epoch=started, deadline=started+8*3600,
        training_deadline=started+8*3600, source_root=old['source_root'], previous_root=str(previous),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha256={str(path): sha256(path) for path in sorted(source_files)},
        expert_checkpoint=str(expert), expert_sha256=sha256(expert),
        old_state_checkpoint=old['checkpoints']['best']['path'],
        old_state_sha256=old['checkpoints']['best']['sha256'],
        single_seeds=list(range(982101, 982109)), four_lane_seeds=[982201, 982202],
        expert_minimum_successes=6, historical_world_ids=[old['cohorts']['replay'][index] for index in (1, 2)],
        bc_train_ids=data['splits']['train'][:4], bc_validation_ids=old['cohorts']['validation'][:4],
        bc_updates=1000, maximum_primary_starts=52, automatic_trajectory_retries=0,
        full_dp_training_authorized_by_this_runner=False, test_calls=0,
        scope='Fresh expert1 eight; expert4 two batches/eight lanes; replay2; source-world2; old State8; BCs16; StateBC saved initials8')
    atomic_json(root/'session.json', session)
    atomic_json(root/'gpu_schedule.json', dict(target_percent=38, admission_percent=42, min_gpu_free_mib=6144, min_ram_gib=14))
    numeric = data_gate(json.loads((previous/'data_audit.json').read_text()), json.loads((previous/'control_audit.json').read_text()))
    atomic_json(root/'gates/G0-A.json', numeric.to_dict())
    rows = json.loads((previous/'diagnostics/result.json').read_text())['trials']
    historical = historical_replay_gate(json.loads((previous/'data_audit.json').read_text()),
        [row for row in rows if row['kind'] == 'replay'], old['cohorts']['replay'])
    atomic_json(root/'gates/G0-C.json', historical.to_dict())
    atomic_json(root/'status.json', dict(state='registered', new_optimizer_updates=0))


def recover_preaction(root):
    session = json.loads((root/'session.json').read_text())
    if time.time() >= session['deadline'] or session.get('preaction_recoveries', 0) >= 1:
        raise RuntimeError('Recovery bound or original deadline reached')
    output = root/'expert1'
    if list(output.glob('seed_*')) or list(output.rglob('*.h5')) or (output/'result.json').exists():
        raise RuntimeError('A trajectory started; no automatic repeat permitted')
    archive = root/'preaction_failure_1'; archive.mkdir(exist_ok=False)
    for name in ('expert1', 'expert1.log', 'blocked.json', 'owner_exit.json'):
        path = root/name
        if path.exists():
            path.rename(archive/name)
    atomic_json(archive/'registered_session.json', session)
    session['preaction_recoveries'] = 1
    session['implementation_sha256'] = {name: sha256(name) for name in session['implementation_sha256']}
    atomic_json(root/'session.json', session)
    atomic_json(root/'recovery.json', dict(utc=time.time(), reason='Metadata requested before first reset; no action or trajectory began',
        original_deadline=session['deadline'], retries=1, source_change='Postpone sensor metadata until native reset; save exceptions before Kit shutdown'))


class RepairRunner:
    def __init__(self, root):
        self.root = root; self.session = json.loads((root/'session.json').read_text())
        self.worker = None; self.monitor = None; self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))

    def check(self):
        if self.stop or (self.root/'STOP').exists():
            raise InterruptedError('Owned experiment stop')
        if time.time() >= self.session['deadline']:
            raise TimeoutError('Fixed eight-hour repair deadline expired')
        if self.monitor is not None and self.monitor.poll() is not None:
            raise RuntimeError('Continuous monitor exited')

    def stage(self, name, module='visuotactile.native_v2_expert', extra=None):
        self.check(); output = self.root/name
        marker = self.root/(name+'_committed.json')
        if marker.exists():
            if sha256(output/'result.json') != json.loads(marker.read_text())['sha256']:
                raise ValueError('Completed result changed')
            return json.loads((output/'result.json').read_text())
        if output.exists():
            raise RuntimeError('Attempt exists; preserve it without automatic trajectory retries: '+name)
        for path, digest in self.session['implementation_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Registered implementation drift: '+path)
        args = ['bash', 'visuotactile/run_cpu.sh', '-m', module, '--root', str(self.root), '--output', str(output)]
        args += extra if extra is not None else ['--mode', name]
        atomic_json(self.root/(name+'_launch.json'), dict(command=args, utc=time.time()))
        with (self.root/(name+'.log')).open('a') as stream:
            self.worker = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            atomic_json(self.root/'status.json', dict(state='running', stage=name, pid=os.getpid(),
                                                    worker_pid=self.worker.pid, utc=time.time()))
            while self.worker.poll() is None:
                self.check(); time.sleep(2)
            code = self.worker.returncode
            atomic_json(self.root/(name+'_process_exit.json'),dict(pid=self.worker.pid,
                code=code,utc=time.time(),reaped=True))
            self.worker = None
        if code != 0 or not (output/'result.json').exists():
            raise RuntimeError(f'{name} failed: exit={code}')
        result = json.loads((output/'result.json').read_text())
        if result.get('software_complete') is not True:
            raise RuntimeError(name+' incomplete software stage')
        if module=='visuotactile.state_benchmark' and self.session.get('position_study'):
            self.verify_worker_exit(name)
        atomic_json(marker, dict(sha256=sha256(output/'result.json'), utc=time.time()))
        return result

    def run(self):
        try:
            (self.root/'STOP_MONITOR').unlink(missing_ok=True)
            with (self.root/'monitor.log').open('a') as stream:
                self.monitor = subprocess.Popen(['bash', 'visuotactile/run_cpu.sh', '-m',
                    'visuotactile.native_resources', '--root', str(self.root)], stdout=stream,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            from .native_v2_gates import Gate
            numeric = Gate(**json.loads((self.root/'gates/G0-A.json').read_text()))
            if not numeric.acceptance_passed:
                raise RuntimeError('G0-A data/control failed')
            expert = self.stage('expert1')
            environment = json.loads((self.root/'expert1/environment.json').read_text())
            online = online_expert_gate(expert['trials'], [f's{seed}_lane0' for seed in self.session['single_seeds']],
                self.session['expert_sha256'], environment['sha256'], self.session['expert_minimum_successes'])
            atomic_json(self.root/'gates/G0-B.json', online.to_dict())
            combined = infrastructure_gate(numeric, online)
            atomic_json(self.root/'gates/G0.json', combined.to_dict())
            self.stage('expert4'); self.stage('fresh_replay'); self.stage('historical_world')
            self.stage('old_state')
            if not combined.acceptance_passed:
                raise RuntimeError('G0-B fresh expert failed registered reliability; investigate environment before learning')
            self.stage('bc_compare', 'visuotactile.native_v2_bc_compare', [])
            self.stage('bc_state'); self.stage('bc_actor')
            checkpoint = self.root/'bc_compare/state'/f"milestone_{self.session['bc_updates']}.pt"
            trials = []
            for cohort, identities in [('train', self.session['bc_train_ids']), ('validation', self.session['bc_validation_ids'])]:
                for ordinal, identity in enumerate(identities):
                    trials.append(dict(name=f'{cohort}_{ordinal}', kind='student', identity=identity, cohort=cohort,
                        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), seed=7091, video=ordinal == 0))
            job = self.root/'bc_saved_job.json'; atomic_json(job, dict(trials=trials))
            self.stage('bc_saved', 'visuotactile.native_v2_evaluation', ['--job', str(job)])
            atomic_json(self.root/'status.json', dict(state='diagnostics_complete', utc=time.time(),
                reason='Review native BC results before any larger DP run; no automatic19000 updates'))
        except BaseException as error:
            atomic_json(self.root/'blocked.json', dict(reason=repr(error), utc=time.time(), automatic_retry=False))
            atomic_json(self.root/'status.json', dict(state='blocked', reason=repr(error), utc=time.time()))
        finally:
            if self.worker is not None and self.worker.poll() is None:
                os.killpg(self.worker.pid, signal.SIGTERM)
                try:
                    self.worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(self.worker.pid, signal.SIGKILL); self.worker.wait()
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch()
                try:
                    self.monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.monitor.terminate(); self.monitor.wait(timeout=10)
            atomic_json(self.root/'owner_exit.json', dict(utc=time.time(), owned_children_exited=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['initialize', 'run', 'status', 'recover-preaction'])
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args(); root = arguments.root.resolve()
    if arguments.command == 'initialize':
        initialize(root)
    elif arguments.command == 'status':
        print((root/'status.json').read_text())
    elif arguments.command == 'recover-preaction':
        recover_preaction(root)
    else:
        with (root/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            RepairRunner(root).run()
