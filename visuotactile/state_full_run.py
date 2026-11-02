"""Bounded resumable owner for pilots, train320, validation, frozen formal test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .compact import sha256
from .native_v2_repair import RepairRunner
from .training_resources import atomic_json


def initialize(root, preflight):
    oldroot = Path('outputs/visuotactile/native_dp_v2/state_history_20261102_02').resolve()
    old = json.loads((oldroot/'session.json').read_text())
    root.mkdir(parents=True, exist_ok=False); started = time.time()
    source = Path(old['source_root']); data = json.loads((source/'data.json').read_text())
    if sha256(source/'data.json') != old['data_sha256'] or sha256(old['expert_checkpoint']) != old['expert_sha256']:
        raise ValueError('Previously audited data/expert identity changed')
    files = list(Path('visuotactile').rglob('*.py'))+list(Path('isaacsimenvs/tasks').rglob('*.py'))
    session = dict(old, protocol_version=5, started_epoch=started, deadline=started+24*3600,
        training_deadline=started+21*3600, resource_policy='shared_recent_v1',
        predecessor=str(oldroot), preflight=str(preflight.resolve()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha256={str(p): sha256(p) for p in files},
        pilot_updates=300, bc_updates=10000, dp_updates=19000,
        bc_milestones=[1000, 3000, 6000, 10000], dp_milestones=[1000, 3000, 6000, 10000, 19000],
        early_stop=dict(minimum_updates=6000, relative_improvement=.01, stale_milestones=2),
        automatic_expansion=True, automatic_retry='one legal checkpoint recovery per training stage; never rerun completed trajectories',
        kit_isolation='one fresh Kit per controlled trajectory; stale articulation scene-query cache after reset',
        test_calls=2, maximum_controlled_trajectories=174,
        formal_specs=[dict(identity=i, cohort='test', seed=983101+n) for n, i in enumerate(sorted(data['splits']['test'])[:20])],
        initialization='same preregistered 4 fresh + train4/validation4; sorted first20 test IDs, no outcome filtering',
        admission='passed unchanged numeric audit; legal expert>=10/12 and saved>=6/8; two exact command replays',
        training_scope='pilots4 then full train320/validation40; test40 only after frozen EMA selection')
    atomic_json(root/'session.json', session)
    atomic_json(root/'gpu_schedule.json', dict(policy='shared_recent_v1', fixed_utilization_veto=False,
        free_vram_mib=6144, available_ram_gib=12, worker_count=1, external_throughput_observable=False,
        recent_window_seconds=[60, 300], pacing='external present .25; recent1m>85:1; 1m>92 and5m>85:2'))
    for name in ('preservation', 'parity'):
        relevant = ('visuotactile/state_history.py', 'visuotactile/environment_rng.py', 'visuotactile/state_history_audit.py',
            'visuotactile/native_data.py', 'visuotactile/models/native_dp.py')
        for path in relevant:
            if old['implementation_sha256'][path] != sha256(path):
                raise ValueError('Prior numerical audit is stale: '+path)
        shutil.copytree(oldroot/name, root/name)
        atomic_json(root/(name+'_committed.json'), dict(sha256=sha256(root/name/'result.json'),
            reused_from=str(oldroot/name), immutable_numeric_sources=True))
    for name in ('freeze',):
        if not (preflight/name/'result.json').exists():
            raise ValueError('Preflight stage incomplete: '+name)
        shutil.copytree(preflight/name, root/name)
        atomic_json(root/(name+'_committed.json'), dict(sha256=sha256(root/name/'result.json'), reused_from=str(preflight/name)))
    atomic_json(root/'status.json', dict(state='registered', new_updates=0))


def should_stop(history, minimum=6000):
    if len(history) < 3 or history[-1]['update'] < minimum:
        return False
    earlier = min(r['validation']['selection_mse'] for r in history[:-2])
    return min(r['validation']['selection_mse'] for r in history[-2:]) >= .99*earlier


def recover_preaction_start(root, name, deadline):
    """Retry one known pre-scene FileThread abort, with a session-wide bound."""
    root = Path(root); output = root/name; log = root/(name+'.log')
    recoveries = root/'startup_recoveries'
    if time.time() >= deadline or not output.is_dir() or not log.is_file():
        return False
    if (recoveries/name).exists() or len(list(recoveries.glob('*/recovery.json'))) >= 3:
        return False
    files = {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()}
    contents = log.read_text(errors='replace')
    if files != {'runtime_controls.json'} or not all(
        signature in contents for signature in ('FileThread', 'std::system_error', 'abort')):
        return False
    if '[INFO]: Completed setting up the environment' in contents or 'Traceback (most recent call last)' in contents:
        return False
    archive = recoveries/name; archive.mkdir(parents=True)
    output.rename(archive/'attempt')
    log.rename(archive/'worker.log')
    launch = root/(name+'_launch.json')
    if launch.exists():
        launch.rename(archive/'launch.json')
    atomic_json(archive/'recovery.json', dict(utc=time.time(), stage=name, deadline=deadline,
        reason='Known native FileThread abort before scene/trajectory; only runtime_controls existed',
        original_log_sha256=sha256(archive/'worker.log'), same_seed_and_job=True,
        new_controlled_trajectories=0, maximum_per_stage=1, maximum_per_session=3))
    return True


class Runner(RepairRunner):
    def stage(self, name, module='visuotactile.state_benchmark', extra=None):
        output = self.root/name; marker = self.root/(name+'_committed.json')
        for path, digest in self.session['implementation_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Registered source drift: '+path)
        if output.exists() and not marker.exists():
            if (output/'result.json').exists() and (self.root/(name+'_launch.json')).exists():
                result = json.loads((output/'result.json').read_text())
                if result.get('software_complete'):
                    atomic_json(marker, dict(sha256=sha256(output/'result.json'), recovered_completion=True))
            elif module == 'visuotactile.state_full_training':
                extra = extra or []; kind = extra[extra.index('--kind')+1]
                directory = self.root/(kind+('_pilot_train' if '--pilot' in extra else '_train'))
                if not (directory/'last.pt').exists() or (self.root/(name+'_recovery.json')).exists():
                    raise RuntimeError('No legal checkpoint or recovery bound reached: '+name)
                output.rename(self.root/(name+'_interrupted'))
                atomic_json(self.root/(name+'_recovery.json'), dict(utc=time.time(),
                    checkpoint_sha256=sha256(directory/'last.pt'), deadline_unchanged=True))
            elif module == 'visuotactile.state_benchmark' and recover_preaction_start(
                    self.root, name, self.session['deadline']):
                pass
            else:
                raise RuntimeError('Incomplete simulation attempt preserved, no blind repeat: '+name)
        try:
            return super().stage(name, module, extra)
        except RuntimeError:
            if module == 'visuotactile.state_benchmark' and recover_preaction_start(
                    self.root, name, self.session['deadline']):
                return self.stage(name, module, extra)
            if module != 'visuotactile.state_full_training' or (self.root/(name+'_recovery.json')).exists():
                raise
            return self.stage(name, module, extra)

    def evaluate(self, name, mode, checkpoint=None, specs=None):
        if specs is None:
            specs = json.loads((self.root/'freeze/benchmark.json').read_text())['specs']
        output = self.root/name; output.mkdir(exist_ok=True)
        trials = []
        for ordinal, spec in enumerate(specs):
            child = f'{name}_single_{ordinal:02d}'
            job = self.root/(child+'_job.json')
            values = dict(specs=[spec], video=ordinal == 0)
            if checkpoint is not None:
                values.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
            if job.exists() and json.loads(job.read_text()) != values:
                raise ValueError('Frozen evaluation job changed')
            atomic_json(job, values)
            result = self.stage(child, extra=['--mode', mode, '--job', str(job)])
            if len(result['trials']) != 1:
                raise ValueError('Singleton trajectory count mismatch')
            trials += result['trials']
            destination = output/spec['identity']
            if not destination.exists():
                destination.symlink_to((self.root/child/spec['identity']).resolve(), target_is_directory=True)
            atomic_json(output/'progress.json', dict(completed=len(trials), trials=trials))
        result = dict(software_complete=True, mode=mode, trials=trials, fresh_kit_per_trial=True)
        atomic_json(output/'result.json', result)
        return result

    def run(self):
        try:
            (self.root/'STOP_MONITOR').unlink(missing_ok=True)
            (self.root/'owner_exit.json').unlink(missing_ok=True)
            with (self.root/'monitor.log').open('a') as log:
                self.monitor = subprocess.Popen(['bash', 'visuotactile/run_cpu.sh', '-m', 'visuotactile.state_resources',
                    '--root', str(self.root)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            numeric = self.stage('parity', 'visuotactile.state_history_audit', [])
            self.stage('freeze', extra=['--mode', 'freeze'])
            benchmark = json.loads((self.root/'freeze/benchmark.json').read_text())['specs']
            expert = self.evaluate('expert', 'expert', specs=benchmark)
            replay = self.evaluate('replay', 'replay', specs=benchmark[:2])
            saved_specs = [s for s in benchmark if s['cohort'] != 'fresh']
            short_specs = sum(([s for s in saved_specs if s['cohort'] == c][:2] for c in ('train', 'validation')), [])
            passed = (numeric['passed'] and len(expert['trials']) == 12 and all(r['legal'] for r in expert['trials'])
                and sum(r['success'] for r in expert['trials']) >= 10
                and sum(r['success'] for r in expert['trials'] if r['cohort'] != 'fresh') >= 6
                and len(replay['trials']) == 2 and all(r.get('bytewise_fields') and all(r['bytewise_fields'].values()) for r in replay['trials']))
            names = ['preservation/result.json', 'parity/result.json', 'freeze/benchmark.json', 'expert/result.json', 'replay/result.json']
            atomic_json(self.root/'foundation.json', dict(passed=bool(passed), evidence_sha256={n: sha256(self.root/n) for n in names},
                historical_open_loop_is_training_veto=False, expert_successes=sum(r['success'] for r in expert['trials'])))
            if not passed:
                raise RuntimeError('Current legal expert/control foundation failed')
            for kind in ('bc', 'dp'):
                result = self.stage(kind+'_pilot', 'visuotactile.state_full_training',
                    ['--kind', kind, '--target', str(self.session['pilot_updates']), '--pilot'])
                if not result['history_gradient_nonzero'] or result['last25_loss'] >= result['first25_loss']:
                    raise RuntimeError(kind+' pilot did not demonstrate valid learning/GRU gradient')
                self.evaluate(kind+'_pilot_loop', kind, self.root/(kind+'_pilot_train/best.pt'), short_specs)
            for kind in ('bc', 'dp'):
                for target in self.session[kind+'_milestones']:
                    if time.time() >= self.session['training_deadline']:
                        raise TimeoutError('Training budget expired; reserve final three hours for existing weights')
                    result = self.stage(f'{kind}_train_{target}', 'visuotactile.state_full_training',
                        ['--kind', kind, '--target', str(target)])
                    self.evaluate(f'{kind}_loop_{target}', kind, self.root/(kind+'_train')/f'milestone_{target}.pt', short_specs)
                    if should_stop(result['history']):
                        atomic_json(self.root/(kind+'_early_stop.json'), dict(updates=target,
                            reason='Two milestones without >=1% validation improvement; no test used'))
                        break
                import torch
                checkpoint = self.root/(kind+'_train/best.pt')
                chosen = torch.load(checkpoint, map_location='cpu', weights_only=False)
                selection = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    selected_updates=chosen['updates'], criterion='minimum validation40 teacher-forced prefix12 equal arm/hand MSE',
                    test_used_for_selection=False, configuration_frozen=True)
                del chosen
                atomic_json(self.root/(kind+'_selection.json'), selection)
                self.evaluate(kind, kind, checkpoint, benchmark)
            self.evaluate('old', 'old', Path(self.session['old_state_checkpoint']), benchmark)
            atomic_json(self.root/'formal_selection_freeze.json', dict(utc=time.time(),
                models={k: json.loads((self.root/(k+'_selection.json')).read_text()) for k in ('bc', 'dp')},
                specs=self.session['formal_specs'], historical_test_usage_disclosed=True))
            for kind in ('bc', 'dp'):
                self.stage(kind+'_test_offline', 'visuotactile.state_full_training',
                    ['--kind', kind, '--target', str(self.session[kind+'_updates']), '--final-test'])
            for mode in ('expert', 'old', 'bc', 'dp'):
                checkpoint = (None if mode == 'expert' else Path(self.session['old_state_checkpoint']) if mode == 'old'
                    else self.root/(mode+'_train/best.pt'))
                self.evaluate(mode+'_formal', mode, checkpoint, self.session['formal_specs'])
            atomic_json(self.root/'status.json', dict(state='complete', utc=time.time(),
                research_success='Read native trial outcomes; training completion is not task success'))
        except BaseException as error:
            atomic_json(self.root/'blocked.json', dict(reason=repr(error), utc=time.time()))
            atomic_json(self.root/'status.json', dict(state='blocked', reason=repr(error), utc=time.time()))
        finally:
            if self.worker is not None and self.worker.poll() is None:
                os.killpg(self.worker.pid, signal.SIGTERM)
                try:
                    self.worker.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(self.worker.pid, signal.SIGKILL); self.worker.wait()
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch(); self.monitor.wait(timeout=20)
            atomic_json(self.root/'owner_exit.json', dict(utc=time.time(), owned_children_exited=True))
            subprocess.run(['bash', 'visuotactile/run_cpu.sh', 'tools/state_history_report.py', '--root', str(self.root),
                '--output', str(self.root/'report')], check=False)
            subprocess.run(['bash', 'visuotactile/run_cpu.sh', 'tools/state_history_finish.py', '--root', str(self.root),
                '--output', str(Path('research_docs/evidence/state_shared_20261102/results'))], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['initialize', 'run'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--preflight', type=Path)
    a = p.parse_args()
    if a.command == 'initialize':
        initialize(a.root.resolve(), a.preflight.resolve())
    else:
        with (a.root/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); Runner(a.root.resolve()).run()
