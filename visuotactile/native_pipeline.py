"""Independent resumable three-way experiment; no model-mediated supervision."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .compact import sha256
from .native_data import CONTRACT, GROUPS, MANIFEST, BASE, prepare
from .native_training import common_budget
from .training_resources import atomic_json


def sources():
    paths = list(Path('visuotactile').rglob('*.py'))+list(Path('isaacsimenvs/tasks/play').rglob('*.py'))
    paths += [Path('visuotactile/run_cpu.sh')]
    return {str(path): sha256(path) for path in sorted(paths)}


def protected():
    old = json.loads(Path('research_docs/evidence/t0_replay_20261031/protected_artifacts.json').read_text())
    result = {}
    for row in old.values():
        path = Path(row['path'])
        digest = sha256(path)
        if digest != row['sha256']:
            raise ValueError('Protected artifact drift: '+str(path))
        result[str(path)] = dict(sha256=digest, bytes=path.stat().st_size)
    for name in ('expert_numeric.h5', 'expert_inputs.h5', 'initial_export.pt'):
        path = Path('outputs/visuotactile/run/t0_control_replay_20261031_01/fresh')/name
        result[str(path)] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    return result


def initialize(root):
    if root.exists():
        raise FileExistsError('A new experiment root is mandatory')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    if branch != 'feature/trained-tactile-residual':
        raise ValueError('Unexpected experiment branch')
    evidence = protected()
    root.mkdir(parents=True)
    session = dict(version=1, contract=CONTRACT, created=time.time(), started_epoch=None,
        gate_deadline=None, training_deadline=None, deadline=None,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_hashes=sources(), protected=evidence, branch=branch,
        test_cohort='lexically sorted first20 held-out IDs, once per model',
        formal_trials=60, final_train_trials=12, automatic_retries=0)
    atomic_json(root/'session.json', session)
    atomic_json(root/'manifest.json', dict(state='initialized', stages={}, created=session['created']))
    atomic_json(root/'gpu_schedule.json', dict(target_percent=40, admission_percent=45,
        min_gpu_free_mib=6144, min_ram_gib=14))
    inventory = []
    for path in Path('outputs/visuotactile').rglob('*'):
        if path.is_file():
            inventory.append(dict(path=str(path), bytes=path.stat().st_size, disposition='retain; reference audit pending'))
    atomic_json(root/'cleanup_inventory.json', dict(files=inventory, deleted=[], released_bytes=0))


def valid_artifacts(stage):
    if stage.get('state') != 'complete' or stage.get('exit_code') != 0:
        return False
    for name, digest in stage.get('artifacts', {}).items():
        if not Path(name).is_file() or sha256(name) != digest:
            return False
    return bool(stage.get('artifacts'))


def review_pre_freeze_update(root, reason):
    if (root/'frozen.json').exists():
        raise ValueError('Formal configuration frozen; source migration forbidden')
    session = json.loads((root/'session.json').read_text())
    current = sources()
    changed = {name for name in set(current) | set(session['source_hashes'])
               if current.get(name) != session['source_hashes'].get(name)}
    allowed = ('visuotactile/native_', 'visuotactile/models/native_', 'visuotactile/sim/native_')
    if not changed or any(not name.startswith(allowed) for name in changed):
        raise ValueError('Reviewed update must be restricted to the new native DP modules')
    session.setdefault('pre_freeze_source_reviews', []).append(dict(utc_epoch=time.time(), reason=reason,
        changed=sorted(changed), previous=session['source_hashes'], current=current,
        original_started_epoch=session['started_epoch'], original_deadline=session['deadline']))
    session['source_hashes'] = current
    atomic_json(root/'session.json', session)


def stop_owned(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


class Pipeline:
    def __init__(self, root, resume=False):
        self.root, self.resume = root, resume
        self.session = json.loads((root/'session.json').read_text())
        self.manifest = json.loads((root/'manifest.json').read_text())
        self.monitor = None
        self.child = None
        self.current = None
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))

    def write(self):
        atomic_json(self.root/'manifest.json', self.manifest)

    def check_identity(self):
        if sources() != self.session['source_hashes']:
            raise ValueError('Source hashes changed after experiment registration')

    def start_gpu_clock(self):
        if self.session['started_epoch'] is None:
            now = time.time()
            self.session.update(started_epoch=now, gate_deadline=now+3*3600,
                training_deadline=now+21*3600, deadline=now+24*3600)
            atomic_json(self.root/'session.json', self.session)
        if time.time() >= self.session['deadline']:
            raise TimeoutError('Original immutable 24-hour deadline expired')
        (self.root/'STOP_MONITOR').unlink(missing_ok=True)
        self.monitor_log = (self.root/'monitor.log').open('a')
        self.monitor = subprocess.Popen(['bash', 'visuotactile/run_cpu.sh', '-m', 'visuotactile.native_resources',
            '--root', str(self.root)], stdin=subprocess.DEVNULL, stdout=self.monitor_log,
            stderr=subprocess.STDOUT, start_new_session=True)

    def stage(self, name, module, arguments, *, budget='gate_deadline', gpu=True):
        self.check_identity()
        self.current = name
        previous = self.manifest['stages'].get(name, {})
        if valid_artifacts(previous):
            return Path(previous['output'])
        if previous and not self.resume:
            raise RuntimeError('Incomplete attempt requires explicit resume: '+name)
        attempt = len(previous.get('attempts', []))+1
        output = self.root/'stages'/name/('attempt_'+str(attempt))
        output.parent.mkdir(parents=True, exist_ok=True)
        command = ['bash', 'visuotactile/run_cpu.sh', '-m', module, '--root', str(self.root), '--output', str(output)]+arguments
        if self.resume and previous and module == 'visuotactile.native_training' and '--stage' in arguments:
            training_stage = arguments[arguments.index('--stage')+1]
            old = Path(previous['output'])
            if training_stage in ('fit1', 'fit4', 'pilot', 'formal') and (old/'last.pt').exists():
                output.mkdir()
                for filename in ('last.pt', 'best.pt', 'validation.json', 'train.jsonl'):
                    if (old/filename).exists():
                        shutil.copy2(old/filename, output/filename)
                command.append('--resume')
        attempts = previous.get('attempts', [])
        row = dict(state='starting', attempt=attempt, command=command, output=str(output),
            dependencies=[key for key, value in self.manifest['stages'].items() if value.get('state') == 'complete'],
            started=time.time(), source_hashes=self.session['source_hashes'], attempts=attempts)
        attempts.append(dict(number=attempt, output=str(output), command=command, started=row['started']))
        self.manifest['stages'][name] = row
        self.manifest.update(state='running', current_stage=name, pid=os.getpid())
        self.write()
        with (output.parent/('attempt_'+str(attempt)+'.log')).open('w') as log:
            self.child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            row['pid'] = self.child.pid
            self.write()
            while self.child.poll() is None:
                if self.stop or (self.root/'STOP').exists():
                    stop_owned(self.child)
                    raise InterruptedError('Explicit stop; completed artifacts and last checkpoint retained')
                if gpu and (time.time() >= self.session[budget] or self.monitor.poll() is not None):
                    stop_owned(self.child)
                    raise TimeoutError('Stage deadline expired or resource monitor exited')
                state = 'running'
                resource_path = self.root/'resources/resources.json'
                if gpu and resource_path.exists():
                    resource = json.loads(resource_path.read_text())
                    latest = resource['latest']
                    if ((self.root/'RESOURCE_PAUSE').exists() or latest['rolling_60s_percent'] is None
                            or latest['rolling_60s_percent'] > 45 or latest['gpu_free_mib'] is None
                            or latest['gpu_free_mib'] < 6144 or latest['ram_available_bytes'] < 14*1024**3):
                        state = 'waiting_resources'
                atomic_json(self.root/'status.json', dict(state=state, stage=name, pid=self.child.pid,
                    stage_output=str(output), utc_epoch=time.time(), deadline=self.session['deadline']))
                time.sleep(2)
            row['exit_code'] = self.child.returncode
        self.child = None
        result_path = output/'result.json'
        if row['exit_code'] != 0 or not result_path.exists():
            raise RuntimeError('Worker failed or completion marker absent: '+name)
        result = json.loads(result_path.read_text())
        if not result.get('passed'):
            raise RuntimeError('Stage acceptance failed: '+name)
        if 'required' in result and result.get('completed') != result['required']:
            raise RuntimeError('Incomplete evaluation cohort')
        artifacts = {str(result_path): sha256(result_path)}
        for filename in ('best.pt', 'last.pt'):
            if (output/filename).exists():
                artifacts[str(output/filename)] = sha256(output/filename)
        row.update(state='complete', completed=time.time(), artifacts=artifacts)
        attempts[-1].update(exit_code=row['exit_code'], state='complete', artifacts=artifacts)
        self.write()
        return output

    def training(self, group, stage):
        return self.stage(group+'_'+stage, 'visuotactile.native_training', ['--group', group, '--stage', stage],
                          budget='training_deadline' if stage == 'formal' else 'deadline' if stage == 'test' else 'gate_deadline')

    def evaluation(self, group, checkpoint, cohort, label):
        return self.stage(group+'_'+label, 'visuotactile.native_evaluation',
            ['--mode', 'autonomous', '--group', group, '--checkpoint', str(checkpoint), '--cohort', cohort],
            budget='gate_deadline' if cohort == 'fit' else 'deadline')

    def run(self):
        try:
            self.stage('data', 'visuotactile.native_pipeline', ['worker', '--task', 'data'], gpu=False)
            self.start_gpu_clock()
            self.stage('oracle_base', 'visuotactile.native_evaluation', ['--mode', 'oracle-base'])
            self.stage('oracle_instrumented', 'visuotactile.native_evaluation', ['--mode', 'oracle-instrumented'])
            pilots, validation = {}, {}
            for group in GROUPS:
                self.training(group, 'network')
                for stage in ('fit1', 'fit4'):
                    output = self.training(group, stage)
                    self.evaluation(group, output/'best.pt', 'fit', stage+'_diagnostic')
                output = self.training(group, 'pilot')
                result = json.loads((output/'result.json').read_text())
                pilots[group], validation[group] = result['seconds_per_update'], result['metrics']['elapsed_s']
            if not (self.root/'frozen.json').exists():
                budget = common_budget(self.session['training_deadline']-time.time(), pilots, validation)
                atomic_json(self.root/'frozen.json', dict(common_updates=budget, contract=CONTRACT,
                    pilots_s=pilots, validation_s=validation, source_hashes=self.session['source_hashes'], frozen=time.time()))
            selected = {}
            for group in GROUPS:
                output = self.training(group, 'formal')
                selected[group] = dict(path=str((output/'best.pt').resolve()), sha256=sha256(output/'best.pt'))
            if (self.root/'selected.json').exists() and json.loads((self.root/'selected.json').read_text()) != selected:
                raise ValueError('Validation-selected identities changed after freeze')
            atomic_json(self.root/'selected.json', selected)
            for group in GROUPS:
                self.training(group, 'test')
            for group in GROUPS:
                self.evaluation(group, Path(selected[group]['path']), 'train', 'train_diagnostic')
            for group in GROUPS:
                self.evaluation(group, Path(selected[group]['path']), 'formal', 'formal_evaluation')
            self.stage('verify', 'visuotactile.native_pipeline', ['worker', '--task', 'verify'], budget='deadline')
            self.manifest.update(state='complete', ended=time.time())
        except BaseException as error:
            if self.child is not None:
                stop_owned(self.child)
                self.child = None
            stage = self.manifest['stages'].get(self.current)
            if stage and stage.get('state') != 'complete':
                stage.update(state='blocked', reason=repr(error), ended=time.time())
                stage['attempts'][-1].update(state='blocked', reason=repr(error))
            self.manifest.update(state='blocked', reason=repr(error), ended=time.time())
            atomic_json(self.root/'blocked.json', dict(stage=self.current, reason=repr(error),
                deadline=self.session['deadline'], utc_epoch=time.time(), automatic_retry=False))
        finally:
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch()
                try:
                    self.monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop_owned(self.monitor)
                self.monitor_log.close()
            self.write()
            atomic_json(self.root/'status.json', dict(state=self.manifest['state'], ended=time.time(), owned_workers_exited=True))
            if self.manifest['state'] == 'complete':
                atomic_json(self.root/'completion.json', dict(completed=time.time(), stages=self.manifest['stages'],
                    owned_workers_exited=True, human_analysis_pending=True, cleanup_pending=True, git_push_pending=True))


def verify(root, output):
    manifest = json.loads((root/'manifest.json').read_text())
    session = json.loads((root/'session.json').read_text())
    for name, info in session['protected'].items():
        if sha256(name) != info['sha256']:
            raise ValueError('Protected artifact changed: '+name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plotting
    figure, axes = plotting.subplots(1, 2, figsize=(11, 4))
    for group in GROUPS:
        directory = Path(manifest['stages'][group+'_formal']['output'])
        train = [json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
        validation = json.loads((directory/'validation.json').read_text())
        axes[0].plot([row['updates'] for row in train], [row['loss'] for row in train], label=group, alpha=.6)
        axes[1].plot([row['updates'] for row in validation], [row['metrics']['selection_mse'] for row in validation], label=group)
    for axis, title in zip(axes, ('Native-command diffusion training loss', 'Validation EMA sampled prefix12 MSE')):
        axis.set_title(title)
        axis.set_xlabel('Optimizer updates')
        axis.legend()
    figure.tight_layout()
    figure.savefig(output/'training_curves.png', dpi=180)
    figure.savefig(output/'training_curves.svg')
    plotting.close(figure)
    videos = list(root.glob('stages/oracle_instrumented/attempt_*/*.mp4'))
    for group in GROUPS:
        stage = manifest['stages'][group+'_formal_evaluation']
        result = json.loads((Path(stage['output'])/'result.json').read_text())
        if len(result['trials']) != 20:
            raise ValueError('Formal cohort must contain20 trials per model')
        videos.append(Path(result['trials'][0]['video']))
    if len(videos) != 4:
        raise ValueError('Exactly four main videos required')
    reports = []
    for video in videos:
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
            '-of', 'json', str(video)], text=True, timeout=30))
        stream = probe['streams'][0]
        if (stream['codec_name'], stream['pix_fmt'], stream['r_frame_rate']) != ('h264', 'yuv420p', '60/1'):
            raise ValueError('Video encoding contract mismatch')
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True, timeout=120)
        reports.append(dict(path=str(video.resolve()), bytes=video.stat().st_size, sha256=sha256(video), metadata=probe))
    artifacts = [dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path))
                 for path in root.rglob('*') if path.is_file() and path.name not in ('gpu.lock', 'runner.lock')]
    atomic_json(output/'artifact_manifest.json', dict(artifacts=artifacts, videos=reports, public_upload=False))
    atomic_json(output/'result.json', dict(passed=True, videos=4, formal_trials=60, protected_hashes_unchanged=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('initialize', 'run', 'resume', 'status', 'stop', 'worker'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--task', choices=('data', 'verify'))
    parser.add_argument('--review-pre-freeze-source-update')
    args = parser.parse_args()
    args.root = args.root.resolve()
    os.environ.setdefault('MALLOC_ARENA_MAX', '2')
    if args.command == 'initialize':
        initialize(args.root)
    elif args.command == 'status':
        print((args.root/'status.json').read_text() if (args.root/'status.json').exists() else (args.root/'manifest.json').read_text())
    elif args.command == 'stop':
        atomic_json(args.root/'STOP', dict(requested=time.time()))
    elif args.command == 'worker':
        args.output.mkdir(parents=True, exist_ok=False)
        if args.task == 'data':
            prepare(args.root)
            atomic_json(args.output/'result.json', json.loads((args.root/'data_gate.json').read_text()))
        else:
            verify(args.root, args.output)
    else:
        with (args.root/'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == 'resume':
                if args.review_pre_freeze_source_update:
                    review_pre_freeze_update(args.root, args.review_pre_freeze_source_update)
                (args.root/'STOP').unlink(missing_ok=True)
                if (args.root/'blocked.json').exists():
                    (args.root/'blocked.json').rename(args.root/('blocked_'+str(time.time_ns())+'.json'))
            pipeline = Pipeline(args.root, resume=args.command == 'resume')
            pipeline.run()
            if pipeline.manifest['state'] != 'complete':
                raise SystemExit(2)


if __name__ == '__main__':
    main()
