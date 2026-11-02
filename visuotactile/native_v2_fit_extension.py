"""One preregistered BC budget sensitivity; same inputs/task, original deadline."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

from .compact import sha256
from .models.vendor.ema_model import EMAModel
from .native_v2_bc_compare import ActorBC, ACTOR_CONTRACT, actor_records, evaluate, features
from .native_v2_gates import require_training_foundation
from .native_v2_resources import LeaseV2
from .native_v2_state import BC_CONTRACT, StateBC, StateWindows
from .native_v2_training import next_batch
from .train_long import cpu_snapshot, save
from .training_resources import atomic_json


def initialize(root, parent):
    original = json.loads((parent/'session.json').read_text())
    if time.time() >= original['deadline']:
        raise ValueError('Original deadline passed')
    root.mkdir(parents=True, exist_ok=False)
    (root/'gates').mkdir()
    for name in ('G0-A', 'G0-B', 'G0'):
        gate = json.loads((parent/'gates'/(name+'.json')).read_text())
        atomic_json(root/'gates'/(name+'.json'), gate)
    session = dict(protocol_version=3, parent_root=str(parent.resolve()),
        created=time.time(), started_epoch=original['started_epoch'], deadline=original['deadline'],
        training_deadline=original['deadline'], source_root=original['source_root'],
        bc_train_ids=original['bc_train_ids'], bc_validation_ids=original['bc_validation_ids'],
        previous_updates=1000, additional_updates=3000, final_updates=4000,
        endpoint_train_arm_max=.05, endpoint_train_hand_max=.08, maximum_additional_starts=8,
        input_checkpoints={kind: dict(path=str((parent/'bc_compare'/kind/'milestone_1000.pt').resolve()),
            sha256=sha256(parent/'bc_compare'/kind/'milestone_1000.pt')) for kind in ('state', 'actor')},
        source_sha256={str(path): sha256(path) for base in ('visuotactile', 'isaacsimenvs/tasks/play')
                       for path in Path(base).rglob('*.py')},
        reason='1000-update BC train error still substantial; single budget-only fitting sensitivity',
        optimizer='Restore AdamW moments; new explicitly separate warmup500/cosine3000 LR segment',
        test_calls=0, automatic_trajectory_retries=0, automatic_further_training=False)
    atomic_json(root/'session.json', session)
    atomic_json(root/'status.json', dict(state='registered', utc=time.time()))


def recover_preaction(root):
    session = json.loads((root/'session.json').read_text())
    attempt = root/'saved_initials'
    lifecycle = json.loads((attempt/'kit_lifecycle.json').read_text())
    if (time.time() >= session['deadline'] or session.get('preaction_recoveries', 0) or
            lifecycle.get('phase') != 'initializing' or any(path.is_dir() for path in attempt.iterdir())):
        raise ValueError('Recovery requires first pre-environment failure inside original deadline')
    if not json.loads((root/'owner_exit.json').read_text())['owned_children_exited']:
        raise ValueError('Prior owner must have exited')
    learned = json.loads((root/'training/result.json').read_text())
    if not learned['software_complete'] or not learned['saved_initial_rollout_eligible']:
        raise ValueError('Completed fitting evidence required')
    for entry in learned['models'].values():
        if sha256(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Completed fitting checkpoint changed')
    archive = root/'preaction_failure_1'; archive.mkdir(exist_ok=False)
    for name in ('saved_initials', 'saved_initials.log', 'status.json', 'owner_exit.json', 'STOP_MONITOR'):
        path = root/name
        if path.exists():
            path.rename(archive/name)
    atomic_json(archive/'registered_session.json', session)
    session['preaction_recoveries'] = 1
    session['completed_training_sha256'] = sha256(root/'training/result.json')
    session['source_sha256'] = {name: sha256(name) for name in session['source_sha256']}
    atomic_json(root/'session.json', session)
    atomic_json(root/'recovery.json', dict(utc=time.time(), prior_exit=-11,
        reason='OmniClient FileThread startup SIGSEGV before environment or trajectory; one reviewed recovery',
        deadline=session['deadline'], optimizer_updates_to_repeat=0, trajectories_to_repeat=0))


def train(root, output):
    require_training_foundation(root)
    session = json.loads((root/'session.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    source = Path(session['source_root'])
    windows = {split: StateWindows(source, session['bc_'+split+'_ids']) for split in ('train', 'validation')}
    records = actor_records(source, session['bc_train_ids']+session['bc_validation_ids'])
    lease = LeaseV2(root, deadline=session['deadline'])
    results = {}
    for kind in ('state', 'actor'):
        entry = session['input_checkpoints'][kind]
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('Continuation checkpoint changed')
        saved = torch.load(entry['path'], map_location='cpu', weights_only=False)
        if saved['updates'] != session['previous_updates']:
            raise ValueError('Continuation update count mismatch')
        directory = output/kind
        directory.mkdir()
        with lease.block():
            model = (StateBC() if kind == 'state' else ActorBC()).cuda()
            model.load_state_dict(saved['model'])
            averaged = copy.deepcopy(model).eval().requires_grad_(False)
            averaged.load_state_dict(saved['ema'])
            ema = EMAModel(averaged, power=.75)
            ema.optimization_step = saved['updates']
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['cpu_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'])
        values = {split: torch.cat((window.robot, window.geometry), -1) if kind == 'state' else
                  features(window, records, model.actor_mean.cpu(), model.actor_std.cpu())
                  for split, window in windows.items()}
        epoch = cursor = 0
        for _ in range(session['previous_updates']):
            _, epoch, cursor = next_batch(len(windows['train']), epoch, cursor, 32)
        history = []
        for step in range(1, session['additional_updates']+1):
            indices, epoch, cursor = next_batch(len(windows['train']), epoch, cursor, 32)
            with lease.block():
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = model.loss(values['train'][indices].cuda(), windows['train'].command[indices].cuda(),
                                  windows['train'].valid[indices].cuda())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite continuation loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite continuation gradient')
                scale = min(1., step/500)*.5*(1+math.cos(math.pi*max(0, step-500)/2500))
                for parameter_group in optimizer.param_groups:
                    parameter_group['lr'] = 1e-4*scale
                optimizer.step(); ema.step(model)
            update = saved['updates']+step
            with (directory/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(updates=update, loss=float(loss.detach()), gradient_norm=float(norm),
                                             lr=1e-4*scale))+'\n')
            if step in (1000, session['additional_updates']):
                row = dict(updates=update, **{split: evaluate(averaged, values[split], windows[split], lease)
                                               for split in ('train', 'validation')})
                history.append(row)
                atomic_json(directory/'metrics.json', history)
                payload = cpu_snapshot(dict(contract=BC_CONTRACT if kind == 'state' else ACTOR_CONTRACT,
                    group='S-NativeDP', diagnostic_only=True, updates=update, epoch=epoch, cursor=cursor,
                    model=model.state_dict(), ema=averaged.state_dict(), optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    ema_optimization_step=ema.optimization_step, continuation_from=entry,
                    train_ids=session['bc_train_ids'], validation_ids=session['bc_validation_ids']))
                save(directory/f'milestone_{update}.pt', payload)
        results[kind] = dict(history=history, checkpoint=str((directory/'milestone_4000.pt').resolve()),
                             checkpoint_sha256=sha256(directory/'milestone_4000.pt'))
    state = results['state']['history'][-1]['train']
    eligible = state['arm_rmse'] <= session['endpoint_train_arm_max'] and state['hand_rmse'] <= session['endpoint_train_hand_max']
    atomic_json(output/'result.json', dict(software_complete=True, models=results,
        saved_initial_rollout_eligible=eligible, native_success_proven=False, test_used=False))


class Owner:
    def __init__(self, root):
        self.root = root
        self.session = json.loads((root/'session.json').read_text())
        self.worker = self.monitor = None
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))

    def check(self):
        if self.stop or (self.root/'STOP').exists():
            raise InterruptedError('Owned extension stop')
        if time.time() >= self.session['deadline']:
            raise TimeoutError('Original repair deadline reached; no reset')
        if self.monitor is not None and self.monitor.poll() is not None:
            raise RuntimeError('Supplement resource monitor exited')

    def stage(self, name, module, extra=()):
        self.check()
        for path, digest in self.session['source_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Registered extension source changed: '+path)
        output = self.root/name
        if output.exists():
            raise ValueError('No repeat of an attempted extension stage')
        command = ['bash', 'visuotactile/run_cpu.sh', '-m', module, '--root', str(self.root),
                   '--output', str(output), *extra]
        with (self.root/(name+'.log')).open('w') as log:
            self.worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            atomic_json(self.root/'status.json', dict(state='running', stage=name, pid=os.getpid(),
                                                    worker_pid=self.worker.pid, utc=time.time()))
            while self.worker.poll() is None:
                self.check(); time.sleep(2)
            code = self.worker.returncode; self.worker = None
        if code != 0 or not (output/'result.json').exists():
            raise RuntimeError(f'{name} failed exit {code}; no automatic retry')
        result = json.loads((output/'result.json').read_text())
        if not result.get('software_complete'):
            raise ValueError('Incomplete extension stage')
        return result

    def run(self):
        try:
            parent = Path(self.session['parent_root'])
            atomic_json(self.root/'status.json', dict(state='waiting_original_cohort', pid=os.getpid(), utc=time.time()))
            while not (parent/'owner_exit.json').exists():
                self.check(); time.sleep(2)
            if json.loads((parent/'status.json').read_text())['state'] != 'diagnostics_complete':
                raise ValueError('Original cohort did not complete; no automatic extension')
            (self.root/'resources').mkdir(exist_ok=True)
            if not (self.root/'resources/resources.jsonl').exists():
                (self.root/'resources/resources.jsonl').write_bytes((parent/'resources/resources.jsonl').read_bytes())
            with (self.root/'monitor.log').open('w') as log:
                self.monitor = subprocess.Popen(['bash', 'visuotactile/run_cpu.sh', '-m',
                    'visuotactile.native_resources', '--root', str(self.root)], stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            if self.session.get('preaction_recoveries'):
                if sha256(self.root/'training/result.json') != self.session['completed_training_sha256']:
                    raise ValueError('Preserved fitting evidence changed')
                learned = json.loads((self.root/'training/result.json').read_text())
            else:
                learned = self.stage('training', 'visuotactile.native_v2_fit_extension', ['--train'])
            if learned['saved_initial_rollout_eligible']:
                checkpoint = Path(learned['models']['state']['checkpoint'])
                trials = [dict(name=f'{cohort}_{ordinal}', kind='student', identity=identity, cohort=cohort,
                    checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), seed=7091, video=ordinal == 0)
                    for cohort, ids in [('train', self.session['bc_train_ids']), ('validation', self.session['bc_validation_ids'])]
                    for ordinal, identity in enumerate(ids)]
                job = self.root/'job.json'; atomic_json(job, dict(trials=trials))
                self.stage('saved_initials', 'visuotactile.native_v2_evaluation', ['--job', str(job)])
            atomic_json(self.root/'status.json', dict(state='diagnostics_complete', utc=time.time(),
                rollout_eligible=learned['saved_initial_rollout_eligible'], automatic_further_training=False))
        except BaseException as error:
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
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--recover-preaction', action='store_true')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if arguments.recover_preaction:
        recover_preaction(arguments.root)
    elif arguments.parent:
        initialize(arguments.root.resolve(), arguments.parent.resolve())
    elif arguments.train:
        train(arguments.root, arguments.output)
    else:
        with (arguments.root/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            Owner(arguments.root.resolve()).run()
