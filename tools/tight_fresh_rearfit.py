"""One explicitly selected backend fitting experiment with a frozen complete DP.

Keep actual caches, data, statistics and original deadline; train both roles then
run the registered complete students. The agent chooses whether to invoke it.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read, completed_pair


def prepare(root, output, epochs):
    if output.exists():
        raise FileExistsError('Preserve previous fitting experiment')
    pairs = {role: completed_pair(root, role) for role in ('A', 'B')}
    if epochs < 1 or epochs > 100:
        raise ValueError('Finite positive paired exposure required')
    for role in pairs:
        data = read(root / role / 'state_data/data.json')
        if not data.get('aggregation'):
            raise ValueError('This continuation requires audited aggregated masks and cache bindings')
    output.mkdir(parents=True)
    for name in ('session.json', 'metadata.json', 'gate.json'):
        shutil.copyfile(root / name, output / name)
    (output / 'shared_resources.json').symlink_to(root / 'shared_resources.json')
    sources = {str(p): sha256(p) for folder in ('tools', 'visuotactile') for p in Path(folder).rglob('*.py')}
    for role, pair in pairs.items():
        destination = output / role
        destination.mkdir()
        for name in ('state_data', 'training', 'cache_index.json'):
            (destination / name).symlink_to(root / role / name, target_is_directory=name in ('state_data', 'training'))
        old = root / role / 'backend'
        rear = destination / 'backend'
        rear.mkdir()
        v = copy.deepcopy(read(old / 'configuration.json')['residual_pair'])
        data = read(root / role / 'state_data/data.json')
        v.update(maximum_control_tick_epochs=epochs, minimum_control_tick_epochs=epochs,
            validation_epoch_interval=1, comparison_control_tick_epochs=[epochs],
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256=sources,
            continuation_from=dict(path=pair['backend'], sha256=pair['backend_sha256'],
                teacher_supervision='u_expert', statistics_sha256=v['residual_statistics_sha256'],
                normalization_identity=v['residual_normalization_identity'],
                round_index=max(e.get('round_index', 0) for e in data['entries'].values()), source_group=role))
        session = copy.deepcopy(read(old / 'session.json'))
        session.update(residual_pair=v, source_head=v['source_commit'], resource_root=str(output))
        atomic_json(rear / 'session.json', session)
        atomic_json(rear / 'configuration.json', dict(residual_pair=v))
        (rear / 'state_data').symlink_to(root / role / 'state_data', target_is_directory=True)
        (rear / 'shared_resources.json').symlink_to(root / 'shared_resources.json')
        child = rear / role
        child.mkdir()
        (child / 'training').mkdir()
        (child / 'state_data').symlink_to(root / role / 'state_data', target_is_directory=True)
        (child / 'nominal').symlink_to(old / role / 'nominal', target_is_directory=True)
        atomic_json(child / 'configuration.json', dict(residual_pair=v))
        atomic_json(child / 'session.json', dict(session, resource_root=str(output), modality=role))
        index = read(child / 'nominal/index.json')
        if (index['base_sha256'] != pair['frontend_sha256'] or not index.get('supervision_mask_sha256')
                or index['data_sha256'] != sha256(child / 'state_data/data.json')
                or index['residual_statistics_sha256'] != v['residual_statistics_sha256']):
            raise ValueError('Current DP and legal supervision mask cache required')
        atomic_json(rear / 'cache_complete.json', dict(complete=True, exact_reuse=True,
            cache_index_sha256=sha256(child / 'nominal/index.json'), frontend_updates=0))
    atomic_json(output / 'prepared.json', dict(source_root=str(root), pairs=pairs,
        backend_epochs=epochs, frontend_updates=0, fresh_optimizer=True,
        source_sha256=sources, deadline=read(root / 'session.json')['deadline'], test_used=False))


def run(root, output, epochs, count, seed):
    if not 1 <= count <= 32:
        raise ValueError('Finite development cohort required before training')
    prepare(root, output, epochs)
    registration = read(output / 'prepared.json')
    deadline = registration['deadline']
    ledger = output / 'stages'
    ledger.mkdir()
    receipts = []
    def execute(name, arguments, completion):
        if time.time() >= deadline:
            raise TimeoutError('Original fresh session deadline exhausted')
        if any(sha256(p) != value for p, value in registration['source_sha256'].items()):
            raise ValueError('Active fitting source changed')
        command = [sys.executable, '-m'] + list(map(str, arguments))
        atomic_json(ledger / 'status.json', dict(stage=name, command=command, started=time.time()))
        began = time.time()
        with (ledger / (name + '.log')).open('x') as stream:
            worker = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                timeout=max(1, deadline-time.time()), env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        receipt = dict(stage=name, returncode=worker.returncode, seconds=time.time()-began,
            command=command, completion_sha256=sha256(completion) if completion.exists() else None)
        atomic_json(ledger / (name + '.json'), receipt)
        receipts.append(receipt)
        if worker.returncode or not completion.exists():
            raise RuntimeError('Fitting stage failed: ' + name)
        proof = read(completion)
        if any(proof.get(key) is False for key in ('complete', 'software_complete', 'target_reached')):
            raise RuntimeError('Incomplete fitting stage: ' + name)
    try:
        for role in ('A', 'B'):
            rear = output / role / 'backend' / role
            execute('train_' + role, ['visuotactile.anchor_continue', 'residual-train', '--root', rear,
                '--job', 'image', '--output', rear / 'training'], rear / 'training/finished.json')
            execute('offline_' + role, ['visuotactile.anchor_continue', 'residual-offline', '--root', rear,
                '--job', 'image', '--output', rear / 'offline'], rear / 'offline/result.json')
        development = output / 'development'
        execute('development_prepare', ['tools.tight_fresh_rollouts', 'prepare', '--root', output,
            '--output', development, '--count', count, '--seed', seed], development / 'plan.json')
        execute('development', ['tools.tight_fresh_rollouts', 'run', '--root', output,
            '--output', development], development / 'result.json')
        atomic_json(ledger / 'finished.json', dict(complete=True, stages=receipts,
            interpretation='Actual scientific outcome requires Agent review; no automatic next research cycle'))
    except BaseException as error:
        atomic_json(ledger / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), stages=receipts))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--seed', type=int, default=28140000)
    args = p.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.epochs, args.count, args.seed)


if __name__ == '__main__':
    main()
