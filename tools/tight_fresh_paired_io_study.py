"""Finite engineering-only A/B I/O study, using the unchanged full-episode trainer.

Preparation is CPU-only. Running is explicit, sequential plain then paired,
without prefetch, deployment, automatic retries, or session deadline changes.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.paired_episode_io import configure_pairing, process_identity, read, shared_inputs, shard_identity
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_cycle import paired_frontends, parallel_admission
from tools.tight_fresh_eval_recovery import stop_owned

VERSION = 'paired_episode_io_engineering_study_v1'
ARMS = ('plain', 'paired')
COUNTS = dict(train=93, validation=6, test=0)


def code_identity():
    source = Path(__file__).resolve().parent.parent
    return dict(root=str(source), commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
        sha256={str(p.relative_to(source)): sha256(p) for folder in ('tools', 'visuotactile')
                for p in sorted((source / folder).rglob('*.py'))})


def bind_file(path, *, payload_hash=True):
    path = Path(path).resolve()
    return dict(path=str(path), stat=shard_identity(path), sha256=sha256(path) if payload_hash else None)


def source_inputs(source):
    """Bind original continuation, numeric inputs and existing audited frozen cache."""
    source = Path(source).resolve()
    session = read(source / 'session.json')
    if session.get('resource_policy') != 'full_gpu_v1' or not read(source / 'gate.json').get('passed'):
        raise ValueError('Completed input gate and full-GPU session contract required')
    shared = shared_inputs(source)
    if len(shared['population']['ids']) != 93 or sum(shared['population']['multiplicities'].values()) != 102:
        raise ValueError('This engineering study requires the original round06 93/102 population')
    files = [bind_file(source / name) for name in ('session.json', 'metadata.json', 'gate.json')]
    all_shards = {}; continuation = {}; common_splits = None
    for role in ('A', 'B'):
        directory = source / role; data = read(directory / 'state_data/data.json')
        index = read(directory / 'cache_index.json'); spec = read(directory / 'continuation.json')
        if ({key: len(value) for key, value in data['splits'].items()} != COUNTS
                or (common_splits is not None and data['splits'] != common_splits)):
            raise ValueError('Preserve exactly the round06 train93/validation6/test0 split')
        common_splits = data['splits']
        if (spec.get('sampling_contract') != dict(version='latest_takeover_repeat_v1', latest_repeat=2)
                or Path(spec['path']).parent.parent.parent.name != 'round_05'
                or spec.get('target_window_contract') != 'recovery_onset_commit12_v1'):
            raise ValueError('Keep original round05 continuation and latest-repeat2 target contract')
        checkpoint = bind_file(spec['path'])
        if checkpoint['sha256'] != spec['sha256']:
            raise ValueError('Original round05 starting checkpoint changed')
        files.append(checkpoint)
        continuation[role] = dict(path=spec['path'], sha256=spec['sha256'],
            statistics_sha256=spec['statistics_sha256'])
        for name in ('continuation.json', 'cache_index.json', 'state_data/data.json',
                     'state_data/numeric.pt', 'state_data/statistics.pt', 'state_data/loss_context.pt'):
            path = directory / name
            if path.exists(): files.append(bind_file(path))
            elif not name.endswith('loss_context.pt'): raise FileNotFoundError(path)
        if sha256(directory / 'state_data/statistics.pt') != spec['statistics_sha256']:
            raise ValueError('Original train-only statistics changed')
        for identity in data['splits']['train'] + data['splits']['validation']:
            entry = index['entries'][identity]
            actual = shard_identity(directory / 'state_data/perception' / (identity + '.pt'))
            if actual != shard_identity(entry['path']): raise ValueError('Cache path identity changed')
            bound = dict(stat=actual, sha256=entry['sha256'], source_sha256=entry['source_sha256'],
                         frames=entry['frames'], camera_time_sha256=entry['camera_time_sha256'])
            if identity in all_shards and bound != all_shards[identity]:
                raise ValueError('A/B must share identical frozen train and validation shards')
            all_shards[identity] = bound
    return dict(source=str(source), files=files, shards=all_shards, shared_train=shared,
                splits=common_splits, continuation=continuation, session_deadline=session['deadline'])


def prepare(source, output, *, max_seconds=720):
    source, output = Path(source).resolve(), Path(output).resolve()
    if type(max_seconds) is not int or not 60 <= max_seconds <= 720:
        raise ValueError('Total study bound must be 60–720 seconds including cleanup reserve')
    if output.exists(): raise FileExistsError('Preserve previous study identity')
    bound = source_inputs(source)
    if time.time() >= bound['session_deadline']: raise TimeoutError('Original session expired')
    code = code_identity(); output.mkdir(parents=True)
    plan = dict(version=VERSION, source_inputs=bound, code=code, arms=list(ARMS), epochs_per_role=1,
        visits_per_role=102, aggregate_visits_per_arm=204, prefetch=False, max_seconds=max_seconds,
        cleanup_reserve_seconds=20, session_deadline=bound['session_deadline'], prepared_utc=time.time(),
        scope='Engineering diagnostic only; no deployment, autonomy result or formal model selection. '
              'Plain first, paired second; OS page-cache warmth and arm order confound causal speed claims. '
              'Existing SHA-audited frozen shards are stat-bound; no payload prewarming, cache dropping or retries.')
    atomic_json(output / 'plan.json', plan)
    try:
        for arm in ARMS:
            root = output / arm; root.mkdir()
            for name in ('session.json', 'metadata.json', 'gate.json'):
                shutil.copyfile(source / name, root / name)
            (root / 'shared_resources.json').symlink_to((source / 'shared_resources.json').resolve())
            for role in ('A', 'B'):
                directory = root / role; directory.mkdir()
                for name in ('continuation.json', 'cache_index.json'):
                    shutil.copyfile(source / role / name, directory / name)
                # Immutable input directory is shared. Trainer writes only its new training/ directory.
                (directory / 'state_data').symlink_to((source / role / 'state_data').resolve(), target_is_directory=True)
            if arm == 'paired': configure_pairing(root, 1)
            atomic_json(root / 'engineering_diagnostic.json', dict(version=VERSION, arm=arm,
                plan_sha256=sha256(output / 'plan.json'), deployable=False, prefetch=False,
                epochs=1, visits_per_role=102, source_root=str(source)))
        arm_files = {str(path.relative_to(output)): sha256(path) for arm in ARMS
            for path in (list((output / arm).glob('*.json')) + list((output / arm).glob('[AB]/*.json')))
            if path.name != 'shared_resources.json'}
        atomic_json(output / 'prepared.json', dict(complete=True, plan_sha256=sha256(output / 'plan.json'), arm_files=arm_files,
            gpu_started=False, cache_payload_preheated=False, utc=time.time()))
    except BaseException as error:
        atomic_json(output / 'prepare_failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    return plan


def verify(output, plan):
    prepared = read(output / 'prepared.json')
    if (plan['version'] != VERSION or plan['arms'] != list(ARMS) or plan['prefetch'] is not False
            or plan['epochs_per_role'] != 1 or plan['visits_per_role'] != 102
            or prepared['plan_sha256'] != sha256(output / 'plan.json')):
        raise ValueError('Study registration changed')
    if any(sha256(output / path) != expected for path, expected in prepared['arm_files'].items()):
        raise ValueError('Prepared arm metadata/configuration changed')
    if code_identity() != plan['code']: raise ValueError('Study frozen source changed')
    for file in plan['source_inputs']['files']:
        if shard_identity(file['path']) != file['stat']:
            raise ValueError('Bound immutable source file changed: ' + file['path'])
    for arm in ARMS:
        root = output / arm
        if read(root / 'session.json') != read(Path(plan['source_inputs']['source']) / 'session.json'):
            raise ValueError('Original session/deadline changed')
        if (root / 'paired_episode_io.json').exists() != (arm == 'paired'):
            raise ValueError('Arm synchronization identity changed')
        if shared_inputs(root) != plan['source_inputs']['shared_train']:
            raise ValueError('Arm frozen training input binding changed')
        for role in ('A', 'B'):
            if (root / role / 'state_data').resolve() != (Path(plan['source_inputs']['source']) / role / 'state_data').resolve():
                raise ValueError('Arm immutable numeric/visual input directory changed')
            if (root / role / 'episode_io.json').exists(): raise ValueError('Both arms must disable prefetch')
            for identity, shard in plan['source_inputs']['shards'].items():
                if shard_identity(root / role / 'state_data/perception' / (identity + '.pt')) != shard['stat']:
                    raise ValueError('Bound validation or training shard changed')


def proc_sample(pid):
    try:
        status = dict(line.split(':', 1) for line in Path('/proc', str(pid), 'status').read_text().splitlines() if ':' in line)
        io = dict(line.split(':', 1) for line in Path('/proc', str(pid), 'io').read_text().splitlines())
        return dict(utc=time.time(), rss_bytes=int(status['VmRSS'].split()[0])*1024,
                    read_bytes=int(io['read_bytes']), rchar=int(io['rchar']))
    except (OSError, KeyError, ValueError): return None


def run_worker(output, arm, name, args, completion, deadline, *, update_status=False):
    ledger = output / 'owner' / arm; ledger.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m'] + list(map(str, args)); began = time.time(); reason = None
    process = None; samples = 0; last = None; peak_rss = 0
    receipt = dict(stage=name, command=command, started_utc=began, completion=str(completion))
    try:
        with (ledger / (name + '.log')).open('x') as stream:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent.parent,
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONFAULTHANDLER='1'))
            identity = process_identity(process.pid)
            receipt.update(pid=process.pid, proc_start_ticks=identity['proc_start_ticks'] if identity else None)
            atomic_json(ledger / (name + '_worker.json'), dict(receipt, deadline=deadline))
            while process.poll() is None:
                if time.time() >= deadline: reason = 'bounded_study_training_cutoff'; break
                if (output / 'STOP').exists(): reason = 'owned_study_stop'; break
                sample = proc_sample(process.pid)
                if sample:
                    last = sample; samples += 1; peak_rss = max(peak_rss, sample['rss_bytes'])
                    atomic_json(ledger / (name + '_progress.json'), dict(sample, pid=process.pid))
                time.sleep(min(.25, max(.001, deadline-time.time())))
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        if process is not None and process.poll() is None: stop_owned(process)
        receipt.update(returncode=process.returncode if process else None, stop_reason=reason,
            seconds=time.time()-began, ended_utc=time.time(), last_process_sample=last, samples=samples,
            peak_sampled_rss_bytes=peak_rss, completion_sha256=sha256(completion) if completion.exists() else None,
            proc_io_scope='Last live sample, not exact final process I/O; exact training epoch deltas are reported separately')
        atomic_json(ledger / (name + '.json'), receipt)
    if reason or process.returncode or not completion.exists() or not read(completion).get('complete'):
        raise RuntimeError('Study worker did not complete: ' + name)
    return receipt


def arm_summary(output, arm, plan):
    import torch
    from visuotactile.latest_takeover_sampling import episode_deck
    population = plan['source_inputs']['shared_train']['population']
    expected = episode_deck(population['ids'], population['multiplicities'], torch.Generator().manual_seed(2026111402))
    roles = {}
    for role in ('A', 'B'):
        directory = output / arm / role / 'training'
        finished = read(directory / 'finished.json')
        rows = [json.loads(line) for line in (directory / 'learning.jsonl').read_text().splitlines()]
        validation = read(directory / 'validation.json'); epoch = validation['training_epoch_io']
        if (finished.get('epochs') != 1 or finished.get('updates') != 102
                or [row['episode'] for row in rows] != expected
                or [row['updates'] for row in rows] != list(range(1, 103))
                or finished.get('sampling_exposure') != population['multiplicities']
                or not epoch['ended_utc'] >= epoch['started_utc']):
            raise ValueError('Actual full-epoch exposure/order differs from registered study')
        # Do not sweep multi-GB new checkpoints between arms and perturb the next cache state.
        checkpoints = {name: bind_file(directory / name, payload_hash=False) for name in ('last.pt', 'best.pt')}
        if any(file['stat']['bytes'] <= 0 for file in checkpoints.values()):
            raise ValueError('Missing complete study checkpoint artifact')
        checkpoints['best.pt']['trainer_reported_sha256'] = finished['checkpoint_sha256']
        roles[role] = dict(updates=finished['updates'], anchors=finished['anchors_seen'],
            target_ticks=finished['target_ticks'], sampling_exposure=finished['sampling_exposure'],
            training_epoch_io=epoch, checkpoint_files=checkpoints,
            finished_sha256=sha256(directory / 'finished.json'),
            learning_sha256=sha256(directory / 'learning.jsonl'), paired_waits=finished.get('paired_episode_io'))
    elapsed = max(v['training_epoch_io']['ended_utc'] for v in roles.values()) - min(
        v['training_epoch_io']['started_utc'] for v in roles.values())
    return dict(arm=arm, roles=roles, aggregate_updates=204, joint_epoch_wall_seconds=elapsed,
        aggregate_updates_per_second=204/elapsed,
        summed_process_training_read_bytes=sum(v['training_epoch_io']['process_io']['read_bytes'] for v in roles.values()),
        scope='Two independent complete train93/latest-repeat2 episodes decks; epoch interval includes pair waits. '
              'Process read_bytes are kernel-accounted worker deltas, not unique device I/O or causal attribution.')


def _run(output):
    output = Path(output).resolve(); plan = read(output / 'plan.json'); began = time.time()
    end = min(plan['session_deadline'], began + plan['max_seconds'])
    cutoff = end - plan['cleanup_reserve_seconds']
    # Exclusive durable owner claim: no automatic skip/retry, including a prior failed owner.
    with (output / 'run_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), proc_start_ticks=process_identity(os.getpid())['proc_start_ticks'],
                       started_utc=began, bounded_deadline=end,
                       training_cutoff=cutoff, original_session_deadline=plan['session_deadline']), stream)
    results = {}
    try:
        verify(output, plan)
        if (output / 'STOP').exists(): raise InterruptedError('Owned study STOP already requested')
        if time.time() >= cutoff: raise TimeoutError('Insufficient original session time')
        if any((output / arm / role / 'training').exists() for arm in ARMS for role in ('A', 'B')):
            raise ValueError('Do not resume or overwrite an already-started study')
        for arm in ARMS:
            verify(output, plan)
            if (output / 'STOP').exists(): raise InterruptedError('Owned study STOP before next arm')
            admission = parallel_admission(read(output / arm / 'shared_resources.json'), time.time())
            atomic_json(output / (arm + '_admission.json'), admission)
            if not admission['admitted']: raise RuntimeError('Joint frontend resource admission failed')
            if time.time() >= cutoff: raise TimeoutError('Bounded study cutoff before next arm')
            atomic_json(output / 'status.json', dict(stage=arm, utc=time.time(), deadline=end))
            arm_began = time.time()
            def execute(name, args, completion, *, update_status=False):
                return run_worker(output, arm, name, args, completion, cutoff, update_status=update_status)
            receipt = paired_frontends(execute, output / arm, 1)
            atomic_json(output / (arm + '_workers.json'), receipt)
            if not receipt['complete']: raise RuntimeError('Study arm failed: ' + str(receipt['errors']))
            result = arm_summary(output, arm, plan)
            full_arm_wall = time.time()-arm_began
            result.update(full_arm_wall_seconds=full_arm_wall,
                          full_arm_aggregate_updates_per_second=204/full_arm_wall)
            results[arm] = result; atomic_json(output / (arm + '_result.json'), result)
        # Both arms must expose exactly the same legal actions, irrespective of any numerical CUDA differences.
        for role in ('A', 'B'):
            for key in ('updates', 'anchors', 'target_ticks', 'sampling_exposure'):
                if results['plain']['roles'][role][key] != results['paired']['roles'][role][key]:
                    raise ValueError('Study arm supervision exposure differs')
        atomic_json(output / 'result.json', dict(software_complete=True, deployable=False, arms=results,
            seconds=time.time()-began, paired_over_plain_epoch_throughput=results['paired']['aggregate_updates_per_second'] /
                results['plain']['aggregate_updates_per_second'],
            interpretation='One sequential plain→paired engineering comparison. Page-cache warmth/order are uncontrolled; '
                'observed ratio alone does not establish causal acceleration or autonomous improvement.'))
        atomic_json(output / 'status.json', dict(stage='complete', utc=time.time(), deployable=False))
    except BaseException as error:
        atomic_json(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            completed_arms=list(results), seconds=time.time()-began, original_deadline_unchanged=True,
            scope='All completed checkpoints and partial episode logs retained; no automatic resume/retry'))
        atomic_json(output / 'status.json', dict(stage='failed', error=repr(error), utc=time.time()))
        raise


def run(output):
    output = Path(output).resolve()
    previous = {}
    def request_stop(signum, frame):
        (output / 'STOP').touch(exist_ok=True)
        atomic_json(output / 'stop_requested.json', dict(signal=signum, utc=time.time(),
            scope='Only this study owner and its children; no automatic resume'))
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        return _run(output)
    finally:
        for signum, handler in previous.items(): signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=720)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.source is None: parser.error('Preparation requires --source round_06')
        prepare(args.source, args.output, max_seconds=args.max_seconds)
    else: run(args.output)


if __name__ == '__main__': main()
