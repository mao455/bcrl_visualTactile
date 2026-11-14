"""Explicit one-shot paired-only recovery of the known plain-complete I/O study.

Preparation and verification are CPU-only. The old failed study remains read-only;
run starts only the missing paired arm with the original frozen trainer.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.paired_episode_io import (configure_pairing, process_alive,
    process_identity, read, shared_inputs, shard_identity)
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_cycle import paired_frontends, parallel_admission

VERSION = 'paired_io_missing_arm_recovery_v1'
STUDY = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/'
             'tight_fresh_20261114_05/paired_io_compare01')
SOURCE_COMMIT = 'b59baf365e4de7570b3eec95e65abe1e076d87d6'
ORIGINAL_PLAN_SHA = '98b9aaffdd3a8d9a83d8e822ef9e9fee8e83caf1ba7fcfbcb89a5d60de866df6'
ORIGINAL_FAILURE_SHA = 'c7993341841a509b4be6ebb59b17aac71fb4aa38a618e0314a6d1b3ea182a449'
SESSION_DEADLINE = 1794664461.6238303
CONFOUNDS = dict(metadata_scan_local_start='2026-11-14 21:26:37 Asia/Shanghai',
    metadata_scan_local_end='2026-11-14 21:29:02 Asia/Shanghai',
    plain_metadata_scan_overlap_seconds=134.76, page_cache_warmth_uncontrolled=True,
    interpretation='Plain overlapped the read-only metadata scan; paired is a later recovery with warm OS '
        'page cache and changed I/O contention. This is not a fair causal speed comparison, '
        'formal model selection, deployment, or evidence of autonomous improvement.')


def frozen_module(plan):
    """The helper's __file__ fixes run_worker's child CWD to the original tree."""
    code = plan['code']
    if code['commit'] != SOURCE_COMMIT:
        raise ValueError('Recovery requires the original b59baf3 trainer')
    path = Path(code['root']) / 'tools/tight_fresh_paired_io_study.py'
    spec = importlib.util.spec_from_file_location('_tight_original_paired_io_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.code_identity() != code:
        raise ValueError('Original frozen trainer source changed')
    return module


def assert_no_worker(root):
    """Reject an orphan trainer even if its ordinary worker ledger is missing."""
    target = str(root.resolve())
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            args = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if target in args and 'tools.tight_fresh_train' in args:
            raise ValueError('A trainer already consumes this paired root: ' + directory.name)


def inspect_original(study):
    study = Path(study).resolve()
    if study != STUDY.resolve() or sha256(study / 'plan.json') != ORIGINAL_PLAN_SHA:
        raise ValueError('Only the explicitly identified failed study is recoverable')
    if sha256(study / 'failure.json') != ORIGINAL_FAILURE_SHA:
        raise ValueError('Original failure identity changed')
    plan = read(study / 'plan.json'); failure = read(study / 'failure.json')
    admission = read(study / 'paired_admission.json'); owner = read(study / 'run_started.json')
    if (failure.get('completed_arms') != ['plain']
            or failure.get('error') != "RuntimeError('Joint frontend resource admission failed')"
            or admission.get('admitted') is not False
            or admission.get('observed_gpu_free_mib') != 16389.0
            or admission.get('required_free_gpu_mib') != 22*1024
            or admission.get('required_available_ram_bytes') != 24*1024**3
            or plan['session_deadline'] != SESSION_DEADLINE
            or owner['original_session_deadline'] != SESSION_DEADLINE):
        raise ValueError('Not the diagnosed pre-paired resource-admission failure')
    if process_alive(owner):
        raise ValueError('Original study owner has not exited')
    if ((study / 'result.json').exists() or (study / 'paired_result.json').exists()
            or (study / 'owner/paired').exists()
            or any((study / 'paired' / role / 'training').exists() for role in ('A', 'B'))
            or any((study / 'paired/paired_episode_io').iterdir())):
        raise ValueError('The old paired arm has already started; never resume or duplicate it')
    assert_no_worker(study / 'paired')
    worker_receipts = []
    for role in ('A', 'B'):
        worker = read(study / 'owner/plain' / ('frontend_' + role + '.json'))
        if (process_alive(worker) or worker.get('returncode') != 0 or worker.get('stop_reason')
                or worker.get('completion_sha256') != sha256(study / 'plain' / role / 'training/finished.json')):
            raise ValueError('Original plain worker is live or incomplete')
        worker_receipts.append(worker)
    workers = read(study / 'plain_workers.json')
    if not workers.get('complete') or workers.get('errors'):
        raise ValueError('Both plain roles must already be complete')
    helper = frozen_module(plan)
    helper.verify(study, plan)
    actual = helper.arm_summary(study, 'plain', plan)
    saved = read(study / 'plain_result.json')
    if any(saved.get(key) != value for key, value in actual.items()):
        raise ValueError('Saved plain result does not match its actual 102-visit artifacts')
    if (plan['epochs_per_role'] != 1 or plan['visits_per_role'] != 102
            or plan['aggregate_visits_per_arm'] != 204 or plan['prefetch'] is not False
            or len(plan['source_inputs']['shared_train']['population']['ids']) != 93):
        raise ValueError('Preserve the original 93-episode/102-visit budget')
    names = ['plan.json', 'prepared.json', 'failure.json', 'run_started.json', 'plain_result.json',
             'plain_workers.json', 'paired_admission.json', 'paired/paired_episode_io.json']
    names += ['owner/plain/frontend_' + role + suffix for role in ('A', 'B')
              for suffix in ('.json', '_worker.json')]
    names += ['plain/' + role + '/training/' + name for role in ('A', 'B')
              for name in ('finished.json', 'learning.jsonl', 'validation.json')]
    bindings = {name: sha256(study / name) for name in names}
    return dict(plan=plan, plain=saved, bindings=bindings, owner=owner,
                plain_workers_exited_utc=max(worker['ended_utc'] for worker in worker_receipts)), helper


def pairing_equivalent(left, right):
    # Only synchronization identity may differ; inputs, RNG deck and all timing contracts stay pinned.
    ignored = {'attempt'}
    return {k: v for k, v in left.items() if k not in ignored} == {
        k: v for k, v in right.items() if k not in ignored}


def prepare(study, output, *, max_seconds=360):
    study, output = Path(study).resolve(), Path(output).resolve()
    if type(max_seconds) is not int or not 60 <= max_seconds <= 360:
        raise ValueError('Independent recovery window must be 60–360 seconds including admission and cleanup')
    if output.exists():
        raise FileExistsError('Preserve the prior recovery identity')
    if output.parent != study.parent:
        raise ValueError('Recovery must be a fresh sibling of the failed study')
    original, _ = inspect_original(study)
    if time.time() >= SESSION_DEADLINE - 20:
        raise TimeoutError('Original session deadline leaves no recovery time')
    # One claim for this exact missing arm, also across differently named new output directories.
    claim = study.parent / ('paired_io_recovery_' + ORIGINAL_PLAN_SHA[:16] + '.claim.json')
    with claim.open('x') as stream:
        json.dump(dict(version=VERSION, study=str(study), output=str(output), utc=time.time(),
                       original_plan_sha256=ORIGINAL_PLAN_SHA, automatic_retry=False), stream)
        stream.flush(); os.fsync(stream.fileno())
    output.mkdir()
    plan = dict(version=VERSION, study=str(study), original=original, claim=str(claim),
        claim_sha256=sha256(claim), max_seconds=max_seconds, admission_max_seconds=30,
        cleanup_reserve_seconds=20, session_deadline=SESSION_DEADLINE, arms=['paired'],
        epochs_per_role=1, visits_per_role=102, aggregate_visits_per_arm=204, prefetch=False,
        recovery_code_sha256=sha256(__file__), prepared_utc=time.time(), confounds=CONFOUNDS,
        deadline_scope='New explicitly authorized recovery window, capped by original session05; '
            'the old study remains failed with its original cutoff and deadline unchanged.')
    atomic_json(output / 'plan.json', plan)
    try:
        root = output / 'paired'; root.mkdir()
        for name in ('session.json', 'metadata.json', 'gate.json'):
            shutil.copyfile(study / 'paired' / name, root / name)
        (root / 'shared_resources.json').symlink_to((study / 'paired/shared_resources.json').resolve())
        for role in ('A', 'B'):
            directory = root / role; directory.mkdir()
            for name in ('continuation.json', 'cache_index.json'):
                shutil.copyfile(study / 'paired' / role / name, directory / name)
            (directory / 'state_data').symlink_to((study / 'paired' / role / 'state_data').resolve(), target_is_directory=True)
        configure_pairing(root, 1)
        if not pairing_equivalent(read(root / 'paired_episode_io.json'), read(study / 'paired/paired_episode_io.json')):
            raise ValueError('Recovered paired inputs or synchronization budget differ')
        atomic_json(root / 'engineering_diagnostic.json', dict(version=VERSION, arm='paired',
            plan_sha256=sha256(output / 'plan.json'), deployable=False, prefetch=False, epochs=1,
            visits_per_role=102, original_plain_result=str(study / 'plain_result.json')))
        files = {str(p.relative_to(output)): sha256(p)
                 for p in list(root.glob('*.json')) + list(root.glob('[AB]/*.json'))
                 if p.name != 'shared_resources.json'}
        atomic_json(output / 'prepared.json', dict(complete=True, plan_sha256=sha256(output / 'plan.json'),
            files=files, gpu_started=False, cache_payload_preheated=False, utc=time.time()))
    except BaseException as error:
        atomic_json(output / 'prepare_failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    return plan


def verify(output, plan):
    prepared = read(output / 'prepared.json')
    if (plan['version'] != VERSION or plan['arms'] != ['paired'] or plan['epochs_per_role'] != 1
            or plan['visits_per_role'] != 102 or plan['aggregate_visits_per_arm'] != 204
            or plan['prefetch'] is not False or not 60 <= plan['max_seconds'] <= 360
            or plan['admission_max_seconds'] != 30 or plan['cleanup_reserve_seconds'] != 20
            or plan['session_deadline'] != SESSION_DEADLINE
            or prepared['plan_sha256'] != sha256(output / 'plan.json')
            or sha256(__file__) != plan['recovery_code_sha256']
            or sha256(plan['claim']) != plan['claim_sha256']
            or read(plan['claim'])['output'] != str(output)):
        raise ValueError('Recovery registration or source changed')
    if any(sha256(output / name) != expected for name, expected in prepared['files'].items()):
        raise ValueError('Prepared recovery metadata changed')
    original, helper = inspect_original(Path(plan['study']))
    if original != plan['original']:
        raise ValueError('Original evidence changed after registration')
    root = output / 'paired'
    if shared_inputs(root) != original['plan']['source_inputs']['shared_train']:
        raise ValueError('Recovery must use exactly the original shared train93 inputs')
    for role in ('A', 'B'):
        source = Path(original['plan']['source_inputs']['source']) / role / 'state_data'
        if (root / role / 'state_data').resolve() != source.resolve() or (root / role / 'episode_io.json').exists():
            raise ValueError('Preserve original immutable input links and disabled prefetch')
    return helper


def await_admission(output, plan, cutoff):
    began = time.time(); end = min(cutoff, began + plan['admission_max_seconds']); samples = []
    exited = plan['original']['plain_workers_exited_utc']
    while True:
        now = time.time()
        if (output / 'STOP').exists():
            raise InterruptedError('Recovery STOP during admission')
        if now >= end:
            raise TimeoutError('Fresh joint admission did not pass within the recovery allowance')
        row = read(output / 'paired/shared_resources.json')
        gate = parallel_admission(row, now)
        gate.update(utc=now, sampled_utc=row['utc_epoch'], after_plain_exit=row['utc_epoch'] > exited,
                    elapsed_seconds=now-began)
        gate['admitted'] = gate['admitted'] and gate['after_plain_exit']
        samples.append(gate)
        atomic_json(output / 'paired_admission.json', dict(samples=samples, latest=gate,
            deadline=end, unchanged_gpu_min_mib=22*1024, unchanged_ram_min_bytes=24*1024**3))
        if gate['admitted']:
            return gate
        time.sleep(min(.25, max(.001, end-time.time())))


def _run(output):
    output = Path(output).resolve(); plan = read(output / 'plan.json'); began = time.time()
    end = min(plan['session_deadline'], began + plan['max_seconds']); cutoff = end-20
    with (output / 'run_started.json').open('x') as stream:
        json.dump(dict(process_identity(os.getpid()), started_utc=began, bounded_deadline=end,
                       training_cutoff=cutoff, original_session_deadline=SESSION_DEADLINE), stream)
        stream.flush(); os.fsync(stream.fileno())
    try:
        helper = verify(output, plan)
        root = output / 'paired'
        if any((root / role / 'training').exists() for role in ('A', 'B')) or any((root / 'paired_episode_io').iterdir()):
            raise ValueError('Never resume or repeat a started recovery')
        assert_no_worker(root)
        if time.time() >= cutoff:
            raise TimeoutError('Insufficient original session time')
        atomic_json(output / 'status.json', dict(stage='admission', utc=time.time(), deadline=end))
        await_admission(output, plan, cutoff)
        if (output / 'STOP').exists() or time.time() >= cutoff:
            raise InterruptedError('Recovery STOP or cutoff before paired launch')
        atomic_json(output / 'status.json', dict(stage='paired', utc=time.time(), deadline=end))
        arm_began = time.time()
        def execute(name, args, completion, *, update_status=False):
            return helper.run_worker(output, 'paired', name, args, completion, cutoff, update_status=update_status)
        workers = paired_frontends(execute, root, 1)
        atomic_json(output / 'paired_workers.json', workers)
        if not workers['complete']:
            raise RuntimeError('Recovered paired arm failed: ' + str(workers['errors']))
        paired = helper.arm_summary(output, 'paired', plan['original']['plan'])
        wall = time.time()-arm_began
        paired.update(full_arm_wall_seconds=wall, full_arm_aggregate_updates_per_second=204/wall)
        atomic_json(output / 'paired_result.json', paired)
        plain = plan['original']['plain']
        for role in ('A', 'B'):
            for key in ('updates', 'anchors', 'target_ticks', 'sampling_exposure'):
                if paired['roles'][role][key] != plain['roles'][role][key]:
                    raise ValueError('Recovered paired legal supervision differs from original plain')
        atomic_json(output / 'result.json', dict(software_complete=True, deployable=False,
            old_study_remains_failed=True, independent_recovery_window=True, seconds=time.time()-began,
            original_plain_result=dict(path=str(Path(plan['study']) / 'plain_result.json'),
                sha256=plan['original']['bindings']['plain_result.json']),
            arms=dict(plain=plain, paired=paired), confounds=plan['confounds'],
            paired_over_plain_epoch_throughput=paired['aggregate_updates_per_second']/plain['aggregate_updates_per_second']))
        atomic_json(output / 'status.json', dict(stage='complete', utc=time.time(), deployable=False))
    except BaseException as error:
        atomic_json(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            seconds=time.time()-began, original_study_unchanged=True, automatic_retry=False))
        atomic_json(output / 'status.json', dict(stage='failed', utc=time.time(), error=repr(error)))
        raise


def run(output):
    output = Path(output).resolve(); previous = {}
    def stop(signum, frame):
        (output / 'STOP').touch(exist_ok=True)
        atomic_json(output / 'stop_requested.json', dict(signal=signum, utc=time.time(),
            scope='This new recovery owner and its owned paired children only; no automatic retry'))
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stop)
    try:
        return _run(output)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=360)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.study, args.output, max_seconds=args.max_seconds)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
