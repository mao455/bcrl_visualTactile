"""Frozen complete-policy replication; ordinary orchestration, not research reasoning."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import time
import traceback

from tools.runtime import RUN, kill_owned, proc, read, sha, supervised_owner, write
from tools.tight_dagger import share_monitor
from tools.tight_execution_centered import ORIGINAL, COMPUTE_DEADLINE, ABSOLUTE_DEADLINE
from tools.tight_execution_learning import run_child
from visuotactile.narrow_data import direct_configuration


ROOT = ORIGINAL / 'success_replication_01'
CONTACT = ORIGINAL / 'adaptive_contact_recovery_01'
PREVIOUS = ORIGINAL / 'adaptive_tick_window_01'
SOURCES = {
    'contact': CONTACT / 'round_01/G01/backend',
    'previous': PREVIOUS / 'formal/G01/backend',
}
ORDER = ((0, 'contact'), (0, 'previous'), (1, 'previous'), (1, 'contact'))
SEED_START = 27160000


def checked_root(root):
    if root.resolve() != ROOT:
        raise ValueError('Only the unique registered replication root is allowed')
    return root.resolve()


def plans(domain):
    result = []
    for index in range(20):
        identity = f'replication_fresh_{index:03d}'
        seed = SEED_START + index
        spec = dict(identity=identity, family=identity, split='new', cohort='fresh',
            seed=seed, collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=direct_configuration(domain, seed))
        result.append(dict(identity=identity, spec=spec, record_video=index % 10 == 0))
    return [result[:10], result[10:]]


def aggregate(summaries):
    rows = {label: [] for label in SOURCES}
    for item in summaries:
        summary = item['summary']
        if not summary['complete'] or summary['software_failures'] or summary['completed'] != 10:
            raise ValueError('Incomplete evaluation is not zero success')
        rows[item['label']].extend(summary['results'])
    indexed = {label: {row['identity']: row for row in values} for label, values in rows.items()}
    if any(len(values) != 20 for values in rows.values()) or any(len(values) != 20 for values in indexed.values()):
        raise ValueError('Expected twenty distinct attempts per complete pair')
    if set(indexed['contact']) != set(indexed['previous']):
        raise ValueError('Paired identities differ')
    paired = []
    for identity, current in indexed['contact'].items():
        old = indexed['previous'][identity]
        if current['seed'] != old['seed'] or current['narrow_reset'] != old['narrow_reset']:
            raise ValueError('Paired reset specifications differ')
        paired.append(dict(identity=identity, seed=current['seed'],
            both_legal=bool(current['legal'] and old['legal']),
            contact_success=bool(current['success'] and current['legal']),
            previous_success=bool(old['success'] and old['legal'])))
    totals = {}
    for label, values in rows.items():
        valid = [row for row in values if row['legal']]
        totals[label] = dict(attempts=len(values), valid=len(valid), rejected=len(values)-len(valid),
            successes=sum(bool(row['success']) for row in valid),
            native_lifts=sum(bool(row['native_lift']) for row in valid),
            first_subgoal_reached=sum(row['native_subgoals'] > 0 for row in valid))
    return dict(totals=totals, paired=paired, test22=False,
        limitation='Development replication after observed success; not untouched test or component-causal evidence. Same specs/seeds, not bitwise hidden-state pairing.')


def prepare(root):
    root = checked_root(root)
    if root.exists():
        raise ValueError('Never overwrite or repeat an existing registered attempt')
    now = time.time()
    if now >= COMPUTE_DEADLINE - 7200:
        raise TimeoutError('Insufficient original budget for the fixed paired replication')
    boundary = read(CONTACT / 'finished.json')
    current = boundary['groups']['G01']
    for group in ('QG01', 'G01'):
        summary = boundary['groups'][group]['summary']
        if not summary['complete'] or summary['software_failures'] or summary['completed'] != 10:
            raise ValueError('Missing completed contact-recovery boundary')
    success = [row for row in current['summary']['results'] if row['success'] and row['legal']]
    if len(success) != 1 or current['summary']['expert_takeover']:
        raise ValueError('Registration requires the observed single autonomous B success')
    session = read(SOURCES['contact'] / 'session.json')
    if session['domain'] != read(SOURCES['previous'] / 'session.json')['domain']:
        raise ValueError('Do not change the development distribution')
    batches = plans(session['domain'])
    end = min(now + 7200, COMPUTE_DEADLINE)
    root.mkdir()
    jobs = []
    pairs = {}
    for label, source in SOURCES.items():
        pair = read(CONTACT / 'round_01/G01/initial_pair.json') if label == 'previous' else current['pair']
        if sha(source / 'G01/training/best.pt') != pair['backend_sha256'] or sha(pair['frontend']) != pair['frontend_sha256']:
            raise ValueError('Frozen complete-pair identity differs')
        pairs[label] = {key: pair[key] for key in ('frontend', 'frontend_sha256', 'backend', 'backend_sha256')}
    for batch, label in ORDER:
        source = SOURCES[label]
        destination = root / f'batch_{batch:02d}_{label}'
        group = destination / 'G01'
        (group / 'training').mkdir(parents=True)
        (group / 'training/best.pt').symlink_to(source / 'G01/training/best.pt')
        (group / 'state_data').symlink_to((source / 'G01/state_data').resolve())
        (group / 'offline').mkdir()
        write(group / 'offline/result.json', read(source / 'G01/offline/result.json'))
        job_session = copy.deepcopy(read(source / 'session.json'))
        job_session.update(resource_root=str(root), owned_worker_roots=[], cwd=str(Path.cwd()),
            deadline=end, evaluation_deadline=end, training_deadline=end, started=now,
            source_head=None, matrix_owner=str(root), matrix_deadline=end)
        job_session['residual_pair'].update(groups=['G01'], cohort='development', videos_per_group=1)
        for directory in (destination, group):
            write(directory / 'session.json', job_session)
            write(directory / 'configuration.json', job_session)
            share_monitor(root, directory)
        write(destination / 'evaluation_plan.json', batches[batch])
        jobs.append(dict(batch=batch, label=label, root=str(destination),
            plan_sha256=sha(destination / 'evaluation_plan.json')))
    session.update(resource_root=str(root), owned_worker_roots=[job['root'] for job in jobs],
        deadline=end, training_deadline=end, evaluation_deadline=end, started=now,
        source_head=None, cwd=str(Path.cwd()))
    write(root / 'session.json', session)
    write(root / 'registration.json', dict(started_utc=now, stage_deadline=end,
        compute_deadline=COMPUTE_DEADLINE, absolute_deadline=ABSOLUTE_DEADLINE,
        jobs=jobs, pairs=pairs, seed_start=SEED_START, attempts_per_pair=20,
        hypothesis='The single B success may represent reproducible improvement after full-policy recovery; sampling variation remains plausible.',
        selection='Both complete pairs frozen before new outcomes; no fitting or outcome-dependent expansion',
        comparison='Contact cycle changes data, both modules and target-window contract; not single-variable or equal-exposure',
        test22=False, new_training=False, expert_takeover=False))
    compact = {}
    for group, values in boundary['groups'].items():
        compact[group] = dict(summary={key: values['summary'][key] for key in
            ('completed', 'valid', 'successes', 'native_lifts', 'first_subgoal_reached', 'software_failures', 'expert_takeover')},
            frontend={key: values['frontend_exposure'][key] for key in ('updates', 'supervised_target_ticks', 'wall_s')},
            backend={key: values['backend_exposure'][key] for key in ('updates', 'supervised_ticks', 'training_wall_s')})
    write(root / 'handoff_result.json', dict(groups=compact, success_identity=success[0]['identity'],
        first_success_tick=success[0]['first_success_tick'], source=str(CONTACT / 'finished.json'),
        source_sha256=sha(CONTACT / 'finished.json'), test22=False, utc=now))
    write(root / 'status.json', dict(state='PREPARED', stage='awaiting_launch', utc=now))


@supervised_owner
def run(root):
    root = checked_root(root)
    registration = read(root / 'registration.json')
    if (root / 'launch.json').exists():
        raise ValueError('No implicit retry of a launched attempt')
    if time.time() >= min(registration['stage_deadline'] - 5400, COMPUTE_DEADLINE):
        raise TimeoutError('Insufficient fixed-cohort reserve at admission')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Committed clean source required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    files = subprocess.check_output(['git', 'ls-files', '*.py', 'visuotactile/run_cpu.sh', 'configs/tight_state.json'], text=True).splitlines()
    write(root / 'source_freeze.json', {str(Path(path).resolve()): sha(path) for path in files})
    write(root / 'launch.json', dict(pid=os.getpid(), process=proc(os.getpid()), source_head=head,
        cwd=str(Path.cwd()), utc=time.time(), compute_deadline=COMPUTE_DEADLINE,
        stage_deadline=registration['stage_deadline']))
    monitor = None
    try:
        with (root / 'monitor.log').open('a') as log:
            monitor = subprocess.Popen(RUN + ['-m', 'visuotactile.state_resources', '--root', str(root)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while not (root / 'shared_resources.json').exists():
            if monitor.poll() is not None or time.time() - started > 30:
                raise RuntimeError('Fresh resource monitor unavailable')
            time.sleep(.2)
        summaries = []
        for job in registration['jobs']:
            if time.time() >= COMPUTE_DEADLINE:
                raise TimeoutError('Original compute cutoff')
            destination = Path(job['root'])
            if sha(destination / 'evaluation_plan.json') != job['plan_sha256']:
                raise ValueError('Registered plan changed')
            for key in ('frontend', 'backend'):
                pair = registration['pairs'][job['label']]
                if sha(pair[key]) != pair[key + '_sha256']:
                    raise ValueError('Registered complete pair changed')
            for directory in (destination, destination / 'G01'):
                session = read(directory / 'session.json')
                session['source_head'] = head
                write(directory / 'session.json', session)
            write(root / 'status.json', dict(state='RUNNING', stage='paired_complete_student_replication',
                batch=job['batch'], label=job['label'], pid=os.getpid(), utc=time.time()))
            run_child(root, destination.name, ['-m', 'tools.tight_finish', 'residual_evaluate', '--root', str(destination)],
                min(registration['stage_deadline'], COMPUTE_DEADLINE))
            summaries.append(dict(label=job['label'], batch=job['batch'],
                summary=read(destination / 'G01/evaluation_summary.json')))
        write(root / 'finished.json', dict(aggregate(summaries), utc=time.time(),
            research_decision='Return to real agent; no automatic adoption, training or added cases'))
        write(root / 'status.json', dict(state='COMPLETE', stage='awaiting_research_decision', utc=time.time()))
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time()))
        write(root / 'status.json', dict(state='FAILED', stage='requires_actual_diagnosis', error=repr(error), utc=time.time()))
        raise
    finally:
        (root / 'STOP_MONITOR').touch()
        if monitor is not None:
            kill_owned(monitor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    (prepare if arguments.mode == 'prepare' else run)(arguments.root)


if __name__ == '__main__':
    main()
