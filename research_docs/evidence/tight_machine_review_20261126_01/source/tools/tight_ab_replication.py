"""Complete the frozen current A cohort; never repeat the completed B cohort."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import time
import traceback

from tools.runtime import RUN, kill_owned, proc, read, sha, supervised_owner, write
from tools.tight_ab_fairness import audit
from tools.tight_dagger import share_monitor
from tools.tight_execution_centered import ORIGINAL, COMPUTE_DEADLINE, ABSOLUTE_DEADLINE
from tools.tight_execution_learning import run_child


ROOT = ORIGINAL / 'ab_replication_completion_01'
CONTACT = ORIGINAL / 'adaptive_contact_recovery_01'
REPLICATION = ORIGINAL / 'success_replication_01'
GROUP = 'QG01'
LIMITATIONS = [
    'A opportunities were registered after B outcomes; this is retrospective cohort completion, not prospectively balanced execution order.',
    'Historical initialization/exposure differ; matched incremental training is not all-history equality.',
    'Development-only, repeated research use; no untouched test or bitwise simulator pairing claim.',
    'B previous-version trials and the original ten-case cohort remain separate denominators.',
]


def checked_root(root):
    if root.resolve() != ROOT:
        raise ValueError('Only the uniquely registered A complement root is allowed')
    return root.resolve()


def compare_batches(pairs):
    totals = {group: dict(attempts=0, valid=0, successes=0, native_lifts=0,
                         first_subgoal_reached=0) for group in ('QG01', 'G01')}
    seen = set()
    paired = []
    for plan, first, second in pairs:
        expected = [(row['identity'], row['spec']['seed'], row['spec']['narrow_reset']) for row in plan]
        if len(plan) != 10:
            raise ValueError('Exactly ten registered opportunities per batch')
        for group, summary in (('QG01', first), ('G01', second)):
            actual = [(row['identity'], row['seed'], row['narrow_reset']) for row in summary['results']]
            if (not summary['complete'] or summary['software_failures'] or summary['completed'] != 10
                    or summary['expert_takeover'] or summary['formal_test22_used'] or actual != expected):
                raise ValueError('Incomplete, nonautonomous or mismatched evaluation evidence')
            for row in summary['results']:
                totals[group]['attempts'] += 1
                if row['legal']:
                    totals[group]['valid'] += 1
                    totals[group]['successes'] += int(bool(row['success']))
                    totals[group]['native_lifts'] += int(bool(row['native_lift']))
                    totals[group]['first_subgoal_reached'] += int(row['native_subgoals'] > 0)
        for first_row, second_row in zip(first['results'], second['results']):
            identity = first_row['identity']
            if identity in seen:
                raise ValueError('Duplicate opportunity; no silent repetition')
            seen.add(identity)
            paired.append(dict(identity=identity, seed=first_row['seed'],
                both_legal=bool(first_row['legal'] and second_row['legal']),
                A_success=bool(first_row['legal'] and first_row['success']),
                B_success=bool(second_row['legal'] and second_row['success'])))
    if len(seen) != 20:
        raise ValueError('Exactly twenty distinct paired opportunities required')
    return dict(totals=totals, paired=paired, opportunities_matched=all(row['both_legal'] for row in paired),
                test22=False, limitations=LIMITATIONS)


def verify_bindings(registration):
    for path, expected in registration['bindings'].items():
        if sha(path) != expected:
            raise ValueError('Registered evidence or model changed: ' + path)


def prepare(root):
    root = checked_root(root)
    if root.exists():
        raise ValueError('Never overwrite an existing attempt')
    now = time.time()
    if now + 3600 >= COMPUTE_DEADLINE:
        raise TimeoutError('Insufficient original budget for twenty fixed opportunities')
    completed = read(REPLICATION / 'finished.json')
    if completed['totals']['contact']['attempts'] != 20:
        raise ValueError('Completed B cohort required')
    prepared = audit(CONTACT / 'round_01', complete=False)
    actual = audit(CONTACT / 'round_01', complete=True)
    if not prepared['matched_increment_passed'] or not actual['matched_increment_passed']:
        raise ValueError('Existing contact increment is not matched; preserve failed audit separately')
    boundary = read(CONTACT / 'finished.json')
    source = CONTACT / 'round_01' / GROUP / 'backend'
    pair = {key: boundary['groups'][GROUP]['pair'][key] for key in
            ('frontend', 'frontend_sha256', 'backend', 'backend_sha256')}
    if sha(source / GROUP / 'training/best.pt') != pair['backend_sha256']:
        raise ValueError('A backend does not match the completed contact pair')
    bindings = {pair[key]: pair[key + '_sha256'] for key in ('frontend', 'backend')}
    bindings[str(REPLICATION / 'finished.json')] = sha(REPLICATION / 'finished.json')
    b_registration = read(REPLICATION / 'registration.json')
    b_pair = b_registration['pairs']['contact']
    if any(b_pair[key] != boundary['groups']['G01']['pair'][key] for key in b_pair):
        raise ValueError('Completed B does not use the matched contact pair')
    bindings.update({b_pair[key]: b_pair[key + '_sha256'] for key in ('frontend', 'backend')})
    frozen = read(REPLICATION / 'source_freeze.json')
    old_cwd = Path(read(REPLICATION / 'launch.json')['cwd'])
    equivalence = {}
    for old_path, expected in frozen.items():
        relative = Path(old_path).relative_to(old_cwd)
        if relative.parts[0] == 'visuotactile' or str(relative) in ('tools/tight_finish.py', 'tools/runtime.py', 'tools/tight_state.py'):
            equivalence[str(relative)] = sha(relative) == expected
    if not equivalence or not all(equivalence.values()):
        raise ValueError('Evaluation runtime differs from completed B source')
    verify_bindings(dict(bindings=bindings))
    root.mkdir()
    write(root / 'increment_prepared_audit.json', prepared)
    write(root / 'increment_completed_audit.json', actual)
    write(root / 'runtime_equivalence.json', equivalence)
    end = min(now + 3600, COMPUTE_DEADLINE)
    jobs = []
    session = copy.deepcopy(read(source / 'session.json'))
    for batch in (0, 1):
        b_root = REPLICATION / f'batch_{batch:02d}_contact'
        plan_path = b_root / 'evaluation_plan.json'
        summary_path = b_root / 'G01/evaluation_summary.json'
        for path in (plan_path, summary_path):
            bindings[str(path)] = sha(path)
        plan = read(plan_path)
        destination = root / f'batch_{batch:02d}_contact'
        group_root = destination / GROUP
        (group_root / 'training').mkdir(parents=True)
        (group_root / 'training/best.pt').symlink_to(Path(pair['backend']))
        (group_root / 'state_data').symlink_to((source / GROUP / 'state_data').resolve())
        write(group_root / 'offline/result.json', read(source / GROUP / 'offline/result.json'))
        job_session = copy.deepcopy(session)
        job_session.update(resource_root=str(root), owned_worker_roots=[], cwd=str(Path.cwd()),
            deadline=end, evaluation_deadline=end, training_deadline=end, started=now,
            source_head=None, matrix_owner=str(root), matrix_deadline=end)
        job_session['residual_pair'].update(groups=[GROUP], cohort='development', videos_per_group=1)
        if job_session['domain'] != read(b_root / 'session.json')['domain']:
            raise ValueError('A and B evaluation distributions differ')
        for directory in (destination, group_root):
            write(directory / 'session.json', job_session)
            write(directory / 'configuration.json', job_session)
            share_monitor(root, directory)
        write(destination / 'evaluation_plan.json', plan)
        if sha(destination / 'evaluation_plan.json') != sha(plan_path):
            raise ValueError('Exact B plan copy required')
        bindings[str(destination / 'evaluation_plan.json')] = sha(plan_path)
        jobs.append(dict(batch=batch, root=str(destination), B_summary=str(summary_path)))
    session.update(resource_root=str(root), owned_worker_roots=[job['root'] for job in jobs],
        deadline=end, training_deadline=end, evaluation_deadline=end, started=now,
        source_head=None, cwd=str(Path.cwd()))
    write(root / 'session.json', session)
    write(root / 'registration.json', dict(started_utc=now, stage_deadline=end,
        compute_deadline=COMPUTE_DEADLINE, absolute_deadline=ABSOLUTE_DEADLINE,
        jobs=jobs, pair=pair, B_pair=b_pair, bindings=bindings, attempts_A=20, new_attempts_B=0,
        hypothesis='The isolated B success did not replicate; completing fixed A counterparts enables a denominator-matched current-policy assessment without additional fitting.',
        no_retries=True, no_outcome_expansion=True, new_training=False, expert_takeover=False,
        test22=False, limitations=LIMITATIONS))
    write(root / 'status.json', dict(state='PREPARED', stage='awaiting_launch', utc=now))


@supervised_owner
def run(root):
    root = checked_root(root)
    registration = read(root / 'registration.json')
    if (root / 'launch.json').exists():
        raise ValueError('No implicit retry of a launched attempt')
    if time.time() + 2400 >= min(registration['stage_deadline'], COMPUTE_DEADLINE):
        raise TimeoutError('Insufficient reserve for the fixed cohort')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean committed source required')
    verify_bindings(registration)
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
        batches = []
        for job in registration['jobs']:
            if time.time() >= COMPUTE_DEADLINE:
                raise TimeoutError('Original compute cutoff')
            verify_bindings(registration)
            destination = Path(job['root'])
            for directory in (destination, destination / GROUP):
                session = read(directory / 'session.json')
                session['source_head'] = head
                write(directory / 'session.json', session)
            write(root / 'status.json', dict(state='RUNNING', stage='frozen_A_cohort_completion',
                batch=job['batch'], group=GROUP, pid=os.getpid(), utc=time.time()))
            run_child(root, destination.name, ['-m', 'tools.tight_finish', 'residual_evaluate', '--root', str(destination)],
                min(registration['stage_deadline'], COMPUTE_DEADLINE))
            batches.append((read(destination / 'evaluation_plan.json'),
                read(destination / GROUP / 'evaluation_summary.json'), read(job['B_summary'])))
        final_audit = audit(CONTACT / 'round_01', complete=True)
        write(root / 'final_increment_audit.json', final_audit)
        result = compare_batches(batches)
        result['matched_increment_passed'] = final_audit['matched_increment_passed']
        result['comparison_status'] = ('matched_increment_and_opportunities' if result['opportunities_matched']
            and result['matched_increment_passed'] else 'incomplete_or_unmatched')
        write(root / 'finished.json', dict(result, utc=time.time(),
            research_decision='Return to real Agent; no automatic training or added opportunities'))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)(args.root)


if __name__ == '__main__':
    main()
