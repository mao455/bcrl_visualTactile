"""A fixed rear replay-distribution experiment, not a new DAgger round."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.runtime import RUN, kill_owned, proc, read, sha, supervised_owner, write
from tools.tight_ab_fairness import audit
from tools.tight_dagger import share_monitor
from tools.tight_execution_centered import ORIGINAL, COMPUTE_DEADLINE, ABSOLUTE_DEADLINE
from tools.tight_execution_follow import residual_command
from tools.tight_execution_learning import run_child
from visuotactile.narrow_data import direct_configuration


ROOT = ORIGINAL / 'rear_replay_balance_01'
CONTACT = ORIGINAL / 'adaptive_contact_recovery_01'
GROUPS = ('QG01', 'G01')
ARMS = ('control', 'population')
ORDER = (('control', 'QG01'), ('population', 'G01'),
         ('population', 'QG01'), ('control', 'G01'))


def checked_root(root):
    if root.resolve() != ROOT:
        raise ValueError('Only the unique registered replay root is allowed')
    return root.resolve()


def population_weights(coverage):
    counts = {name: row['supervised_tick_population'] for name, row in coverage['strata'].items()}
    if set(counts) != {'core', 'historical_correction', 'current_correction'} or min(counts.values()) <= 0:
        raise ValueError('Three positive legal supervision populations required')
    total = sum(counts.values())
    result = {name: count / total for name, count in counts.items()}
    if result['core'] > .5:
        raise ValueError('This experiment retains at least half correction sampling mass')
    return result


def evaluation_plan(domain):
    rows = []
    for index in range(10):
        identity = f'replay_balance_fresh_{index:03d}'
        seed = 27170000 + index
        spec = dict(identity=identity, family=identity, split='new', cohort='fresh', seed=seed,
            collision_gate='live_collision_mesh_bounds_v1', narrow_reset=direct_configuration(domain, seed))
        rows.append(dict(identity=identity, spec=spec, record_video=index == 0))
    return rows


def verify_bindings(registration):
    for path, expected in registration['bindings'].items():
        if sha(path) != expected:
            raise ValueError('Registered source changed: ' + path)


def prepare(root):
    from tools.tight_finish import residual_prepare
    root = checked_root(root)
    if root.exists():
        raise ValueError('Never overwrite an existing attempt')
    now = time.time()
    if now + 9000 >= COMPUTE_DEADLINE:
        raise TimeoutError('Insufficient original budget for four bounded complete-policy cells')
    boundary = read(ORIGINAL / 'ab_replication_completion_01/finished.json')
    if not boundary['opportunities_matched'] or not boundary['matched_increment_passed']:
        raise ValueError('Require the completed matched frozen cohort boundary')
    prior = read(CONTACT / 'finished.json')
    coverage = {group: prior['groups'][group]['backend_exposure']['coverage'] for group in GROUPS}
    if coverage[GROUPS[0]] != coverage[GROUPS[1]]:
        raise ValueError('Source coverage differs')
    weights = dict(control=dict(core=.5, historical_correction=.2, current_correction=.3),
        population=population_weights(coverage[GROUPS[0]]))
    configuration = read(CONTACT / 'round_01/QG01/backend/configuration.json')
    plans = evaluation_plan(configuration['domain'])
    end = min(now + 9000, COMPUTE_DEADLINE - 300)
    root.mkdir()
    session = copy.deepcopy(read(CONTACT / 'session.json'))
    session.update(resource_root=str(root), matrix_owner=str(root), owned_worker_roots=[str(root)],
        started=now, deadline=ABSOLUTE_DEADLINE, compute_deadline=COMPUTE_DEADLINE, cwd=str(Path.cwd()))
    write(root / 'session.json', session)
    registration = dict(started_utc=now, stage_deadline=end, compute_deadline=COMPUTE_DEADLINE,
        absolute_deadline=ABSOLUTE_DEADLINE, weights=weights, source_coverage=coverage,
        hypothesis='Current16 corrections received12.879 exposures while only144/480 historical episodes appeared. Test population-proportional rear replay against another equal-budget old-mixture increment; stable success benefit is unproven.',
        frontend_updates=0, new_collection=False, new_dagger_round=False, test22=False,
        epochs_each=1, attempts_per_cell=10, total_evaluations=40, order=ORDER,
        optimizer='fresh AdamW from the same per-group selected contact tensors in both arms; exact resume within an attempt only',
        selection='fixed final trained endpoint in all four cells; validation-best retained separately, never outcome-dependent adoption',
        limits=['All-history initialization differs between A/B.',
            'Whole-episode rounding and optimizer counts can differ across sampling arms; report actual counts, not exact equal-update causality.',
            'Frozen frontends/cache; this tests rear weighting only and is not DAgger or an all-module coverage fix.',
            'Ten development opportunities per cell, no test22, no outcome-dependent expansion.'], bindings={})
    for arm in ARMS:
        frozen = {}
        arm_root = root / arm
        for group in GROUPS:
            source = CONTACT / 'round_01' / group / 'backend'
            pair = prior['groups'][group]['pair']
            for key in ('frontend', 'backend'):
                registration['bindings'][pair[key]] = pair[key + '_sha256']
            local = copy.deepcopy(read(source / 'configuration.json'))
            rear = local['residual_pair']
            rear.update(sampling_weights=weights[arm], maximum_control_tick_epochs=1,
                minimum_control_tick_epochs=1, comparison_control_tick_epochs=[1], validation_epoch_interval=1,
                initialization_seed=2026111319, continuation_optimizer_policy='fresh_adamw_source_tensors_only',
                evaluation_selection_policy='fixed_final_endpoint_no_adoption',
                continuation_from=dict(rear['continuation_from'], path=pair['backend'], sha256=pair['backend_sha256']),
                dagger_evaluation=dict(initial_trials=10, max_trials=10,
                    expand_if_success=1, expand_if_subgoal=1, expand_if_native_lift=2))
            destination = arm_root / group / 'backend'
            residual_prepare(destination, configuration=local)
            (arm_root / group / 'dp').symlink_to(CONTACT / 'round_01' / group / 'dp', target_is_directory=True)
            nominal = destination / group / 'nominal'
            nominal.rmdir()
            nominal.symlink_to(source / group / 'nominal', target_is_directory=True)
            write(destination / 'evaluation_plan.json', plans)
            write(destination / 'cache_complete.json', dict(complete=True, exact_reuse=True,
                source=str(source / group / 'nominal'), new_frontend_updates=0))
            share_monitor(root, destination)
            for path in (destination / 'session.json', destination / group / 'session.json'):
                current = read(path)
                current.update(resource_root=str(root), matrix_owner=str(root), owned_worker_roots=[str(root)],
                    started=now, deadline=ABSOLUTE_DEADLINE, compute_deadline=COMPUTE_DEADLINE,
                    training_deadline=end-4200, offline_deadline=end-3600, evaluation_deadline=end,
                    cwd=str(Path.cwd()))
                write(path, current)
            prepared = copy.deepcopy(read(source / group / 'training/configuration.json'))
            prepared.update(rear)
            write(destination / group / 'prepared_training_contract.json', prepared)
            frozen[group] = dict(checkpoint=pair['frontend'], sha256=pair['frontend_sha256'],
                new_optimizer_updates=0, source_dp=str(CONTACT / 'round_01' / group / 'dp'))
        write(arm_root / 'frozen_frontends.json', frozen)
        result = audit(arm_root, complete=False, frozen_frontend=True)
        write(arm_root / 'prepared_audit.json', result)
        if not result['matched_increment_passed']:
            raise ValueError('Prepared comparison mismatch: ' + repr(result['failed']))
    for path in (root.glob('*/frozen_frontends.json')):
        registration['bindings'][str(path)] = sha(path)
    for pattern in ('*/*/backend/configuration.json', '*/*/backend/evaluation_plan.json',
            '*/*/backend/*/configuration.json', '*/*/backend/*/prepared_training_contract.json'):
        for path in root.glob(pattern):
            registration['bindings'][str(path)] = sha(path)
    registration['bindings'][str(CONTACT / 'finished.json')] = sha(CONTACT / 'finished.json')
    verify_bindings(registration)
    write(root / 'registration.json', registration)
    write(root / 'handoff_result.json', dict(previous_replication=boundary,
        historical_result='Original A0/10 B1/10 remains separate; new frozen A0/20 B0/20 does not demonstrate stable B superiority',
        review_paths=[str(ORIGINAL / name / 'findings.json') for name in
            ('review_20261113_1208', 'takeover_phase_review_01', 'system_selfcheck_01')]))
    write(root / 'status.json', dict(state='PREPARED', stage='awaiting_reviewed_launch', utc=time.time()))


def endpoint_selection(directory):
    import torch
    saved = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
    if saved['updates'] <= 0:
        raise ValueError('An endpoint needs actual optimizer updates')
    shutil.copy2(directory / 'best.pt', directory / 'validation_best.pt')
    shutil.copy2(directory / 'last.pt', directory / 'best.pt')
    return dict(selection='fixed trained endpoint; diagnostic only, not adopted', updates=saved['updates'],
        sha256=sha(directory / 'best.pt'), validation_best_sha256=sha(directory / 'validation_best.pt'))


@supervised_owner
def run(root):
    root = checked_root(root)
    registration = read(root / 'registration.json')
    end = registration['stage_deadline']
    if (root / 'launch.json').exists():
        raise ValueError('No implicit rerun; recovery must use a unique identity')
    if time.time() + 7200 >= min(end, COMPUTE_DEADLINE):
        raise TimeoutError('Insufficient stage reserve at admission')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Reviewed clean committed source required')
    verify_bindings(registration)
    files = subprocess.check_output(['git', 'ls-files', '*.py', 'visuotactile/run_cpu.sh', 'configs/tight_state.json'], text=True).splitlines()
    write(root / 'source_freeze.json', {str(Path(path).resolve()): sha(path) for path in files})
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write(root / 'launch.json', dict(pid=os.getpid(), process=proc(os.getpid()), source_head=head,
        cwd=str(Path.cwd()), utc=time.time(), stage_deadline=end, compute_deadline=COMPUTE_DEADLINE))
    for path in (root / 'session.json', *root.glob('*/*/backend/session.json'), *root.glob('*/*/backend/*/session.json')):
        session = read(path)
        session['source_head'] = head
        write(path, session)
    monitor = None
    def status(stage, **fields):
        write(root / 'status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(), utc=time.time(), **fields))
    try:
        status('resource_admission')
        with (root / 'monitor.log').open('a') as log:
            monitor = subprocess.Popen(RUN + ['-m', 'visuotactile.state_resources', '--root', str(root)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while not (root / 'shared_resources.json').exists():
            if monitor.poll() is not None or time.time() - started > 30:
                raise RuntimeError('Resource monitor unavailable')
            time.sleep(.2)
        for arm, group in ORDER:
            verify_bindings(registration)
            if time.time() >= min(end - 4200, COMPUTE_DEADLINE):
                raise TimeoutError('No remaining complete training/evaluation reserve')
            backend = root / arm / group / 'backend'
            status('rear_replay_training', arm=arm, group=group)
            run_child(root, arm + '_' + group + '_train', residual_command('residual-train', backend / group), min(end-4200, time.time()+1200))
            write(backend / 'candidate_selection.json', endpoint_selection(backend / group / 'training'))
            status('held_out_offline_validation', arm=arm, group=group)
            run_child(root, arm + '_' + group + '_offline', residual_command('residual-offline', backend / group), min(end-3600, time.time()+600))
        for arm in ARMS:
            result = audit(root / arm, complete=False, frozen_frontend=True)
            write(root / arm / 'trained_pre_evaluation_audit.json', result)
            if not result['matched_increment_passed']:
                raise ValueError('Actual trained contract mismatch')
        for arm, group in ORDER:
            if time.time() >= min(end, COMPUTE_DEADLINE):
                raise TimeoutError('Original computation budget exhausted')
            backend = root / arm / group / 'backend'
            status('complete_student_development', arm=arm, group=group)
            run_child(root, arm + '_' + group + '_eval', ['-m', 'tools.tight_finish', 'residual_evaluate', '--root', str(backend)], min(end, time.time()+1200))
        results = {}
        for arm in ARMS:
            result = audit(root / arm, complete=True, frozen_frontend=True)
            write(root / arm / 'completed_audit.json', result)
            results[arm] = dict(audit=result, groups={})
            for group in GROUPS:
                backend = root / arm / group / 'backend'
                summary = read(backend / group / 'evaluation_summary.json')
                results[arm]['groups'][group] = dict(exposure=read(backend / group / 'training/finished.json'),
                    summary={key: value for key, value in summary.items() if key != 'results'},
                    selection=read(backend / 'candidate_selection.json'))
        write(root / 'finished.json', dict(arms=results, utc=time.time(), test22=False,
            new_frontend_updates=0, research_decision='Return to real Agent; no automatic adoption or extra opportunities'))
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
    arguments = parser.parse_args()
    (prepare if arguments.mode == 'prepare' else run)(arguments.root)


if __name__ == '__main__':
    main()
