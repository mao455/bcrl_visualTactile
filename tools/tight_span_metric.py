"""Bounded reachable-span loss experiment; frozen reference endpoints are reused."""
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
from visuotactile.execution_learning import SPAN_PROJECTION_FY_LOSS


ROOT = ORIGINAL / 'rear_span_metric_01'
RECOVERY_ROOT = ORIGINAL / 'rear_span_metric_eval_recovery_01'
REFERENCE = ORIGINAL / 'rear_replay_balance_01/population'
CONTACT = ORIGINAL / 'adaptive_contact_recovery_01'
GROUPS = ('QG01', 'G01')
ARMS = ('reference', 'span')
ORDER = (('reference', 'QG01'), ('span', 'G01'),
         ('span', 'QG01'), ('reference', 'G01'))


def checked_root(root, recovery=False):
    if root.resolve() != (RECOVERY_ROOT if recovery else ROOT):
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
        identity = f'span_metric_fresh_{index:03d}'
        seed = 27190000 + index
        spec = dict(identity=identity, family=identity, split='new', cohort='fresh', seed=seed,
            collision_gate='live_collision_mesh_bounds_v1', narrow_reset=direct_configuration(domain, seed))
        rows.append(dict(identity=identity, spec=spec, record_video=index == 0))
    return rows


def verify_bindings(registration):
    for path, expected in registration['bindings'].items():
        if sha(path) != expected:
            raise ValueError('Registered source changed: ' + path)


def conditioning_measurement():
    import torch
    from visuotactile.execution_learning import ideal_execution_bounds
    source = REFERENCE / 'QG01/backend/QG01'
    saved = torch.load(source / 'training/last.pt', map_location='cpu', weights_only=False)
    train = read(source / 'state_data/data.json')['splits']['train']
    coefficients = []
    for identity in train:
        record = torch.load(source / 'nominal' / (identity+'.pt'), map_location='cpu', weights_only=False)
        bounds = ideal_execution_bounds(record['robot'][:, 58:87], record['limits'], saved['configuration']['execution_adapter'])
        span = bounds[..., 1]-bounds[..., 0]
        coefficients.append(((span*saved['model']['action_scale'])**2)[record['supervision_mask'].bool()])
    values = torch.cat(coefficients)
    medians = values.median(0).values
    return dict(episodes=len(train), legal_ticks=len(values), per_joint_median_coefficient=medians.tolist(),
        minimum=values.min().item(), maximum=values.max().item(),
        median_coordinate_ratio=(medians.max()/medians.min()).item(),
        arm_median=values[:, :7].median().item(), hand_median=values[:, 7:].median().item(),
        scope='Full legal training population, not parameter-gradient contributions or proof of a success bottleneck',
        coefficient='(action_scale * reachable_span)^2 in the centered raw-space FY gradient',
        source_cache=str(source / 'nominal/index.json'), source_cache_sha256=sha(source / 'nominal/index.json'),
        source_statistics_sha256=saved['configuration']['residual_statistics_sha256'])


def prepare(root):
    from tools.tight_finish import residual_prepare
    root = checked_root(root)
    if root.exists():
        raise ValueError('Never overwrite an existing attempt')
    now = time.time()
    if now + 7200 >= COMPUTE_DEADLINE:
        raise TimeoutError('Insufficient original budget for four bounded complete-policy cells')
    boundary = read(ORIGINAL / 'ab_replication_completion_01/finished.json')
    if not boundary['opportunities_matched'] or not boundary['matched_increment_passed']:
        raise ValueError('Require the completed matched frozen cohort boundary')
    completed_replay = read(REFERENCE.parent / 'finished.json')
    if not all(item['audit']['matched_increment_passed'] for item in completed_replay['arms'].values()):
        raise ValueError('Require completed matched replay evidence')
    prior = read(CONTACT / 'finished.json')
    coverage = {group: prior['groups'][group]['backend_exposure']['coverage'] for group in GROUPS}
    if coverage[GROUPS[0]] != coverage[GROUPS[1]]:
        raise ValueError('Source coverage differs')
    weights = {arm: population_weights(coverage[GROUPS[0]]) for arm in ARMS}
    configuration = read(CONTACT / 'round_01/QG01/backend/configuration.json')
    plans = evaluation_plan(configuration['domain'])
    end = min(now + 7200, COMPUTE_DEADLINE - 300)
    root.mkdir()
    session = copy.deepcopy(read(CONTACT / 'session.json'))
    session.update(resource_root=str(root), matrix_owner=str(root), owned_worker_roots=[str(root)],
        started=now, deadline=ABSOLUTE_DEADLINE, compute_deadline=COMPUTE_DEADLINE, cwd=str(Path.cwd()))
    write(root / 'session.json', session)
    write(root / 'conditioning.json', conditioning_measurement())
    registration = dict(started_utc=now, stage_deadline=end, compute_deadline=COMPUTE_DEADLINE,
        absolute_deadline=ABSOLUTE_DEADLINE, weights=weights, source_coverage=coverage,
        hypothesis='Span-normalized FY may reduce measured1551-fold centered-head coordinate imbalance; coefficient imbalance is not proof of causal success bottleneck.',
        loss_contract=SPAN_PROJECTION_FY_LOSS,
        reference_training='Reuse completed population endpoints/receipts; no new reference training',
        candidate_training='Same original contact tensors, fresh AdamW, same population deck and one exposure; only loss metric changes',
        frontend_updates=0, new_collection=False, new_dagger_round=False, test22=False,
        epochs_each=1, attempts_per_cell=10, total_evaluations=40, order=ORDER,
        optimizer='Both endpoints originate from fresh AdamW with identical per-group contact tensors; reference training already complete and not repeated',
        selection='fixed final trained endpoint in all four cells; validation-best retained separately, never outcome-dependent adoption',
        limits=['All-history initialization differs between A/B.',
            'Metric changes coordinate weighting and global scale; Adam epsilon and clipping can mediate effects.',
            'Frozen frontends/cache; not DAgger or an all-module coverage fix.',
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
            if arm == 'span':
                rear['loss_contract'] = dict(SPAN_PROJECTION_FY_LOSS)
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
            if arm == 'reference':
                inherited = REFERENCE / group / 'backend'
                (destination / group / 'training').rmdir()
                (destination / group / 'training').symlink_to(inherited / group / 'training', target_is_directory=True)
                write(destination / 'candidate_selection.json', dict(read(inherited / 'candidate_selection.json'), inherited_training=True, new_optimizer_updates_this_attempt=0))
                for name in ('best.pt', 'finished.json', 'configuration.json'):
                    path = inherited / group / 'training' / name
                    registration['bindings'][str(path)] = sha(path)
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
        historical_result='Both replay mixtures completed A0/10 B0/10; population covered all662 train episodes without demonstrated success gain',
        previous_replay=read(REFERENCE.parent / 'finished.json'),
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
def run(root, evaluation_only=False):
    root = checked_root(root, recovery=evaluation_only)
    registration = read(root / 'registration.json')
    end = registration['stage_deadline']
    if (root / 'launch.json').exists():
        raise ValueError('No implicit rerun; recovery must use a unique identity')
    if time.time() + (4200 if evaluation_only else 6000) >= min(end, COMPUTE_DEADLINE):
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
            if arm == 'reference' or evaluation_only:
                continue
            verify_bindings(registration)
            if time.time() >= min(end - 4200, COMPUTE_DEADLINE):
                raise TimeoutError('No remaining complete training/evaluation reserve')
            backend = root / arm / group / 'backend'
            status('span_metric_training', arm=arm, group=group)
            run_child(root, arm + '_' + group + '_train', residual_command('residual-train', backend / group), min(end-4200, time.time()+1200))
            write(backend / 'candidate_selection.json', endpoint_selection(backend / group / 'training'))
            status('held_out_offline_validation', arm=arm, group=group)
            run_child(root, arm + '_' + group + '_offline', residual_command('residual-offline', backend / group), min(end-3600, time.time()+600))
        for arm in ARMS:
            result = audit(root / arm, complete=False, frozen_frontend=True)
            write(root / arm / 'trained_pre_evaluation_audit.json', result)
            if not result['matched_increment_passed']:
                raise ValueError('Actual trained contract mismatch')
        cross_metric = {}
        for group in GROUPS:
            reference = read(root / 'reference' / group / 'backend' / group / 'training/configuration.json')
            candidate = read(root / 'span' / group / 'backend' / group / 'training/configuration.json')
            changed = sorted(key for key in set(reference) | set(candidate) if reference.get(key) != candidate.get(key))
            unexpected = sorted(set(changed)-{'loss_contract', 'loss_id', 'continuation_audit'})
            cross_metric[group] = dict(changed=changed, unexpected=unexpected, matched_except_loss=not unexpected)
        write(root / 'cross_metric_trained_check.json', cross_metric)
        if not all(item['matched_except_loss'] for item in cross_metric.values()):
            raise ValueError('Unexpected cross-metric training difference; preserve endpoints for diagnosis')
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
                results[arm]['groups'][group] = dict(new_training_this_attempt=arm == 'span' and not evaluation_only, exposure=read(backend / group / 'training/finished.json'),
                    summary={key: value for key, value in summary.items() if key != 'results'},
                    selection=read(backend / 'candidate_selection.json'))
        write(root / 'finished.json', dict(arms=results, utc=time.time(), test22=False,
            evaluation_only_recovery=evaluation_only, new_frontend_updates=0,
            research_decision='Return to real Agent; no automatic adoption or extra opportunities'))
        write(root / 'status.json', dict(state='COMPLETE', stage='awaiting_research_decision', utc=time.time()))
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time()))
        write(root / 'status.json', dict(state='FAILED', stage='requires_actual_diagnosis', error=repr(error), utc=time.time()))
        raise
    finally:
        (root / 'STOP_MONITOR').touch()
        if monitor is not None:
            kill_owned(monitor)


def prepare_recovery(root):
    root = checked_root(root, recovery=True)
    if root.exists():
        raise ValueError('Never overwrite a recovery identity')
    source = read(ROOT / 'registration.json')
    if time.time()+4800 >= min(source['stage_deadline'], COMPUTE_DEADLINE):
        raise TimeoutError('No reserve for the existing evaluation cohort')
    launch = read(ROOT / 'launch.json')
    write(root / 'registration.json', dict(source, recovery_source=str(ROOT),
        source_owner=dict(pid=launch['pid'], start_ticks=launch['process']['start_ticks']),
        new_optimizer_updates=0, original_stage_deadline_preserved=True,
        recovery_reason='loss_id is derived from the intentional loss contract; old guard omitted this allowed metadata field'))
    write(root / 'status.json', dict(state='PREPARED', stage='awaiting_safe_source_exit', utc=time.time()))


@supervised_owner
def recover(root):
    from tools.tight_finish import residual_prepare
    root = checked_root(root, recovery=True)
    registration = read(root / 'registration.json')
    if (root / 'wait_launch.json').exists():
        raise ValueError('No implicit recovery rerun')
    write(root / 'wait_launch.json', dict(pid=os.getpid(), process=proc(os.getpid()), utc=time.time(),
        cwd=str(Path.cwd()), source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        stage_deadline=registration['stage_deadline']))
    try:
        while True:
            source_owner = proc(registration['source_owner']['pid'])
            if not source_owner or source_owner['start_ticks'] != registration['source_owner']['start_ticks']:
                break
            if time.time()+4200 >= min(registration['stage_deadline'], COMPUTE_DEADLINE):
                raise TimeoutError('Insufficient remaining fixed-cohort reserve; source training not interrupted')
            source_status = read(ROOT / 'status.json')
            progress = ROOT / 'span' / source_status.get('group', 'G01') / 'backend' / source_status.get('group', 'G01') / 'training/progress.json'
            if progress.exists():
                write(root / 'source_training/progress.json', dict(read(progress), source=str(progress), source_group=source_status.get('group')))
            write(root / 'status.json', dict(state='RUNNING', stage='waiting_for_healthy_source_training',
                pid=os.getpid(), source_status=source_status, utc=time.time()))
            time.sleep(5)
        failure = read(ROOT / 'failure.json')
        if 'Unexpected cross-metric training difference' not in failure['error']:
            raise ValueError('Unrecognized source failure, do not silently recover')
        if list(ROOT.glob('*/*/backend/evaluation_complete.json')) or list(ROOT.glob('*/*/backend/*/evaluation_summary.json')):
            raise ValueError('Source evaluated samples exist; reconcile without repeating them')
        for group in GROUPS:
            done = read(ROOT / 'span' / group / 'backend' / group / 'training/finished.json')
            if not done['target_reached']:
                raise ValueError('Source candidate training incomplete')
        for path, expected in read(ROOT / 'source_freeze.json').items():
            if sha(path) != expected:
                raise ValueError('Source freeze drift')
        session = read(ROOT / 'session.json')
        session.update(resource_root=str(root), matrix_owner=str(root), owned_worker_roots=[str(root)], cwd=str(Path.cwd()))
        write(root / 'session.json', session)
        for arm in ARMS:
            write(root / arm / 'frozen_frontends.json', read(ROOT / arm / 'frozen_frontends.json'))
            for group in GROUPS:
                source = ROOT / arm / group / 'backend'
                destination = root / arm / group / 'backend'
                residual_prepare(destination, configuration=read(source / 'configuration.json'))
                (root / arm / group / 'dp').symlink_to((ROOT / arm / group / 'dp').resolve(), target_is_directory=True)
                for name in ('training', 'nominal'):
                    (destination / group / name).rmdir()
                    (destination / group / name).symlink_to((source / group / name).resolve(), target_is_directory=True)
                for name in ('evaluation_plan.json', 'candidate_selection.json', 'cache_complete.json'):
                    write(destination / name, read(source / name))
                share_monitor(root, destination)
                for path in (destination / 'session.json', destination / group / 'session.json'):
                    current = read(path)
                    current.update(resource_root=str(root), matrix_owner=str(root), owned_worker_roots=[str(root)],
                        cwd=str(Path.cwd()), evaluation_deadline=registration['stage_deadline'], compute_deadline=COMPUTE_DEADLINE)
                    write(path, current)
                for path in (source / group / 'training/best.pt', destination / 'evaluation_plan.json', destination / 'configuration.json'):
                    registration['bindings'][str(path)] = sha(path)
            result = audit(root / arm, complete=False, frozen_frontend=True)
            write(root / arm / 'prepared_audit.json', result)
            if not result['matched_increment_passed']:
                raise ValueError('Recovery prepared fairness mismatch')
        write(root / 'registration.json', registration)
        write(root / 'source_handoff.json', dict(source_failure=failure, training_preserved=True,
            original_opportunities_preserved=True, repeated_training=False, repeated_evaluation=False, utc=time.time()))
        run(root, evaluation_only=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time()))
        write(root / 'status.json', dict(state='FAILED', stage='recovery_requires_diagnosis', error=repr(error), utc=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'prepare-recovery', 'recover'))
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    dict(prepare=prepare, run=run, **{'prepare-recovery': prepare_recovery, 'recover': recover})[arguments.mode](arguments.root)


if __name__ == '__main__':
    main()
