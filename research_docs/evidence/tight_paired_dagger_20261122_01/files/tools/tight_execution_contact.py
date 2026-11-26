"""One registered contact-recovery experiment, returning decisions to the agent."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from tools.runtime import RUN, kill_owned, proc, read, sha, supervised_owner, write
from tools.tight_dagger import (Stages, assert_paired_plans, candidate_pair,
    phase_deadlines, prepare_backend, prepare_frontend_stages, share_monitor,
    takeover_plan, train_frontend_stage)
from tools.tight_execution_centered import ORIGINAL, COMPUTE_DEADLINE, ABSOLUTE_DEADLINE
from tools.tight_execution_follow import residual_command, select_safe_candidate
from tools.tight_execution_learning import GROUPS, HISTORY, PARENT, run_child
from visuotactile.execution_learning import PROJECTION_FY_LOSS
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT


ROOT = ORIGINAL / 'adaptive_contact_recovery_01'
COMPARATOR = ORIGINAL / 'adaptive_tick_window_01'


def checked_root(root):
    if root.resolve() != ROOT:
        raise ValueError('Only the registered unique contact-recovery root is allowed')
    return root.resolve()


def contact_evidence():
    rows = []
    for group in GROUPS:
        directory = COMPARATOR / 'formal' / group / 'backend' / group
        for path in sorted((directory / 'evaluation').glob('*/trace.npz')):
            with np.load(path) as trace:
                count = len(trace['command'])
                contact = np.linalg.norm(trace['object_contact'], axis=-1).reshape(count, -1).max(-1) > .1
                lifted = trace['lifted'].reshape(count, -1).any(-1)
                first = lambda values: int(np.flatnonzero(values)[0]) if np.any(values) else None
                rows.append(dict(group=group, identity=path.parent.name, ticks=count,
                    first_contact=first(contact), first_contact_loss=first(contact[:-1] & ~contact[1:]),
                    first_lift=first(lifted), source=str(path), sha256=sha(path)))
    if len(rows) != 20:
        raise ValueError('Expected all twenty completed full-student traces')
    return dict(rows=rows, contact_threshold=.1, test22=False,
        limitation='Observable contact transitions, not proof of failure causality or hidden-state counterfactual')


def backend_contract():
    return dict(output_contract=REACHABLE_CENTERED_OUTPUT,
        loss_contract=dict(PROJECTION_FY_LOSS, regularizer_weight=0.),
        history_contract=HISTORY, tick_window_contract=dict(TICK_WINDOW_CONTRACT), learning_rate=1e-4)


def prepare(root):
    root = checked_root(root)
    if root.exists():
        raise ValueError('Do not overwrite or repeat a registered attempt')
    now = time.time()
    if now > COMPUTE_DEADLINE - 7 * 3600:
        raise TimeoutError('Insufficient original budget for this bounded paired cycle')
    previous = read(COMPARATOR / 'finished.json')
    for group in GROUPS:
        summary = previous['groups'][group]['summary']
        if not summary['complete'] or summary['software_failures'] or summary['completed'] != 10:
            raise ValueError('Missing actual complete-policy boundary')
    evidence = contact_evidence()
    root.mkdir()
    write(root / 'handoff_result.json', dict(previous=previous, contact=evidence,
        intentional_old_handoff='adaptive_recovery_01 InterruptedError is not a failed simulation'))
    source_round = Path(read(ORIGINAL / 'adaptive_recovery_01/formal_sources.json')['parent_round'])
    source_pairs = read(source_round / 'round_finished.json')['groups']
    pairs = {}
    for group in GROUPS:
        pair = copy.deepcopy(source_pairs[group]['selection']['selected_pair'])
        checkpoint = COMPARATOR / 'formal' / group / 'backend' / group / 'training/best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['configuration']['base_sha256'] != pair['frontend_sha256']:
            raise ValueError('Current complete pair frontend binding differs')
        pair.update(backend=str(checkpoint), backend_sha256=sha(checkpoint), backend_updates=saved['updates'],
            evaluation_root=str(checkpoint.parents[1] / 'evaluation'),
            output_contract=saved['configuration']['output_contract'],
            loss_contract=saved['configuration']['loss_contract'], history_contract=HISTORY)
        pairs[group] = pair
    configuration = copy.deepcopy(read(PARENT / 'configuration.json'))
    source_pool = Path(configuration['pool_root']).resolve()
    pool = root / 'pool'
    pool.mkdir()
    with (source_pool / 'pool.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        shutil.copy2(source_pool / 'manifest.json', pool / 'manifest.json')
    pool_count = len(read(pool / 'manifest.json')['entries'])
    end = min(now + 6 * 3600, COMPUTE_DEADLINE)
    configuration.update(pool_root=str(pool), rounds=1, first_round=1, maximum_hours=6, groups=pairs)
    configuration['frontend'].update(maximum_anchor_epochs=2, minimum_anchor_epochs=2,
        validation_anchor_epochs=1, maximum_hours=1, target_window_contract='recovery_onset_commit12_v1')
    configuration['backend'].update(maximum_control_tick_epochs=1, minimum_control_tick_epochs=1,
        validation_epoch_interval=1, maximum_hours=.5, learning_rate=1e-4)
    configuration['collection'].update(attempts_per_group=12, takeover_ticks=[58, 70, 82],
        plan_seed=2026111313, seed_start=36113000, event_seed_start=37113000,
        development_indices=[1, 7], maximum_hours=.75)
    configuration['evaluation'].update(attempts_per_group=10, maximum_attempts_per_group=10,
        rollback='No silent adoption; validation-selected trained candidate, explicit diagnostic endpoint if update0 wins')
    configuration['matched_protocol'].update(version='contact_recovery_20261113_v1', rounds=1,
        attempts_per_group=12, frontend_anchor_epochs=2, backend_tick_epochs=1, maximum_hours=6,
        endpoint_selection='Common frontend exposure; validation selected rear, diagnostic endpoint fallback')
    configuration['performance'].update(first_frontend_round=1, parallel_frontends=False)
    configuration.pop('round_schedule', None)
    configuration['stopping'] = 'Exactly this registered cycle, then real-agent decision; no automatic next round'
    write(root / 'configuration.json', configuration)
    shutil.copy2(COMPARATOR / 'formal/QG01/backend/evaluation_plan.json', root / 'evaluation_plan.json')
    registration = dict(started_utc=now, compute_deadline=COMPUTE_DEADLINE,
        absolute_deadline=ABSOLUTE_DEADLINE, stage_deadline=end,
        hypothesis='Improved held-history fit has not transferred to closed-loop grasp. New centered-policy recovery around observed contact transitions may cover its shifted state distribution; not proven causal.',
        takeover_ticks=[58, 70, 82], phase='keep existing clean global commit12 alignment; no new phase-diversity claim',
        attempts_each=12, frontend_anchor_epochs=2, backend_tick_epochs=1, evaluation_each=10,
        pool_source=str(source_pool), pool_source_sha256=sha(pool / 'manifest.json'), inherited_corrections=pool_count,
        comparison_limit='Latest legitimate pool plus new on-policy corrections, both modules and legacy onset window change; not single-variable or equal-total-exposure',
        target_window_commit='58ed6e162342c8546c850f1804fabbd51805516c',
        current_pool_differs_from_previous_training=True, test22=False,
        minimum_new_train_anchors=4, original_success_and_executor_unchanged=True)
    write(root / 'registration.json', registration)
    session = copy.deepcopy(read(COMPARATOR / 'session.json'))
    session.update(started=now, deadline=end, compute_deadline=end, original_compute_deadline=COMPUTE_DEADLINE,
        original_absolute_deadline=ABSOLUTE_DEADLINE, resource_root=str(root),
        owned_worker_roots=[str(root)], dagger=configuration, cwd=str(Path.cwd()))
    session.pop('matrix_owner', None)
    write(root / 'session.json', session)
    families = read(Path(read(Path(configuration['family_source']))['parent']) / 'frozen_families.json')
    plans = {group: takeover_plan(read(Path(pairs[group]['data_root']) / 'data.json'), families,
        dict(configuration['collection'], round_index=1, source_group=group)) for group in GROUPS}
    write(root / 'paired_opportunities.json', assert_paired_plans(plans))
    evaluation_seeds = {item['spec']['seed'] for item in read(root / 'evaluation_plan.json')}
    for group in GROUPS:
        if evaluation_seeds & {item['seed'] for item in plans[group]}:
            raise ValueError('Collection and development seeds overlap')
        job = root / 'round_01' / group
        local = copy.deepcopy(configuration)
        local['collection'].update(round_index=1, source_group=group)
        write(job / 'configuration.json', local)
        write(job / 'initial_pair.json', pairs[group])
        write(job / 'collection_plan.json', plans[group])
        write(job / 'session.json', dict(session, source_root=pairs[group]['data_root']))
        write(job / 'collector_registration.json', {name: sha(job / name) for name in
            ('configuration.json', 'initial_pair.json', 'collection_plan.json')})
        share_monitor(root, job)
    write(root / 'status.json', dict(state='PREPARED', stage='awaiting_reviewed_launch', utc=time.time()))


def status(root, stage, **fields):
    write(root / 'status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(), utc=time.time(), **fields))


@supervised_owner
def run(root):
    from visuotactile.narrow_data import ingest_dagger_pool, freeze_dagger_snapshot
    root = checked_root(root)
    registration = read(root / 'registration.json')
    end = registration['stage_deadline']
    if (root / 'launch.json').exists():
        raise ValueError('No implicit repeat; recovery requires a separately registered attempt')
    if time.time() >= min(end - 5 * 3600, COMPUTE_DEADLINE):
        raise TimeoutError('Admission budget exhausted')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Committed clean source required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    files = subprocess.check_output(['git', 'ls-files', '*.py', 'visuotactile/run_cpu.sh', 'configs/tight_state.json'], text=True).splitlines()
    write(root / 'source_freeze.json', {str(Path(path).resolve()): sha(path) for path in files})
    write(root / 'launch.json', dict(pid=os.getpid(), process=proc(os.getpid()), source_head=head,
        cwd=str(Path.cwd()), utc=time.time(), compute_deadline=COMPUTE_DEADLINE, stage_deadline=end))
    for path in (root / 'session.json', *root.glob('round_01/*/session.json')):
        session = read(path)
        session['source_head'] = head
        write(path, session)
    monitor = None
    stages = Stages(root)
    try:
        status(root, 'resource_admission')
        with (root / 'monitor.log').open('a') as log:
            monitor = subprocess.Popen(RUN + ['-m', 'visuotactile.state_resources', '--root', str(root)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while not (root / 'shared_resources.json').exists():
            if monitor.poll() is not None or time.time() - started > 30:
                raise RuntimeError('Resource monitor unavailable')
            time.sleep(.2)
        pool = root / 'pool'
        run_id = ORIGINAL.name + '_contact_recovery_01'
        for group in GROUPS:
            job = root / 'round_01' / group
            status(root, 'full_student_contact_collection', group=group)
            phase_deadlines(root, job, collection_deadline=min(2700, end - time.time() - 14400))
            run_child(root, group + '_collect', ['-m', 'tools.tight_finish', 'dagger_collect', '--root', str(job)], min(end - 14400, time.time() + 2800))
            pair = read(job / 'initial_pair.json')
            receipt = ingest_dagger_pool(pool, job, run_id=run_id, source_group=group,
                round_index=1, core_data_root=pair['data_root'])
            write(job / 'ingestion.json', receipt)
            if not receipt['integrity_passed']:
                raise ValueError('Actual correction integrity check failed')
        snapshot = root / 'round_01/snapshot.json'
        frozen = freeze_dagger_snapshot(pool, snapshot, run_id=run_id, round_index=1)
        current = [entry for entry in frozen['entries'].values()
            if entry['cohort'] == 'train' and entry['pool_source']['run_id'] == run_id]
        if sum(entry['exposure']['supervised_anchors'] for entry in current) < 4:
            write(root / 'finished.json', dict(boundary='insufficient_new_legal_supervision',
                automatic_replacement=False, new_training=False, return_to_agent=True, utc=time.time()))
        else:
            reuse = None
            for group in GROUPS:
                status(root, 'frontend_features_and_training', group=group)
                if time.time() >= min(end - 7200, COMPUTE_DEADLINE):
                    raise TimeoutError('No complete-stage reserve before frontend')
                front = prepare_frontend_stages(root, group, 1, stages, snapshot, reuse)
                train_frontend_stage(root, group, 1, stages, front)
                reuse = front / 'cache_index.json'
            for group in GROUPS:
                status(root, 'paired_cache_refresh', group=group)
                backend = prepare_backend(root, group, 1)
                for path in (backend / 'configuration.json', backend / group / 'configuration.json',
                        backend / 'session.json', backend / group / 'session.json'):
                    configuration = read(path)
                    configuration['residual_pair'].update(backend_contract())
                    write(path, configuration)
                phase_deadlines(root, backend, cache_deadline=min(3600, end - time.time() - 3600))
                run_child(root, group + '_cache', ['-m', 'tools.tight_finish', 'residual_cache', '--root', str(backend)], min(end - 3600, time.time() + 3600))
                status(root, 'backend_tick_window_training', group=group)
                phase_deadlines(root, backend, training_deadline=min(1800, end - time.time() - 3000))
                run_child(root, group + '_train', residual_command('residual-train', backend / group), min(end - 3000, time.time() + 1800))
                write(backend / 'candidate_selection.json', select_safe_candidate(backend / group / 'training'))
                status(root, 'held_out_offline_validation', group=group)
                phase_deadlines(root, backend, offline_deadline=min(900, end - time.time() - 2400))
                run_child(root, group + '_offline', residual_command('residual-offline', backend / group), min(end - 2400, time.time() + 900))
            results = {}
            for group in GROUPS:
                job = root / 'round_01' / group
                backend = job / 'backend'
                status(root, 'complete_student_development', group=group)
                phase_deadlines(root, backend, evaluation_deadline=min(1800, end - time.time()))
                run_child(root, group + '_eval', ['-m', 'tools.tight_finish', 'residual_evaluate', '--root', str(backend)], min(end, time.time() + 1800))
                results[group] = dict(pair=candidate_pair(job),
                    frontend_exposure=read(job / 'dp/training/finished.json'),
                    backend_exposure=read(backend / group / 'training/finished.json'),
                    selection=read(backend / 'candidate_selection.json'),
                    evaluation_receipt=read(backend / 'evaluation_complete.json'),
                    summary=read(backend / group / 'evaluation_summary.json'))
            write(root / 'finished.json', dict(groups=results, utc=time.time(), test22=False,
                research_decision='Return to real agent; no automatic next round or success claim'))
        write(root / 'status.json', dict(state='COMPLETE', stage='awaiting_research_decision', utc=time.time()))
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time()))
        write(root / 'status.json', dict(state='FAILED', stage='requires_actual_diagnosis', error=repr(error), utc=time.time()))
        raise
    finally:
        stages.close()
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
