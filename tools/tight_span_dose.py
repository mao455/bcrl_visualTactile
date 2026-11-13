"""One bounded common A/B continuation, with a frozen one-pass reference."""
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
from tools.tight_execution_follow import residual_command
from tools.tight_execution_learning import run_child
from tools.tight_span_metric import endpoint_selection, reuse_selected_offline, verify_bindings
from visuotactile.narrow_data import direct_configuration


ROOT = ORIGINAL / 'rear_span_dose_02'
SOURCE = ORIGINAL / 'rear_span_metric_01/span'
BOUNDARY = ORIGINAL / 'rear_span_metric_eval_recovery_02'
GROUPS = ('QG01', 'G01')
ARMS = ('reference', 'continued')
ORDER = (('reference', 'QG01'), ('continued', 'G01'),
         ('continued', 'QG01'), ('reference', 'G01'))


def checked_root(root):
    if root.resolve() != ROOT:
        raise ValueError('Only the unique registered continuation root is allowed')
    return root.resolve()


def evaluation_plan(domain):
    rows = []
    for index in range(10):
        identity = f'span_dose_fresh_{index:03d}'
        seed = 27200000 + index
        spec = dict(identity=identity, family=identity, split='new', cohort='fresh', seed=seed,
            collision_gate='live_collision_mesh_bounds_v1', narrow_reset=direct_configuration(domain, seed))
        rows.append(dict(identity=identity, spec=spec, record_video=index == 0))
    return rows


def require_budget(now, end, reserve):
    if now + reserve >= min(end, COMPUTE_DEADLINE):
        raise TimeoutError('Original budget insufficient; no deadline extension')


def prepare(root):
    from tools.tight_finish import residual_prepare
    root = checked_root(root)
    if root.exists():
        raise ValueError('Never overwrite a registered attempt')
    now = time.time()
    end = min(now + 6000, COMPUTE_DEADLINE - 120)
    require_budget(now, end, 5400)
    boundary = read(BOUNDARY / 'finished.json')
    if not all(arm['audit']['matched_increment_passed'] for arm in boundary['arms'].values()):
        raise ValueError('Previous matched cohort not complete')
    configuration = read(SOURCE / 'QG01/backend/configuration.json')
    plans = evaluation_plan(configuration['domain'])
    root.mkdir()
    session = copy.deepcopy(read(SOURCE.parent / 'session.json'))
    session.update(resource_root=str(root), matrix_owner=str(root), owned_worker_roots=[str(root)],
        started=now, deadline=ABSOLUTE_DEADLINE, compute_deadline=COMPUTE_DEADLINE, cwd=str(Path.cwd()))
    write(root / 'session.json', session)
    registration = dict(started_utc=now, stage_deadline=end, compute_deadline=COMPUTE_DEADLINE,
        absolute_deadline=ABSOLUTE_DEADLINE, hypothesis='One span-normalized pass improved arm offline errors in both groups but produced no first subgoal. Test whether two additional common population exposures improve the complete policy, without another loss change.',
        frontend_updates=0, new_collection=False, new_dagger_round=False, test22=False,
        additional_epochs_each=2, attempts_per_cell=10, total_evaluations=40, order=ORDER,
        optimizer='Common fresh AdamW from each one-pass span endpoint; not an exact optimizer continuation or a pure exposure-only causal comparison',
        selection='Fixed last endpoint; retain validation-best, no outcome-dependent expansion or adoption',
        limitations=['Historical A/B initialization/exposure differs.',
            'Within-arm incremental A/B protocol is matched; between arms training dose and optimizer restart intentionally differ.',
            'Reference has zero new updates; its inherited one-pass receipts are not new exposure.',
            'Old cohort has nine valid and one rejected opportunity per cell; do not call it ten legal task failures.'],
        bindings={str(BOUNDARY / 'finished.json'): sha(BOUNDARY / 'finished.json')})
    for arm in ARMS:
        frozen = {}
        for group in GROUPS:
            source = SOURCE / group / 'backend'
            local = copy.deepcopy(read(source / 'configuration.json'))
            rear = local['residual_pair']
            checkpoint = source / group / 'training/best.pt'
            registration['bindings'][str(checkpoint)] = sha(checkpoint)
            if arm == 'continued':
                rear.update(maximum_control_tick_epochs=2, minimum_control_tick_epochs=2,
                    comparison_control_tick_epochs=[1, 2], validation_epoch_interval=1,
                    continuation_from=dict(rear['continuation_from'], path=str(checkpoint), sha256=sha(checkpoint)))
            destination = root / arm / group / 'backend'
            residual_prepare(destination, configuration=local)
            (destination.parent / 'dp').symlink_to((SOURCE / group / 'dp').resolve(), target_is_directory=True)
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
                    training_deadline=end-3000, offline_deadline=end-2600,
                    evaluation_deadline=end, cwd=str(Path.cwd()))
                write(path, current)
            prepared = copy.deepcopy(read(source / group / 'training/configuration.json'))
            prepared.update(rear)
            write(destination / group / 'prepared_training_contract.json', prepared)
            if arm == 'reference':
                (destination / group / 'training').rmdir()
                (destination / group / 'training').symlink_to(source / group / 'training', target_is_directory=True)
                write(destination / 'candidate_selection.json', dict(read(source / 'candidate_selection.json'),
                    inherited_training=True, new_optimizer_updates_this_attempt=0))
                reuse_selected_offline(destination / group)
            frozen[group] = copy.deepcopy(read(SOURCE / 'frozen_frontends.json')[group])
        write(root / arm / 'frozen_frontends.json', frozen)
        result = audit(root / arm, complete=False, frozen_frontend=True)
        write(root / arm / 'prepared_audit.json', result)
        if not result['matched_increment_passed']:
            raise ValueError('Prepared comparison mismatch: ' + repr(result['failed']))
    for pattern in ('*/frozen_frontends.json', '*/*/backend/configuration.json',
            '*/*/backend/evaluation_plan.json', '*/*/backend/*/configuration.json',
            '*/*/backend/*/prepared_training_contract.json'):
        for path in root.glob(pattern):
            registration['bindings'][str(path)] = sha(path)
    verify_bindings(registration)
    write(root / 'registration.json', registration)
    write(root / 'status.json', dict(state='PREPARED', stage='awaiting_reviewed_launch', utc=time.time()))


@supervised_owner
def run(root):
    root = checked_root(root)
    registration = read(root / 'registration.json')
    end = registration['stage_deadline']
    if (root / 'launch.json').exists():
        raise ValueError('No implicit rerun of a launched attempt')
    require_budget(time.time(), end, 5300)
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean committed source required')
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
        began = time.time()
        while not (root / 'shared_resources.json').exists():
            if monitor.poll() is not None or time.time()-began > 30:
                raise RuntimeError('Resource monitor unavailable')
            time.sleep(.2)
        for group in GROUPS:
            require_budget(time.time(), end, 3000)
            verify_bindings(registration)
            backend = root / 'continued' / group / 'backend'
            status('common_two_exposure_training', arm='continued', group=group)
            run_child(root, group+'_train', residual_command('residual-train', backend / group), min(end-3000, time.time()+1500))
            finished = read(backend / group / 'training/finished.json')
            if not finished.get('target_reached') or finished['control_tick_epochs'] < 2:
                raise ValueError('Incomplete common training; preserve checkpoint, do not call exposure matched')
            write(backend / 'candidate_selection.json', endpoint_selection(backend / group / 'training'))
            require_budget(time.time(), end, 2600)
            status('held_out_offline_validation', arm='continued', group=group)
            run_child(root, group+'_offline', residual_command('residual-offline', backend / group), min(end-2600, time.time()+500))
        for arm in ARMS:
            result = audit(root / arm, complete=False, frozen_frontend=True)
            write(root / arm / 'trained_pre_evaluation_audit.json', result)
            if not result['matched_increment_passed']:
                raise ValueError('Trained A/B contract mismatch')
        for arm, group in ORDER:
            require_budget(time.time(), end, 0)
            status('complete_student_development', arm=arm, group=group)
            backend = root / arm / group / 'backend'
            run_child(root, arm+'_'+group+'_eval', ['-m', 'tools.tight_finish', 'residual_evaluate', '--root', str(backend)], min(end, time.time()+1100))
        results = {}
        for arm in ARMS:
            result = audit(root / arm, complete=True, frozen_frontend=True)
            write(root / arm / 'completed_audit.json', result)
            results[arm] = dict(audit=result, groups={})
            for group in GROUPS:
                backend = root / arm / group / 'backend'
                summary = read(backend / group / 'evaluation_summary.json')
                results[arm]['groups'][group] = dict(new_training_this_attempt=arm == 'continued',
                    exposure=read(backend / group / 'training/finished.json'),
                    summary={key: value for key, value in summary.items() if key != 'results'},
                    selection=read(backend / 'candidate_selection.json'))
        write(root / 'finished.json', dict(arms=results, utc=time.time(), test22=False,
            new_frontend_updates=0, research_decision='Return to real Agent; no automatic adoption or extra stage'))
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
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.action](args.root)


if __name__ == '__main__':
    main()
