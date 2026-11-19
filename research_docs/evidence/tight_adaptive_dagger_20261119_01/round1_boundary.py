"""Continue the existing training owner, then stop at an evidence-based R2 boundary.

No source/optimizer/physics mutation. The prior controller is suspended alone;
its existing independent tight_project training owner runs to completion.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import psutil
from tools import tight_adaptive_dagger as experiment
from tools.tight_frozen_frontend import read, write

ROOT = experiment.ROOT
PUBLIC = experiment.PUBLIC
OWNER = ROOT / 'owners/r1_train'
TARGET = ROOT / 'round01/learning'
RECEIPT = ROOT / 'round1_boundary_started.json'


def handover(pid):
    if RECEIPT.exists():
        raise FileExistsError('Do not repeat a controller handover')
    controller = psutil.Process(pid)
    assert controller.cwd() == str(experiment.PROJECT)
    assert controller.cmdline()[-2:] == ['tools.tight_adaptive_dagger', 'recover-admission']
    assert read(ROOT / 'progress.json')['stage'] == 'train'
    owner_pid = read(OWNER / 'project_job_started.json')['pid']
    owner = psutil.Process(owner_pid)
    assert owner.ppid() == pid and owner.cwd() == controller.cwd()
    assert not (OWNER / 'project_complete.json').exists()
    write(RECEIPT, dict(pid=os.getpid(), prior_controller_pid=pid,
        prior_controller_created=controller.create_time(), existing_owner_pid=owner_pid,
        existing_owner_created=owner.create_time(), utc=time.time(),
        purpose='Finish unchanged R1 training and evaluation, then diagnose before adapting R2',
        scope='Only suspend coordinator; existing training owner, worker and monitor keep running',
        original_deadline=read(ROOT / 'session.json')['deadline']))
    controller.send_signal(signal.SIGSTOP)
    until = time.monotonic() + 2
    while controller.status() != psutil.STATUS_STOPPED and time.monotonic() < until:
        time.sleep(.01)
    assert controller.status() == psutil.STATUS_STOPPED
    write(ROOT / 'coordinator_handover.json', read(RECEIPT))
    print('Prior coordinator suspended; existing training owner remains active', flush=True)


def await_owner():
    receipt = read(RECEIPT)
    deadline = receipt['original_deadline']
    while not (OWNER / 'project_complete.json').exists():
        if (OWNER / 'project_failure.json').exists():
            raise RuntimeError('Existing R1 owner failed: ' + str(read(OWNER / 'project_failure.json')))
        if time.time() >= deadline:
            raise TimeoutError('Original finite session deadline reached')
        process = psutil.Process(receipt['existing_owner_pid'])
        assert process.create_time() == receipt['existing_owner_created']
        assert process.status() != psutil.STATUS_ZOMBIE
        write(ROOT / 'round1_boundary_progress.json', dict(stage='waiting_existing_train',
            existing_owner=process.pid, utc=time.time()))
        time.sleep(20)
    # Retire only the suspended coordinator, after its job has a durable success receipt.
    try:
        coordinator = psutil.Process(receipt['prior_controller_pid'])
        assert coordinator.create_time() == receipt['prior_controller_created']
        coordinator.send_signal(signal.SIGTERM)
        coordinator.send_signal(signal.SIGCONT)
    except psutil.NoSuchProcess:
        pass
    write(ROOT / 'coordinator_handover_complete.json', dict(**receipt,
        existing_training_complete=read(OWNER / 'project_complete.json'),
        retired_utc=time.time(), optimizer_updates_repeated=0, physical_opportunities_repeated=0))


def run():
    await_owner()
    write(ROOT / 'progress.json', dict(round=1, stage='exposure_audit', utc=time.time()))
    experiment.audit_training(TARGET, 1)
    write(ROOT / 'progress.json', dict(round=1, stage='evaluate', utc=time.time()))
    if (ROOT / 'round01/evaluation').exists():
        raise FileExistsError('Evaluation already attempted; recover its uncompleted opportunities explicitly')
    result = experiment.evaluate(experiment.BASE, TARGET, 1)
    write(ROOT / 'round01/complete.json', dict(collection=str(ROOT / 'round01/collection_combined'),
        learning=str(TARGET), evaluation=True, utc=time.time()))
    write(ROOT / 'progress.json', dict(round=1, stage='round1_review', utc=time.time()))
    write(ROOT / 'round1_review_ready.json', dict(utc=time.time(),
        cases=len(result['rows']), training='All four selected models passed independent exposure replay',
        next_action='Inspect R1 failures and full offline errors; preregister adapted R2 collection',
        final_test_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-controller', type=int)
    parser.add_argument('--continue-existing', action='store_true')
    args = parser.parse_args()
    if args.prior_controller:
        handover(args.prior_controller)
    elif not args.continue_existing:
        parser.error('Explicit existing owner required')
    try:
        run()
    except BaseException as error:
        write(ROOT / 'round1_boundary_failure.json', dict(error=repr(error), utc=time.time()))
        raise
