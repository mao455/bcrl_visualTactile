"""Finish the existing owner, independently check raw masks, then evaluate.

Only the waiting coordinator is replaced. The running training owner is kept.
Every evaluation lane records real video so later representative selection does
not require repeating a physical opportunity or inventing missing images.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import time

import psutil

from tools import tight_adaptive_dagger as experiment
from tools.tight_frozen_frontend import read, write

ROOT=experiment.ROOT
PUBLIC=experiment.PUBLIC
TARGET=ROOT/'round01/learning'


def load(name):
    spec=importlib.util.spec_from_file_location(name,PUBLIC/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def evaluate_with_videos():
    assert read(TARGET/'exposure_acceptance.json')['passed']
    root,waves=experiment.evaluation_plans(experiment.BASE,TARGET,1)
    for wave in waves:
        plan=read(wave/'plan.json')
        for case in plan['cases']:case['video']=True
        plan['video_policy']='Every actual complete lane; select representative outcomes after evaluation without repeating physics'
        write(wave/'plan.json',plan)
    stages=[dict(entry='evaluate',args=['--output',str(wave)]) for wave in waves]
    experiment.execute('evaluate','r1_evaluate',stages,simulation=True)
    return experiment.evaluation_report(root,1)


def run():
    load('round1_boundary').await_owner()
    write(ROOT/'progress.json',dict(round=1,stage='exposure_audit',utc=time.time()))
    experiment.audit_training(TARGET,1)
    masks=load('audit_backend_masks')
    for role in 'AB':
        masks.audit(TARGET,role,PUBLIC/f'round01/{role}_backend_raw_mask_acceptance.json')
    load('exposure_phases').export(1)
    write(ROOT/'progress.json',dict(round=1,stage='evaluate',utc=time.time()))
    if (ROOT/'round01/evaluation').exists():
        raise FileExistsError('Preserve attempted evaluation; explicitly recover incomplete opportunities')
    result=evaluate_with_videos()
    write(ROOT/'round01/complete.json',dict(collection=str(ROOT/'round01/collection_combined'),
        learning=str(TARGET),evaluation=True,utc=time.time()))
    write(ROOT/'progress.json',dict(round=1,stage='round1_diagnosis',utc=time.time()))
    analysis=load('analyze_round');analysis.forks(1)
    registration=read(ROOT/'round01/evaluation/registration.json')
    load('execution_audit').audit([Path(p) for p in registration['waves']],PUBLIC/'round01/evaluation/execution_audit')
    analysis.offline(TARGET,PUBLIC/'round01/offline')
    write(ROOT/'progress.json',dict(round=1,stage='round1_review',utc=time.time()))
    write(ROOT/'round1_review_ready.json',dict(utc=time.time(),cases=len(result['rows']),
        training='Four selected models passed committed-step replay; both backends independently rebuilt from raw masks and seeded windows',
        next_action='Inspect R1 failures and complete offline errors; preregister adapted R2 collection',
        final_test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--existing-waiter',type=int)
    p.add_argument('--continue-existing',action='store_true')
    a=p.parse_args();receipt=ROOT/'verified_boundary_handover.json'
    if a.continue_existing:
        assert a.existing_waiter is None and receipt.exists()
        assert read(ROOT/'progress.json')['stage']=='train'
        assert not (ROOT/'round01/evaluation').exists()
        os.environ.setdefault('RESEARCH_CPUS','8-23')
        try:run()
        except BaseException as error:
            write(ROOT/'round1_boundary_failure.json',dict(error=repr(error),utc=time.time(),coordinator='verified_boundary'))
            raise
        raise SystemExit(0)
    assert a.existing_waiter is not None
    if receipt.exists():raise FileExistsError('Do not repeat a coordinator handover')
    process=psutil.Process(a.existing_waiter)
    assert process.cwd()==str(experiment.PROJECT)
    assert process.cmdline()[-2:]==['research_docs/evidence/tight_adaptive_dagger_20261119_01/round1_boundary.py','--continue-existing']
    assert read(ROOT/'progress.json')['stage']=='train'
    assert not (ROOT/'owners/r1_train/project_complete.json').exists()
    assert not (ROOT/'round01/evaluation').exists()
    owner=read(ROOT/'round1_boundary_started.json')
    active=psutil.Process(owner['existing_owner_pid'])
    assert active.create_time()==owner['existing_owner_created']
    assert active.status()!=psutil.STATUS_ZOMBIE
    write(receipt,dict(pid=os.getpid(),prior_waiter_pid=process.pid,
        prior_waiter_created=process.create_time(),existing_owner_pid=active.pid,
        existing_owner_created=active.create_time(),utc=time.time(),
        reason='Independent raw-mask acceptance before evaluation, full actual videos and complete R1 diagnosis',
        training_owner_unchanged=True,original_deadline=read(ROOT/'session.json')['deadline']))
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=5)
    try:
        run()
    except BaseException as error:
        write(ROOT/'round1_boundary_failure.json',dict(error=repr(error),utc=time.time(),coordinator='verified_boundary'))
        raise
