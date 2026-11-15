"""One explicit recovery of the unstarted round16 evaluation, without training.

Uses the original frozen workers and exact prepared plan. Publishes only a real
successful evaluation receipt, then lets the original coordinator reuse all
completed stages and automatically continue round17. No bad attempt is erased.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import time

from tools.tight_fresh_cycle import validate_unstarted_evaluation
from tools.tight_fresh_collect_cycle_owner import run_stage
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261116_01')
TREE=Path('/home/CNS2025331827/docs/bc+rl_RSS_fresh_cycles_20261116')
PYTHON=str(TREE/'.venv_isaacsim/bin/python')
read=lambda p:json.loads(p.read_text())


def run():
    candidate=ROOT/'round_16';stages=ROOT/'round_16_stages';development=candidate/'development'
    ledger=ROOT/'night/prepared_evaluation_recovery';ledger.mkdir(exist_ok=False)
    deadline=min(read(ROOT/'session.json')['deadline'],read(ROOT/'night_configuration.json')['absolute_wall_deadline'])-120
    assert deadline-time.time()>1800
    assert read(ROOT/'night/state.json')['stage']=='safe_failure'
    assert (ROOT/'monitor_exit.json').exists()
    for path in stages.glob('*_worker.json'):
        binding=read(path);p=Path('/proc',str(binding['pid']),'stat')
        assert not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=binding['proc_start_ticks']
    for name in ('monitor.json','monitor_exit.json','shared_resources.json','night/state.json',
                 'round_16_stages/failure.json','round_16_stages/status.json'):
        atomic_json(ledger/('prior_'+name.replace('/','_')),read(ROOT/name))
    validate_unstarted_evaluation(development,'development')
    prepared=read(stages/'development_prepare.json')
    assert prepared['returncode']==0 and sha256(development/'plan.json')==prepared['completion_sha256']
    frozen=read(stages/'registration.json')['source_sha256']
    assert all(sha256(TREE/p)==digest for p,digest in frozen.items())
    before={str(p):sha256(p) for role in 'AB' for p in
            (candidate/role/'training/finished.json',candidate/role/'backend'/role/'training/finished.json')}
    atomic_json(ledger/'registration.json',dict(utc=time.time(),deadline=deadline,controller_sha256=sha256(__file__),
        frozen_worker_tree=str(TREE),training_completion_hashes=before,prepared_plan_sha256=prepared['completion_sha256'],
        reason='Resume guard misclassified a prepared but never-started physical evaluation. No simulation attempt or optimizer update is repeated.'))
    (ROOT/'STOP_MONITOR').unlink()
    subprocess.run(['systemd-run','--user','--unit=tight-fresh-night-recovery-monitor-20261116-01',
        '--service-type=exec','--property=WorkingDirectory='+str(TREE),'--property=Restart=no',
        '--property=RuntimeMaxSec='+str(math.floor(deadline-time.time()+120)),
        PYTHON,'-m','visuotactile.state_resources','--root',str(ROOT)],check=True)
    until=time.time()+35
    while time.time()-read(ROOT/'shared_resources.json')['utc_epoch']>8:
        if time.time()>until:raise RuntimeError('Recovery resource monitor did not start')
        time.sleep(.5)
    context=dict(ledger=ledger,paths=dict(worker_tree=TREE,collection=ROOT/'new_family_16',output=candidate))
    command=[PYTHON,'-m','tools.tight_fresh_batch_run','run','--output',str(development)]
    receipt=run_stage('development_recovery',command,context,deadline,(ROOT/'STOP',ledger/'STOP'))
    assert receipt['complete'] and read(development/'result.json')['software_complete']
    assert not (stages/'development.json').exists()
    atomic_json(stages/'development.json',dict(stage='development',command=command,returncode=0,
        started=receipt['started'],seconds=receipt['seconds'],stop_reason=None,
        completion=str(development/'result.json'),completion_sha256=sha256(development/'result.json'),utc=time.time(),
        explicit_recovery=str(ledger),worker_source_unchanged=True))
    close=run_stage('close_completed_cycle',read(ROOT/'night/cycle_16_retry.json')['command'],context,
        deadline,(ROOT/'STOP',ledger/'STOP'))
    assert close['complete'] and read(stages/'finished.json')['software_complete']
    assert all(sha256(p)==digest for p,digest in before.items())
    atomic_json(ledger/'finished.json',dict(complete=True,utc=time.time(),unchanged_training_completions=before,
        action='Real unstarted evaluation completed; original frozen coordinator reused all completed stages. Automatic original owner continuation follows; unattended review remains unreleased.'))
    os.chdir(TREE)
    os.execv(PYTHON,[PYTHON,'-m','tools.tight_fresh_night_owner','--root',str(ROOT),
                  '--configuration',str(ROOT/'night_configuration.json')])


if __name__=='__main__':
    try:run()
    except BaseException as error:
        atomic_json(ROOT/'night/prepared_evaluation_recovery/error.json',dict(error=repr(error),utc=time.time()))
        (ROOT/'STOP_MONITOR').touch(exist_ok=True)
        raise
