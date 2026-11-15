#!/usr/bin/env python3
"""Reuse the completed collection; launch only its registered training/evaluation.

No recollection, session-clock reset or automatic subsequent cohort. The old
collection source and failed owners remain immutable. Preview is the default.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import time

BUNDLE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cycle_recovery_helpers',BUNDLE/'recover_rgb01.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
common=helpers.common
read,sha,require,write=common.read,common.sha,common.require,common.write
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261115_01')
OLD=ROOT/'research_loop_rgb01/round_11/collect_cycle/registration.json'
OLD_SHA='fb3b3deae58c48fb3030ff033f3bf0f85f0b5a0f3b64d09b5d3315f3017bf3b8'
SESSION_SHA='73e5382bdc625d15b4591c12144a2e1680b76f6c0611558f342999975af35570'
DEADLINE=1794725918.9665225
LEDGER=ROOT/'recovery_cycle01'
OWNER='tight-fresh-cycle-20261115-01-recovery01.service'
MONITOR='tight-fresh-monitor-20261115-01-cycle01.service'


def prepare(tree,expected):
    sys.path.insert(0,str(tree))
    from tools.tight_fresh_collect_cycle_owner import validate,collection_admission
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==expected,
        'Explicit frozen source commit differs')
    require(sha(OLD)==OLD_SHA and sha(ROOT/'session.json')==SESSION_SHA
        and read(ROOT/'session.json')['deadline']==DEADLINE,'Original registration or session changed')
    context=validate(OLD,fresh=False)
    admission=collection_admission(context,time.time())
    require(admission['admitted'],'Completed collection is not admitted: '+str(admission))
    require(read(OLD.parent/'owner_finished.json').get('cycle_state')=='not_started',
        'This recovery is only for a cycle that never started')
    output=context['paths']['output']
    require(not output.exists() and not output.with_name(output.name+'_stages').exists(),
        'Preserve existing cycle outputs')
    require(not (ROOT/'STOP').exists() and not (OLD.parent/'STOP').exists(), 'An explicit STOP remains set')
    command=list(context['registration']['cycle_command'])
    command[0]=str(tree/'.venv_isaacsim/bin/python')
    return dict(version='completed_collection_cycle_recovery_v1',source_commit=expected,
        worker_tree=str(tree),original_deadline=DEADLINE,original_registration_sha256=OLD_SHA,
        session_sha256=SESSION_SHA,collection_admission=admission,cycle_command=command,
        collection_source_commit=context['registration']['worker_commit'],
        collection_repeated=False,subsequent_cohorts_started=False)


def launch(state,tree):
    require(Path(__file__).resolve()==tree/'scripts/tight_fresh_20261115_01/recover_cycle01.py',
        'Execute the committed frozen copy')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=normal','--',
        'tools','visuotactile','scripts/tight_fresh_20261115_01'],cwd=tree,text=True).strip(),
        'Frozen recovery source is dirty')
    require(not LEDGER.exists(),'Preserve the previous recovery attempt')
    for unit in ('tight-fresh-research-loop-20261115-01-rgb01.service',
                 'tight-fresh-monitor-20261115-01-rgb01.service'):
        p=common.unit_state(unit)['properties']
        require(p['ActiveState'] in ('inactive','failed') and p['MainPID']=='0','Old service remains active: '+unit)
    for unit in (OWNER,MONITOR):
        require(common.unit_state(unit)['properties']['LoadState']=='not-found','Preserve existing recovery service: '+unit)
    monitor=read(ROOT/'monitor.json');ended=read(ROOT/'monitor_exit.json')
    require((ROOT/'STOP_MONITOR').exists() and ended['pid']==monitor['pid']
        and ended['ended']>=monitor['started'],'Old resource monitor has not closed')
    LEDGER.mkdir();state.update(started=time.time(),launch_complete=False)
    write(LEDGER/'registration.json',state);submitted=False
    try:
        state['archived_monitor']=helpers.archive_monitor(ROOT,LEDGER)
        runtime=math.floor(DEADLINE-time.time())
        base=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(tree),
            '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=30',
            '--property=LimitNOFILE=65536','--setenv=OMP_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2']
        began=time.time()
        common.execute('monitor',base+['--unit='+MONITOR,'--property=RuntimeMaxSec='+str(runtime),
            str(tree/'.venv_isaacsim/bin/python'),'-m','visuotactile.state_resources','--root',str(ROOT)],tree,LEDGER,timeout=30)
        def fresh_monitor():
            identity=read(ROOT/'monitor.json');sample=read(ROOT/'shared_resources.json')
            return dict(identity=identity,sample=sample) if identity.get('started',0)>=began and sample.get('utc_epoch',0)>=began else None
        observed=common.wait_started(fresh_monitor,45,'New cycle monitor')
        previous=read(LEDGER/'previous_monitor/shared_resources.json')['utc_epoch']
        write(LEDGER/'monitor_observed.json',dict(**observed,unobserved_gap_seconds=observed['sample']['utc_epoch']-previous,
            gap_is_unobserved_not_zero_gpu=True))
        submitted=True
        common.execute('cycle',base+['--unit='+OWNER,
            '--property=RuntimeMaxSec='+str(math.floor(DEADLINE-time.time())-60),
            '--property=ExecStopPost=/usr/bin/touch '+str(ROOT/'STOP_MONITOR')]+state['cycle_command'],tree,LEDGER,timeout=30)
        stage=ROOT/'round_11_stages/registration.json'
        state['cycle_started']=common.wait_started(lambda:read(stage) if stage.exists() else None,45,'Recovered cycle')
        state.update(launch_complete=True,owner_unit=common.unit_state(OWNER),monitor_unit=common.unit_state(MONITOR),
            interpretation='Cycle startup only. Actual training and evaluation completion require their stage receipts.')
    except BaseException as error:
        state['error']=repr(error)
        if not submitted:(ROOT/'STOP_MONITOR').touch(exist_ok=True)
        raise
    finally:
        state['ended']=time.time();write(LEDGER/'status.json',state)
    return state


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-tree',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();tree=args.worker_tree.resolve(strict=True)
    state=prepare(tree,args.expected_commit)
    print(json.dumps(launch(state,tree) if args.launch else state,indent=2))


if __name__=='__main__':main()
