#!/usr/bin/env python3
"""Preview by default; one explicit RGB recovery within the original session clock."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

BUNDLE=Path(__file__).resolve().parent
RELATIVE=Path('scripts/tight_fresh_20261115_01')
spec=importlib.util.spec_from_file_location('rgb01_launch_helpers',BUNDLE/'launch.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
read,sha,require,write=common.read,common.sha,common.require,common.write


def load(tree,expected):
    require(re.fullmatch('[0-9a-f]{40}',expected) is not None,'Explicit full frozen commit SHA required')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==expected,'Frozen HEAD differs')
    for name in ('recover_rgb01.py','recovery_rgb01.json','launch.py'):
        require(sha(BUNDLE/name)==sha(tree/RELATIVE/name),'Invoked and frozen recovery files differ: '+name)
    config=read(BUNDLE/'recovery_rgb01.json');root=Path(config['session_root']).resolve(strict=True)
    require(config['version']=='same_session_rgb_recovery_v1','Unknown recovery identity')
    session=read(root/'session.json')
    require(session['deadline']==config['deadline']==1794725918.9665225
        and sha(root/'session.json')==config['session_sha256'],'Original session or deadline changed')
    require(sha(root/'research_loop/registration.json')==config['previous_registration_sha256'],'Original registration changed')
    old=read(root/'research_loop/registration.json');loop=read(root/'launcher/loop_config.json')
    require(loop==old['configuration'],'Original expanded loop configuration differs')
    require(old['deadline']==config['deadline'],'Original loop deadline differs')
    result=copy.deepcopy(loop);mapping=config['round_mapping']
    require([(v['old_index'],v['new_index']) for v in mapping]==[(10,11),(11,12),(12,13)],'Wrong recovery cohort mapping')
    for row,replacement in zip(result['rounds'],mapping):
        require(row['index']==replacement['old_index'],'Original round ordering differs')
        row['index']=replacement['new_index'];row['registry_id']=replacement['registry_id']
    require(len(result['rounds'])==len(mapping)==3,'Preserve all three registered cohorts')
    return config,result


def quiescent(config):
    root=Path(config['session_root']);states={}
    for key in ('old_owner_unit','old_monitor_unit'):
        state=common.unit_state(config[key]);properties=state['properties']
        require(properties.get('ActiveState') in ('inactive','failed') and properties.get('MainPID')=='0',
            'Previous service remains active: '+config[key]);states[key]=state
    owner=read(root/'research_loop/owner_finished.json');worker=read(root/'new_family_10/owner_finished.json')
    require(owner.get('software_complete') is False and owner.get('completed_rounds')==0
        and worker.get('software_complete') is False and worker.get('returncode')==1,
        'Expected preserved first collection software failure is missing')
    monitor=read(root/'monitor.json');ended=read(root/'monitor_exit.json')
    require((root/'STOP_MONITOR').is_file() and ended.get('pid')==monitor.get('pid')
        and ended.get('ended',0)>=monitor.get('started',math.inf),'Previous monitor has not closed')
    require(not (root/'STOP').exists(),'Session STOP remains set; do not clear it automatically')
    live=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            if directory.stat().st_uid!=os.getuid():continue
            if directory.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            argv=directory.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            for index,argument in enumerate(argv):
                for flag in ('--root','--output','--session-root','--resource-root','--registration'):
                    value=(argv[index+1] if argument==flag and index+1<len(argv) else
                        argument[len(flag)+1:] if argument.startswith(flag+'=') else None)
                    if value is None:continue
                    path=Path(value)
                    if not path.is_absolute():path=directory.joinpath('cwd').resolve(strict=True)/path
                    path=path.resolve()
                    if path==root or path.is_relative_to(root):live.append(dict(pid=int(directory.name),argv=argv))
        except (FileNotFoundError,ProcessLookupError):continue
    require(not live,'Original-session workers still live: '+json.dumps(live))
    return dict(units=states,owner=owner,worker=worker,monitor=monitor,monitor_exit=ended,live_workers=live,signals_sent=0)


def commands(config,tree,runtime):
    root=Path(config['session_root']);ledger=root/config['ledger'];loop=root/config['loop_ledger']
    python=str(tree/'.venv_isaacsim/bin/python')
    systemd=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(tree),
        '--property=RuntimeMaxSec='+str(runtime),'--property=Restart=no','--property=KillMode=control-group',
        '--property=TimeoutStopSec=30','--property=LimitNOFILE=65536','--setenv=OMP_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2']
    return dict(monitor=systemd+['--unit='+config['monitor_unit'],python,'-m','visuotactile.state_resources','--root',str(root)],
        loop_prepare=[python,'-m','tools.tight_fresh_research_loop','prepare','--session-root',str(root),
            '--output',str(loop),'--worker-tree',str(tree),'--configuration',str(ledger/'loop_config.json')],
        owner=systemd+['--unit='+config['owner_unit'],'--property=ExecStopPost=/usr/bin/touch '+str(root/'STOP_MONITOR'),
            python,'-m','tools.tight_fresh_research_loop','run','--registration',str(loop/'registration.json')])


def archive_monitor(root,ledger):
    archive=ledger/'previous_monitor';archive.mkdir();files={}
    # Keep append-only logs at their original paths for restore_telemetry.
    # Snapshot both the old summaries and complete short first-attempt logs.
    for name in ('monitor.json','monitor_exit.json','shared_resources.json','STOP_MONITOR',
            'resources/resources.json','resources/resources.jsonl','resources/shared.jsonl','resource_decisions.jsonl'):
        source=root/name;destination=archive/name;destination.parent.mkdir(parents=True,exist_ok=True)
        require(source.is_file(),'Previous telemetry evidence missing: '+name)
        shutil.copy2(source,destination)
        require(sha(source)==sha(destination),'Telemetry changed during recovery archive: '+name)
        files[name]=dict(sha256=sha(destination),bytes=destination.stat().st_size)
    write(archive/'archive.json',dict(archived_utc=time.time(),files=files,
        append_logs_retained_in_place=True,previous_last_shared_utc=read(root/'shared_resources.json')['utc_epoch']))
    # Clear only closed-monitor markers after their exact bytes are archived.
    (root/'monitor_exit.json').unlink();(root/'STOP_MONITOR').unlink()
    return files


def launch(config,loop,tree,expected):
    require(Path(__file__).resolve()==tree/RELATIVE/'recover_rgb01.py','Run from the new frozen tree')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal','--',
        'tools','visuotactile',str(RELATIVE)],cwd=tree,text=True)
    require(not dirty.strip(),'Frozen recovery source has uncommitted changes')
    root=Path(config['session_root']);ledger=root/config['ledger'];loop_root=root/config['loop_ledger']
    require(not ledger.exists() and not loop_root.exists(),'Single recovery attempt only; preserve existing output')
    predecessor=quiescent(config)
    for key in ('owner_unit','monitor_unit'):
        require(common.unit_state(config[key])['properties']['LoadState']=='not-found','Preserve existing recovery unit identity')
    require(shutil.disk_usage(root).free>=loop['bounds']['minimum_free_bytes'],'Insufficient registered disk admission reserve')
    runtime=math.floor(config['deadline']-time.time())
    require(runtime>loop['bounds']['minimum_round_remaining_seconds'],'Insufficient original-session time')
    ledger.mkdir();write(ledger/'loop_config.json',loop)
    state=dict(version=config['version'],started_utc=time.time(),complete=False,source_commit=expected,
        worker_tree=str(tree),session_sha256=config['session_sha256'],original_deadline=config['deadline'],
        predecessor=predecessor,round_mapping=config['round_mapping'],no_clock_reset=True,no_retry=True,
        input_configuration_source='launcher/loop_config.json; only round indices and registry IDs changed',
        bundle_sha256={name:sha(BUNDLE/name) for name in ('recover_rgb01.py','recovery_rgb01.json','launch.py')})
    write(ledger/'status.json',state);owner_submitted=False
    try:
        state['archived_monitor']=archive_monitor(root,ledger);write(ledger/'status.json',state)
        submitted=time.time();runtime=math.floor(config['deadline']-submitted)
        common.execute('monitor',commands(config,tree,runtime)['monitor'],tree,ledger,timeout=30)
        def telemetry():
            path=root/'shared_resources.json';identity=root/'monitor.json'
            if not path.exists() or not identity.exists():return None
            value=read(path);monitor=read(identity)
            return dict(sample=value,monitor=monitor) if (monitor.get('started',0)>=submitted
                and value.get('utc_epoch',0)>=submitted and time.time()-value['utc_epoch']<8) else None
        first=common.wait_started(telemetry,45,'Recovery monitor')
        old_sample=read(ledger/'previous_monitor/shared_resources.json')
        write(ledger/'monitor_observed.json',dict(**first,unit=common.unit_state(config['monitor_unit']),
            previous_last_shared_utc=old_sample['utc_epoch'],shared_sample_gap_seconds=first['sample']['utc_epoch']-old_sample['utc_epoch'],
            gap_is_unobserved_not_zero_gpu=True,history_logs_append=True))
        common.execute('loop_prepare',commands(config,tree,runtime)['loop_prepare'],tree,ledger,
            timeout=min(600,max(1,math.floor(config['deadline']-time.time()))))
        require(read(loop_root/'registration.json')['deadline']==config['deadline']
            and sha(root/'session.json')==config['session_sha256'],'Recovery altered original session/deadline')
        runtime=math.floor(config['deadline']-time.time())
        require(runtime>loop['bounds']['minimum_round_remaining_seconds'],'Insufficient remaining original-session time')
        owner_submitted=True
        common.execute('owner',commands(config,tree,runtime)['owner'],tree,ledger,timeout=30)
        observed=common.wait_started(lambda:read(loop_root/'owner_started.json') if (loop_root/'owner_started.json').exists() else None,30,'Recovery owner')
        state.update(complete=True,owner_started=observed,owner_unit=common.unit_state(config['owner_unit']),
            monitor_unit=common.unit_state(config['monitor_unit']),interpretation='Startup observed only; finite loop owns collection, training and evaluation.')
    except BaseException as error:
        state['error']=repr(error)
        if not owner_submitted:
            (root/'STOP_MONITOR').touch(exist_ok=True);state['new_monitor_stop_requested']=True
        else:state.update(owner_may_be_running=True,instruction='Inspect owned recovery service; do not retry this identity.')
        raise
    finally:
        state['ended_utc']=time.time();write(ledger/'status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker-tree',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();tree=args.worker_tree.resolve(strict=True);config,loop=load(tree,args.expected_commit)
    value=launch(config,loop,tree,args.expected_commit) if args.launch else dict(started=False,session_created=False,
        original_deadline=config['deadline'],remaining_seconds=math.floor(config['deadline']-time.time()),
        configuration=config,loop_configuration=loop,commands_preview=commands(config,tree,max(1,math.floor(config['deadline']-time.time()))),
        note='Preview only; actual launch verifies closed workers, archives old monitor and invokes existing loop prepare once.')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
