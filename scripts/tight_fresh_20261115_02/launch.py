#!/usr/bin/env python3
"""One reviewed held-focused cycle after completed round11; preview by default."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

BUNDLE=Path(__file__).resolve().parent
TREE=BUNDLE.parents[1]
spec=importlib.util.spec_from_file_location('held_followup_helpers',BUNDLE.parent/'tight_fresh_20261115_01/launch.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
read,write,sha,require=common.read,common.write,common.sha,common.require


def prepare(tree,expected):
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==expected,
        'Frozen HEAD differs from requested source')
    config=read(BUNDLE/'launch_config.json');loop=read(BUNDLE/'loop_config.json')
    require(config['hours']==6 and len(loop['rounds'])==1,'This is one finite six-hour cycle')
    loop['inputs']={key:str((Path(value) if Path(value).is_absolute() else BUNDLE/value).resolve(strict=True))
        for key,value in loop['inputs'].items()}
    sys.path.insert(0,str(tree))
    from tools.tight_fresh_research_loop import validate_configuration
    validate_configuration(loop)
    return config,loop


def launch(tree,expected,config,loop):
    require(tree==TREE and not subprocess.check_output(['git','status','--porcelain','--untracked-files=normal',
        '--','tools','visuotactile','scripts'],cwd=tree,text=True).strip(),'Run from the clean frozen source tree')
    root=Path(config['output']);ledger=root.with_name(root.name+'_launch')
    require(not root.exists() and not ledger.exists(),'Preserve existing session and launch attempts')
    old=Path(config['old_session']);source=Path(config['source'])
    require(read(source.with_name(source.name+'_stages')/'finished.json')['software_complete']
        and read(source/'development/result.json')['software_complete'],'Previous cycle is incomplete')
    previous_monitor=read(old/'monitor.json');exited=read(old/'monitor_exit.json')
    require((old/'STOP_MONITOR').exists() and exited['pid']==previous_monitor['pid']
        and exited['ended']>=previous_monitor['started'],'Previous monitor has not closed')
    for key in ('old_owner','old_monitor'):
        state=common.unit_state(config[key])['properties']
        require(state['ActiveState'] in ('inactive','failed') and state['MainPID']=='0','Previous service is active')
    for key in ('owner_unit','monitor_unit'):
        require(common.unit_state(config[key])['properties']['LoadState']=='not-found','Preserve existing service identity')
    require(shutil.disk_usage(root.parent).free>=loop['bounds']['minimum_free_bytes'],'Insufficient registered disk reserve')
    ledger.mkdir();python=str(tree/'.venv_isaacsim/bin/python')
    state=dict(started=time.time(),source_commit=expected,worker_tree=str(tree),configuration=config,
        old_session_sha256=sha(old/'session.json'),old_deadline=read(old/'session.json')['deadline'],
        bundle_sha256={path.name:sha(path) for path in sorted(BUNDLE.iterdir()) if path.suffix in ('.py','.json')},
        launch_complete=False,new_session_not_old_extension=True)
    write(ledger/'status.json',state);submitted=False
    try:
        followup=[python,'-m','tools.tight_fresh_followup_session','--source-root',config['source'],
            '--data-source-root',config['data_source'],'--evaluation',str(source/'development'),
            '--output',str(root),'--hours',str(config['hours']),'--allow-illegal-starts']
        for role,path in config['role_sources'].items():followup+=['--role-source',role+'='+path]
        if config.get('inherit_absolute_deadline'):
            followup+=['--absolute-deadline',str(read(old/'session.json')['deadline']),'--resource-parent',str(old)]
        if config.get('reserve_gib') is not None:
            followup+=['--reserve-gib',str(config['reserve_gib'])]
        common.execute('followup',followup,tree,ledger,timeout=600)
        session=read(root/'session.json');deadline=session['deadline']
        require((deadline==read(old/'session.json')['deadline'] if config.get('inherit_absolute_deadline') else deadline==session['started_utc']+21600) and sha(old/'session.json')==state['old_session_sha256'],
            'Independent clock or preserved previous session differs')
        state.update(deadline=deadline,session_sha256=sha(root/'session.json'),
            inherited_absolute_deadline=config.get('inherit_absolute_deadline',False))
        write(ledger/'loop_config.json',loop);write(ledger/'status.json',state)
        base=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(tree),
            '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=30',
            '--property=LimitNOFILE=65536','--setenv=OMP_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2']
        common.execute('monitor',base+['--unit='+config['monitor_unit'],
            '--property=RuntimeMaxSec='+str(math.floor(deadline-time.time())),
            python,'-m','visuotactile.state_resources','--root',str(root)],tree,ledger,timeout=30)
        def observed_monitor():
            path=root/'shared_resources.json'
            if not path.exists():return None
            value=read(path)
            return value if time.time()-value['utc_epoch']<8 and (root/'monitor.json').exists() else None
        state['monitor_observed']=common.wait_started(observed_monitor,45,'Independent session monitor')
        if config.get('throughput_preflight'):
            preflight=(BUNDLE/config['throughput_preflight']).resolve(strict=True)
            common.execute('throughput_preflight',[python,str(preflight),'--root',str(root),
                '--loop',str(ledger/'loop_config.json')],tree,ledger,
                timeout=min(2400,max(1,deadline-time.time()-loop['bounds']['minimum_round_remaining_seconds'])))
            loop=read(ledger/'loop_config.json')
        common.execute('prepare_loop',[python,'-m','tools.tight_fresh_research_loop','prepare',
            '--session-root',str(root),'--output',str(root/'research_loop'),'--worker-tree',str(tree),
            '--configuration',str(ledger/'loop_config.json')],tree,ledger,timeout=600)
        require(read(root/'research_loop/registration.json')['deadline']==deadline,'Loop deadline differs')
        submitted=True
        common.execute('owner',base+['--unit='+config['owner_unit'],
            '--property=RuntimeMaxSec='+str(math.floor(deadline-time.time())),
            '--property=ExecStopPost=/usr/bin/touch '+str(root/'STOP_MONITOR'),
            python,'-m','tools.tight_fresh_research_loop','run','--registration',str(root/'research_loop/registration.json')],
            tree,ledger,timeout=30)
        path=root/'research_loop/owner_started.json'
        state['owner_observed']=common.wait_started(lambda:read(path) if path.exists() else None,30,'Held follow-up owner')
        state.update(launch_complete=True,owner_unit=common.unit_state(config['owner_unit']),
            monitor_unit=common.unit_state(config['monitor_unit']),interpretation='Startup observed only; actual stage results remain authoritative')
    except BaseException as error:
        state['error']=repr(error)
        if root.exists() and not submitted:(root/'STOP_MONITOR').touch(exist_ok=True)
        raise
    finally:
        state['ended']=time.time();write(ledger/'status.json',state)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-tree',type=Path,required=True);parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    tree=args.worker_tree.resolve(strict=True);config,loop=prepare(tree,args.expected_commit)
    print(json.dumps(launch(tree,args.expected_commit,config,loop) if args.launch else dict(configuration=config,loop=loop,started=False),indent=2))
