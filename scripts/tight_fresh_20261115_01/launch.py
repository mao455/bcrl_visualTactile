#!/usr/bin/env python3
"""Prepare/check by default; --launch starts the reviewed finite research loop.

No old process is signalled. This launcher neither creates a frozen checkout nor
retries an existing output. Execute the committed copy inside the supplied tree.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


BUNDLE = Path(__file__).resolve().parent
RELATIVE = Path('scripts/tight_fresh_20261115_01')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def unit_state(name):
    result = subprocess.run(['systemctl','--user','show',name,
        '--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus,ExecStart,RuntimeMaxUSec,ControlGroup'],
        text=True, capture_output=True, timeout=10)
    values = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    require(values.get('LoadState') in ('loaded','not-found'), 'Cannot inspect user service: '+name)
    return dict(unit=name,properties=values,returncode=result.returncode,stderr=result.stderr)


def no_old_work(config):
    old = Path(config['old_session']).resolve(strict=True)
    source = Path(config['source'])
    for path in (old/'collect_cycle09_owner/owner_finished.json',
                 source.with_name(source.name+'_stages')/'finished.json',
                 Path(config['evaluation'])/'result.json'):
        require(read(path).get('software_complete') is True, 'Incomplete source stage: '+str(path))
    monitor = read(old/'monitor.json'); finished = read(old/'monitor_exit.json')
    require((old/'STOP_MONITOR').is_file() and finished.get('pid')==monitor.get('pid')
            and finished.get('ended',0)>=monitor.get('started',math.inf),
            'Old monitor must explicitly stop and write its matching exit receipt first')
    state = unit_state(config['old_owner_unit'])
    require(state['properties'].get('ActiveState') in ('inactive','failed')
            and state['properties'].get('MainPID')=='0', 'Old owner service is still active')
    live = []
    # Read only our user's argv. Include --registration owners as well as
    # --root/--output workers; a historical PID by itself grants no ownership.
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            if directory.stat().st_uid!=os.getuid():continue
            if directory.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            argv=directory.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            bound=[]
            for index, argument in enumerate(argv):
                for flag in ('--root','--output','--source-root','--resource-root','--session-root','--registration'):
                    value=(argv[index+1] if argument==flag and index+1<len(argv)
                           else argument[len(flag)+1:] if argument.startswith(flag+'=') else None)
                    if value is not None:
                        path=Path(value)
                        if not path.is_absolute():path=directory.joinpath('cwd').resolve(strict=True)/path
                        path=path.resolve()
                        if path==old or path.is_relative_to(old):bound.append(str(path))
            if bound:live.append(dict(pid=int(directory.name),argv=argv,bound_paths=bound))
        except (FileNotFoundError,ProcessLookupError):continue
    require(not live, 'Old-root work still live: '+json.dumps(live))
    return dict(old_owner=state,monitor_exit=dict(path=str(old/'monitor_exit.json'),sha256=sha(old/'monitor_exit.json')),
                old_root_processes=live,checked_utc=time.time(),signals_sent=0)


def load(tree, expected):
    require(re.fullmatch('[0-9a-f]{40}',expected) is not None, 'Use the full frozen commit SHA')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
    require(head==expected, 'Frozen HEAD differs from the explicit launch commit')
    config=read(BUNDLE/'launch_config.json')
    require(config['version']=='tight_fresh_20261115_01_launch_v1'
            and config['hours']==12 and config['allow_illegal_starts'] is True,
            'Unexpected session identity or clock contract')
    bundle=tree/RELATIVE
    for path in sorted(BUNDLE.glob('*.json')):
        require(sha(path)==sha(bundle/path.name), 'Invoked and frozen static configuration differ: '+path.name)
    loop=read(bundle/config['loop_configuration'])
    loop['inputs']={key:str((Path(value) if Path(value).is_absolute() else bundle/value).resolve(strict=True))
                    for key,value in loop['inputs'].items()}
    # Do not import PyTorch merely to preview static argv. The existing loop
    # prepare performs authoritative schema/profile validation once at launch.
    require(set(loop)=={'inputs','bounds','collection','cycle','rounds'},'Unexpected loop configuration sections')
    for key,device in (('collection_backend_json','cuda:0'),('evaluation_backend_json','cpu')):
        backend=read(loop['inputs'][key])
        require(backend.get('version')=='tight_execution_backend_v2'
                and backend.get('physics_device')==device and backend.get('camera_backend')=='standard'
                and backend.get('kit_tasking_threads')==16 and backend.get('physics_threads')==2,
                'Wrong registered simulation backend')
    sampling=read(loop['inputs']['training_sampling_json'])
    require(sampling.get('version')=='fresh_random_visit_training_v1'
            and sampling.get('geometry',{}).get('fraction')==.5
            and sampling['geometry'].get('max_samples') is None,'Wrong training sampling candidate')
    require(loop['collection']['schedules']==['fixed80']*4+['held0']*2+['held_root60','contact0'], 'Schedule changed')
    require(loop['cycle']['evaluation_lanes']==loop['cycle']['development_count_per_role']==8,
            'This repeated development cohort has eight cases per role')
    require(loop['cycle']['continue_rear'] and loop['cycle']['reuse_source_visual_index'], 'Continuation or visual reuse omitted')
    return config,loop


def commands(config, tree, runtime):
    root=Path(config['output']);python=str(tree/'.venv_isaacsim/bin/python')
    ledger=root/'launcher';registration=root/'research_loop/registration.json'
    common=['systemd-run','--user','--service-type=exec',
        '--property=WorkingDirectory='+str(tree),'--property=RuntimeMaxSec='+str(runtime),
        '--property=Restart=no','--property=KillMode=control-group',
        '--property=TimeoutStopSec=30','--property=LimitNOFILE=65536',
        '--setenv=OMP_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2']
    return dict(
        followup=[python,'-m','tools.tight_fresh_followup_session','--source-root',config['source'],
            '--evaluation',config['evaluation'],'--output',str(root),'--hours','12','--allow-illegal-starts'],
        monitor=common+['--unit='+config['monitor_unit'],python,'-m','visuotactile.state_resources','--root',str(root)],
        loop_prepare=[python,'-m','tools.tight_fresh_research_loop','prepare','--session-root',str(root),
            '--output',str(root/'research_loop'),'--worker-tree',str(tree),'--configuration',str(ledger/'loop_config.json')],
        owner=common+['--unit='+config['owner_unit'],
            '--property=ExecStopPost=/usr/bin/touch '+str(root/'STOP_MONITOR'),
            python,'-m','tools.tight_fresh_research_loop','run','--registration',str(registration)])


def execute(name, argv, tree, ledger, *, timeout=600):
    receipt=dict(stage=name,argv=argv,cwd=str(tree),started_utc=time.time(),complete=False,timeout_seconds=timeout)
    write(ledger/(name+'_command.json'),receipt)
    try:
        with (ledger/(name+'.log')).open('x') as stream:
            result=subprocess.run(argv,cwd=tree,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        receipt.update(returncode=result.returncode,complete=result.returncode==0)
    except BaseException as error:
        receipt['error']=repr(error)
        raise
    finally:
        receipt['ended_utc']=time.time();write(ledger/(name+'_command.json'),receipt)
    require(receipt['complete'], name+' failed; preserve '+str(ledger/(name+'.log')))


def wait_started(test, seconds, label):
    until=time.monotonic()+seconds
    while time.monotonic()<until:
        value=test()
        if value:return value
        time.sleep(.5)
    raise TimeoutError(label+' startup was not observed within '+str(seconds)+' seconds')


def launch(config, loop, tree, expected):
    require(Path(__file__).resolve()==tree/RELATIVE/'launch.py', 'Launch the committed script from the frozen tree')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal','--',
        'tools','visuotactile',str(RELATIVE)],cwd=tree,text=True)
    require(not dirty.strip(),'Frozen launch source/configuration has uncommitted changes')
    root=Path(config['output']);attempt=root.with_name(root.name+'_launch')
    require(not root.exists() and not root.is_symlink() and not attempt.exists(), 'Existing session/launch identity cannot be retried')
    predecessor=no_old_work(config)
    require(shutil.disk_usage(root.parent).free>=loop['bounds']['minimum_free_bytes'], 'Less than the registered 110 GiB admission reserve')
    for name in (config['owner_unit'],config['monitor_unit']):
        require(unit_state(name)['properties']['LoadState']=='not-found','Preserve existing unit identity: '+name)
    attempt.mkdir()
    state=dict(version='finite_research_session_launch_v1',started_utc=time.time(),complete=False,
        source_commit=expected,worker_tree=str(tree),source=config['source'],output=str(root),
        predecessor=predecessor,bundle_sha256={p.name:sha(p) for p in sorted(BUNDLE.iterdir()) if p.suffix in ('.json','.py')},
        no_retry=True,old_process_signals_sent=0)
    write(attempt/'status.json',state)
    owner_submitted=False
    try:
        execute('followup',commands(config,tree,43200)['followup'],tree,attempt)
        session=read(root/'session.json');deadline=session['deadline']
        require(deadline==session['started_utc']+43200,'Follow-up did not create exactly one independent 12-hour window')
        require(session['followup']['source_root']==config['source'],'New session has the wrong source')
        ledger=root/'launcher';ledger.mkdir()
        write(ledger/'loop_config.json',loop)
        write(ledger/'launch_binding.json',dict(attempt=str(attempt),source_commit=expected,
            original_deadline=deadline,session_sha256=sha(root/'session.json'),static_config_sha256=state['bundle_sha256']))
        state.update(original_deadline=deadline,session_sha256=sha(root/'session.json'))
        write(attempt/'status.json',state)
        runtime=math.floor(deadline-time.time())
        require(runtime>loop['bounds']['minimum_round_remaining_seconds'],'Insufficient original session time')
        execute('monitor',commands(config,tree,runtime)['monitor'],tree,attempt,timeout=30)
        def telemetry():
            path=root/'shared_resources.json'
            if not path.is_file():return None
            row=read(path)
            return row if time.time()-row.get('utc_epoch',0)<8 else None
        first=wait_started(telemetry,45,'New monitor')
        write(attempt/'monitor_observed.json',dict(utc=time.time(),sample=first,unit=unit_state(config['monitor_unit'])))
        execute('loop_prepare',commands(config,tree,runtime)['loop_prepare'],tree,attempt,timeout=min(600,runtime))
        require(read(root/'research_loop/registration.json')['deadline']==deadline,'Loop registration changed the new deadline')
        runtime=math.floor(deadline-time.time())
        require(runtime>loop['bounds']['minimum_round_remaining_seconds'],'Insufficient remaining session time')
        owner_submitted=True
        execute('owner',commands(config,tree,runtime)['owner'],tree,attempt,timeout=30)
        def owner_started():
            path=root/'research_loop/owner_started.json'
            return read(path) if path.is_file() else None
        observed=wait_started(owner_started,30,'New owner')
        state.update(complete=True,owner_started=observed,owner_unit=unit_state(config['owner_unit']),
            monitor_unit=unit_state(config['monitor_unit']),
            interpretation='Startup observed only; future collection/training/evaluation remains owned by the finite loop')
    except BaseException as error:
        state['error']=repr(error)
        # Never signal the old monitor or any unrelated PID. Before owner
        # submission, only our new monitor is asked to end through its own flag.
        if root.exists() and not owner_submitted:
            (root/'STOP_MONITOR').touch(exist_ok=True)
            state['new_monitor_stop_requested']=True
        elif owner_submitted:
            state['owner_may_be_running']=True
            state['instruction']='Read the registered owner and systemd state; do not retry the launch.'
        raise
    finally:
        state['ended_utc']=time.time();write(attempt/'status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-tree',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--launch',action='store_true',help='Create the new session and submit its monitor and owner; absent means static preview only')
    args=parser.parse_args();tree=args.worker_tree.resolve(strict=True)
    config,loop=load(tree,args.expected_commit)
    value=(launch(config,loop,tree,args.expected_commit) if args.launch else
        dict(static_configuration_valid=True,started=False,session_created=False,
            source_completeness_and_quiescence_checked=False,configuration=config,loop_configuration=loop,
            validation_scope='Frozen bundle/paths and requested configuration; authoritative full schema validation remains in loop prepare',
            commands_preview=commands(config,tree,43200),
            runtime_note='Actual service RuntimeMaxSec is floor(original new deadline - submission time), never a reset clock'))
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
