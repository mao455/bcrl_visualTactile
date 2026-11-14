"""Own one preregistered collection followed by one explicitly chosen full cycle.

Validation is read-only. No retries, new resource window, monitor or model selection.
The registration's parent is this owner's exclusive ledger directory.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from tools.tight_batch_owner import (bind_worker_group, census, cleanup_owned_group,
    process_identity)
from tools.tight_fresh_stage_fit_owner import source_files
from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

VERSION = 'tight_collect_cycle_owner_v1'
CLEANUP_SECONDS = 60


def require(value, message):
    if not value:raise ValueError(message)


def command_paths(command, module, tree, expected):
    require(isinstance(command,list) and all(isinstance(x,str) for x in command)
        and len(command)>=3 and command[1:3]==['-m',module]
        and Path(command[0]).resolve(strict=True)==(tree/'.venv_isaacsim/bin/python').resolve(strict=True),
        'Explicit frozen Python module argv required: '+module)
    flags=[x.split('=',1)[0] for x in command[3:] if x.startswith('--')]
    require(len(flags)==len(set(flags)), 'Duplicate command options are ambiguous')
    for flag,path in expected.items():
        require(command.count(flag)==1 and command.index(flag)+1<len(command), 'Explicit command path required: '+flag)
        require(Path(command[command.index(flag)+1]).resolve()==path, 'Command path differs: '+flag)


def validate_plan(plan):
    from tools.tight_fresh_batch_run import validate_takeover_plan
    validate_takeover_plan(plan)


def validate(registration, *, fresh=True):
    registration=Path(registration).resolve(strict=True);r=read(registration)
    require(r.get('version')==VERSION,'Unknown collect-cycle registration')
    for key in ('collection_max_seconds','minimum_cycle_remaining_seconds','minimum_accepted'):
        require(type(r.get(key)) is int,'Explicit integer bound required: '+key)
    require(CLEANUP_SECONDS<r['collection_max_seconds']<=7200 and r['minimum_cycle_remaining_seconds']>=3600
        and r['minimum_accepted']>=8,'Collection or cycle admission budget was weakened')
    require(type(r.get('deadline')) in (int,float) and math.isfinite(r['deadline']),'Finite original deadline required')
    paths={k:Path(r[k]).resolve(strict=k not in ('output',)) for k in
        ('session_root','data_source','collection','output','worker_tree')}
    session_root=paths['session_root'];collection=paths['collection'];output=paths['output'];tree=paths['worker_tree']
    ledger=registration.parent
    require(all(p!=session_root and p.is_relative_to(session_root) for p in (collection,output,ledger)),
        'Collection, cycle and owner ledger must be original-session children')
    require(not any(a==b or a.is_relative_to(b) or b.is_relative_to(a)
        for a,b in ((ledger,collection),(ledger,output),(collection,output))), 'Separate immutable output identities required')
    session=read(session_root/'session.json');plan=read(collection/'plan.json')
    require(session['deadline']==r['deadline'] and sha256(session_root/'session.json')==r['session_sha256']
        and read(collection/'session.json')==session and read(paths['data_source']/'session.json')==session,
        'Original session/deadline changed')
    require(sha256(collection/'plan.json')==r['collection_plan_sha256']
        and plan.get('kind')=='takeover' and plan.get('test_used') is False
        and plan.get('batch',{}).get('version')=='whole_wave_complete_student_v1'
        and Path(plan['resource_root']).resolve()==session_root,'Prepared collection binding differs')
    validate_plan(plan)
    require(all(sha256(paths['data_source']/role/'state_data/data.json')==plan['source_data_sha256'][role]
        for role in ('A','B')),'Cycle data source must retain the complete collection-source population')
    sources=source_files(tree)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
    require(head==r['worker_commit']==plan['source_commit'] and sources==plan['source_sha256'],
        'Frozen collection/cycle source commit or bytes differ')
    command_paths(r['collection_command'],'tools.tight_batch_owner',tree,
        {'--output':collection,'--worker-tree':tree})
    command_paths(r['cycle_command'],'tools.tight_fresh_cycle',tree,
        {'--root':paths['data_source'],'--collection':collection,'--output':output})
    if fresh:
        require(not any((ledger/n).exists() for n in ('owner_started.json','owner_finished.json')),
            'No automatic owner restart')
        require(not output.exists() and not output.with_name(output.name+'_stages').exists(),'Preserve existing cycle outputs')
        require(not any((collection/n).exists() for n in ('owner_started.json','run_started.json','worker.log','result.json'))
            and not any((collection/c['name']).exists() for c in plan['cases']), 'Preserve existing collection attempts')
    inputs={}
    for command in (r['collection_command'],r['cycle_command']):
        for flag in ('--backend-json','--reuse-index','--training-sampling-json'):
            if flag in command:
                path=Path(command[command.index(flag)+1]).resolve(strict=True);inputs[str(path)]=sha256(path)
    return dict(registration=r,paths=paths,plan=plan,ledger=ledger,sources=sources,command_input_sha256=inputs,
        registration_sha256=sha256(registration))


def collection_admission(context, now):
    """No raw payload reread: retain the inner owner's verified capture bindings."""
    r=context['registration'];p=context['paths']['collection'];plan=context['plan']
    owner=read(p/'owner_finished.json');final=read(p/'result.json');rows=read(p/'results.json');entries=read(p/'correction_entries.json')
    counted=census(p,plan);by_name={row['case']:row for row in rows}
    require(owner.get('software_complete') is True and owner.get('returncode')==0
        and owner.get('plan_sha256')==r['collection_plan_sha256'] and final.get('software_complete') is True
        and not final.get('worker_error') and final.get('cases')==len(plan['cases'])==len(rows)
        and len(by_name)==len(rows) and set(by_name)=={c['name'] for c in plan['cases']}
        and all(row['state']=='complete' for row in counted),'Not every registered collection opportunity completed')
    accepted={}
    for case in plan['cases']:
        result=read(p/case['name']/'result.json')
        require(by_name[case['name']]['result']==result and by_name[case['name']]['role']==case['role'],
            'Collection row differs from preserved case result')
        require(all(result.get(k)==v for k,v in case['spec'].items()),'Completed case differs from its registered initial state')
        if result.get('accepted_for_training') is not True:continue
        identity=case['spec']['identity'];entry=entries[identity];pair=plan['pairs'][case['role']]
        capture=result.get('dataset_capture',{})
        require(result.get('legal') is True and result.get('success') is True
            and result.get('dagger',{}).get('expert_valid_ticks',0)>0
            and entry['episode']==identity and entry['split']=='train' and entry['family']==case['spec']['family']
            and entry['source_group']==case['role']
            and entry['frontend_sha256']==pair['frontend_sha256'] and entry['backend_sha256']==pair['backend_sha256']
            and capture.get('accepted_success') is True and capture.get('verified_source_fields')==185
            and capture.get('sha256')==entry['source_sha256'] and capture.get('path')==entry['path'],
            'Accepted capture lost actual success, expert actions or source policy binding')
        accepted[identity]=entry
    require(set(entries)==set(accepted),'Admission manifest differs from verified successful cases')
    remaining=r['deadline']-CLEANUP_SECONDS-now
    reason=('too_few_accepted' if len(accepted)<r['minimum_accepted'] else
        'insufficient_cycle_time' if remaining<r['minimum_cycle_remaining_seconds'] else 'admitted')
    return dict(admitted=reason=='admitted',reason=reason,accepted=len(accepted),opportunities=len(rows),
        remaining_work_seconds=remaining,minimum_accepted=r['minimum_accepted'],
        minimum_cycle_remaining_seconds=r['minimum_cycle_remaining_seconds'],
        collection_files={n:sha256(p/n) for n in ('owner_finished.json','result.json','results.json','correction_entries.json')})


def discover_groups(binding, context):
    """Bind descendants before termination, plus inner owners' persisted groups.

    The cycle and batch owner create new sessions, so outer killpg alone is not
    sufficient. Never infer ownership from a command name or a historical PID.
    """
    groups={binding['pid']:binding};identities={}
    for directory in Path('/proc').iterdir():
        if directory.name.isdigit():
            row=process_identity(int(directory.name))
            if row is not None:identities[row['pid']]=row
    live_root=identities.get(binding['pid'])
    descendants=({binding['pid']} if live_root is not None and all(live_root[k]==binding[k]
        for k in ('start_ticks','uid','pgid','sid')) else set())
    while True:
        found={pid for pid,row in identities.items() if row['ppid'] in descendants
            and row['uid']==binding['uid'] and row['start_ticks']>=binding['start_ticks']}
        if found<=descendants:break
        descendants.update(found)
    for pid in descendants:
        row=identities.get(pid)
        if row and row['pgid']==pid and row['sid']==pid:groups[pid]=row
    p=context['paths'];receipts=[p['collection']/'owner_worker_started.json']
    receipts+=list(p['output'].with_name(p['output'].name+'_stages').glob('*_worker.json'))
    for path in receipts:
        if not path.exists():continue
        receipt=read(path);saved=receipt.get('worker_group')
        if saved is None and type(receipt.get('pid')) is int and type(receipt.get('proc_start_ticks')) is int:
            saved=dict(pid=receipt['pid'],pgid=receipt['pid'],sid=receipt['pid'],uid=os.getuid(),
                start_ticks=receipt['proc_start_ticks'])
        if saved is None or saved['start_ticks']<binding['start_ticks']:continue
        live=process_identity(saved['pid'])
        if live is not None and any(live[k]!=saved[k] for k in ('start_ticks','uid','pgid','sid')):continue
        groups[saved['pid']]=saved
    return groups


def run_stage(name, command, context, deadline, stop_paths):
    ledger=context['ledger'];start=time.time();process=None;binding=None;groups={};reason=None;error=None;cleanup=[]
    atomic_json(ledger/'owner_status.json',dict(stage=name,started=start,work_deadline=deadline))
    try:
        with (ledger/(name+'.log')).open('x') as stream:
            process=subprocess.Popen(command,cwd=context['paths']['worker_tree'],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1'))
            binding=bind_worker_group(process.pid);groups[binding['pid']]=binding
            atomic_json(ledger/(name+'_started.json'),dict(command=command,cwd=str(context['paths']['worker_tree']),
                worker_group=binding,started=start,work_deadline=deadline))
            while True:
                groups.update(discover_groups(binding,context))
                if process.poll() is not None:break
                if any(p.exists() for p in stop_paths):reason='requested_stop';break
                if time.time()>=deadline:reason='stage_work_deadline';break
                time.sleep(min(.5,max(.001,deadline-time.time())))
    except BaseException as caught:
        error=repr(caught)
    finally:
        if binding is not None:
            try:groups.update(discover_groups(binding,context))
            except BaseException as caught:error=error or repr(caught)
        # Let inner owners handle their own termination and partial census first.
        # Then clean any independently-sessioned workers they left behind.
        for group in sorted(groups.values(),key=lambda g:(g['pid']!=getattr(process,'pid',None),-g['start_ticks'])):
            try:cleanup.append(cleanup_owned_group(group,grace_seconds=15. if group['pid']==getattr(process,'pid',None) else 3.))
            except BaseException as caught:cleanup.append(dict(binding=group,live_descendants_exited=False,error=repr(caught)))
        if process is not None:
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:error=error or 'Owned leader did not exit after cleanup'
        base=context['paths']['collection'] if name=='collection' else context['paths']['output'].with_name(context['paths']['output'].name+'_stages')
        names=('owner_finished.json','result.json','results.json','correction_entries.json') if name=='collection' else ('finished.json','failure.json','status.json')
        receipt=dict(stage=name,command=command,started=start,ended=time.time(),seconds=time.time()-start,
            returncode=process.returncode if process else None,stop_reason=reason,error=error,cleanup=cleanup,
            result_files={str(base/n):sha256(base/n) for n in names if (base/n).is_file()},
            complete=bool(process is not None and process.returncode==0 and reason is None and error is None
                and all(row['live_descendants_exited'] for row in cleanup)))
        atomic_json(ledger/(name+'_finished.json'),receipt)
    return receipt


def run(registration):
    registration=Path(registration).resolve(strict=True);context=validate(registration);r=context['registration'];p=context['paths']
    ledger=context['ledger'];began=time.time();deadline=r['deadline']-CLEANUP_SECONDS
    collection_deadline=min(began+r['collection_max_seconds']-CLEANUP_SECONDS,
        deadline-r['minimum_cycle_remaining_seconds']-CLEANUP_SECONDS)
    stop_paths=(p['session_root']/'STOP',ledger/'STOP',p['collection']/'STOP',p['output']/'STOP')
    identity=process_identity(os.getpid());handlers={};stages=[];error=None;cycle_state='not_started';reason='owner_interrupted';complete=False
    with (ledger/'owner_started.json').open('x') as stream:
        json.dump(dict(version=VERSION,identity=identity,started=began,registration_sha256=context['registration_sha256'],
            original_deadline=r['deadline'],work_deadline=deadline,collection_work_deadline=collection_deadline,
            cleanup_reserve_seconds=CLEANUP_SECONDS,command_input_sha256=context['command_input_sha256'],
            owner_source_sha256=sha256(__file__),stop_paths=list(map(str,stop_paths))),stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    def interrupted(signum,frame):raise SystemExit(128+signum)
    def unchanged():
        require(sha256(registration)==context['registration_sha256'],'Owner registration changed')
        require(all(sha256(path)==digest for path,digest in context['command_input_sha256'].items()),'Command input changed')
        validate(registration,fresh=False)
        if any(path.exists() for path in stop_paths):raise InterruptedError('Original session or owner STOP')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        unchanged()
        if time.time()>=collection_deadline:reason='insufficient_time_before_collection'
        else:
            stages.append(run_stage('collection',r['collection_command'],context,collection_deadline,stop_paths))
            if not stages[-1]['complete']:reason='collection_incomplete'
            else:
                unchanged();gate=collection_admission(context,time.time());atomic_json(ledger/'cycle_admission.json',gate)
                reason=gate['reason']
                if gate['admitted']:
                    cycle_state='started';stages.append(run_stage('cycle',r['cycle_command'],context,deadline,stop_paths))
                    completion=p['output'].with_name(p['output'].name+'_stages')/'finished.json'
                    complete=stages[-1]['complete'] and completion.exists() and read(completion).get('software_complete') is True
                    stages[-1]['completion_sha256']=sha256(completion) if completion.exists() else None
                    atomic_json(ledger/'cycle_finished.json',stages[-1]);cycle_state='complete' if complete else 'incomplete'
                    reason='complete' if complete else 'cycle_incomplete'
    except BaseException as caught:error=repr(caught);reason='owner_exception'
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        try:atomic_json(ledger/'partial_census.json',census(p['collection'],context['plan']))
        except BaseException as caught:error=error or repr(caught);complete=False
        result=dict(version=VERSION,software_complete=complete,cycle_state=cycle_state,reason=reason,error=error,
            started=began,ended=time.time(),seconds=time.time()-began,original_deadline=r['deadline'],
            registration_sha256=context['registration_sha256'],stages=stages)
        atomic_json(ledger/'owner_finished.json',result)
        atomic_json(ledger/'owner_status.json',dict(stage=cycle_state,reason=reason,software_complete=complete,error=error))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['validate','run'])
    parser.add_argument('--registration',type=Path,required=True);args=parser.parse_args()
    if args.mode=='validate':
        context=validate(args.registration)
        print(json.dumps(dict(valid=True,registration_sha256=context['registration_sha256'],deadline=context['registration']['deadline'])))
    else:
        result=run(args.registration);print(json.dumps(result),flush=True)
        if not result['software_complete']:raise SystemExit(1)


if __name__=='__main__':main()
