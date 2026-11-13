"""External event monitor and real Codex re-entry; never edits frozen workers.

The initial owner is parked only while its final evaluation child runs, then
terminated after that child exits. This leaves both full-policy cohorts intact
and hands the next scientific choice to an actual agent, not an if-success rule.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def process(pid):
    try:
        f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),state=f[0],ppid=int(f[1]),start_ticks=int(f[19]))
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(identity):
    now=process(identity['pid']) if identity else None
    return now if now and now['start_ticks']==identity['start_ticks'] else None


def running(identity):
    value=same_process(identity)
    return value is not None and value['state'] not in {'Z','X'}


def notify(root,stage,**kw):
    write(root/'status.json',dict(stage=stage,utc=time.time(),pid=os.getpid(),**kw))


def validate_watch(plan,outer):
    path=Path(plan['root']).resolve()
    if not path.is_relative_to(outer.resolve()):raise ValueError('Watch root must stay inside this research run')
    if not plan['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in plan['id']):
        raise ValueError('Invalid watch identity')
    for name in [*plan.get('completion_files',[]),plan.get('status_file','follow_status.json'),plan.get('launch_file','follow_launch.json')]:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Completion paths must be relative')
    identity=plan.get('owner_identity')
    if identity and (type(identity.get('pid')) is not int or identity['pid']<=1 or type(identity.get('start_ticks')) is not int or identity['start_ticks']<1):
        raise ValueError('Watch owner requires a real PID and start_ticks')
    return path


def progress_signature(root):
    """Progress artifacts only; resource telemetry does not count as learning."""
    patterns=['**/training/progress.json','**/training/loss.jsonl','**/training/validation.json',
        '**/evaluation_progress.json','**/evaluation_summary.json','**/collection_results.json',
        '**/cache_progress.json','**/offline/progress.json','*_receipt.json','**/stages/*.json']
    rows=[]
    for pattern in patterns:
        for p in root.glob(pattern):
            if p.is_file() and not p.is_symlink():
                stat=p.stat();rows.append((str(p.relative_to(root)),stat.st_size,stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(sorted(set(rows))).encode()).hexdigest()


def choose_event(plan,root,handled,now,last_progress):
    for name in plan.get('completion_files',[]):
        p=root/name
        if p.exists():
            key=plan['id']+':'+name+':'+hashlib.sha256(p.read_bytes()).hexdigest()
            if key not in handled:return dict(key=key,kind='stage_result',path=str(p))
    status=read(root/plan.get('status_file','follow_status.json'),{})
    owner=read(root/plan.get('launch_file','follow_launch.json'),{})
    expected=plan.get('owner_identity')
    owner_gone=not running(expected) if expected else bool(owner and process(owner['pid']) is None)
    if owner_gone:
        key=plan['id']+':owner_exit:'+str((expected or owner).get('start_ticks',owner.get('utc',owner.get('pid'))))
        if key not in handled:return dict(key=key,kind='unexpected_owner_exit',status=status)
    if now-last_progress>=plan.get('stall_seconds',900):
        active=read(root/'active_stage.json',{})
        key=plan['id']+':no_progress:'+str(active.get('pid'))+':'+str(last_progress)
        if key not in handled:return dict(key=key,kind='suspected_no_progress',status=status,
            seconds=now-last_progress,stage=active,warning='Check live resources/CPU/phase before declaring a hang; do not kill blindly')
    return None


def park_initial_owner(supervisor,root,deadline):
    """Pause only the Python coordinator, never its running evaluation child."""
    ledger=supervisor/'initial_boundary.json';record=read(ledger)
    if record and record.get('complete'):return dict(key='initial_full_policy_boundary',kind='full_policy_boundary',record=record)
    if record is None:
        active=read(root/'active_stage.json',{});launch=read(root/'follow_launch.json',{})
        if active.get('name')!='G01_formal_eval' or not launch:return None
        expected=read(supervisor/'registration.json')['initial_owner']
        if launch['pid']!=expected['pid']:raise ValueError('Initial owner identity changed')
        owner=same_process(expected);worker=active.get('process')
        if not owner or not running(worker):return None
        live=same_process(worker)
        if live['ppid']!=owner['pid']:raise ValueError('Final evaluation worker is not owned by the registered coordinator')
        if not read(root/'formal/QG01/backend/evaluation_complete.json',{}).get('complete'):raise ValueError('A evaluation is incomplete')
        if not (root/'formal/G01/backend/G01/offline/result.json').exists():raise ValueError('B offline prerequisite missing')
        record=dict(owner=owner,worker=worker,stage_deadline=active['deadline'],parked_utc=time.time(),
            reason='User requests agent research judgment after complete current full-policy pair; automatic targeted rounds superseded')
        # A vanished/reused PID is never signalled.
        if not running(owner):return None
        # Persist identity first so shutdown can resume the owner even if this
        # supervisor is interrupted immediately after delivering SIGSTOP.
        write(ledger,record);os.kill(owner['pid'],signal.SIGSTOP)
        notify(supervisor,'current_full_policy_evaluation_continues',boundary=record)
    if running(record['worker']) and time.time()<min(deadline,record['stage_deadline']):return None
    if 'owner_stop_requested_utc' not in record:
        if same_process(record['owner']):
            os.kill(record['owner']['pid'],signal.SIGTERM)
            os.kill(record['owner']['pid'],signal.SIGCONT)
        record['owner_stop_requested_utc']=time.time();write(ledger,record)
    if running(record['owner']):return None
    record.update(complete=True,completed_utc=time.time(),evaluation_receipts={
        g:read(root/'formal'/g/'backend/evaluation_complete.json',{}) for g in ('QG01','G01')},
        summaries={g:read(root/'formal'/g/'backend'/g/'evaluation_summary.json',{}) for g in ('QG01','G01')},
        note='Coordinator SIGTERM was intentional after its last evaluation child; preserve its InterruptedError evidence')
    write(ledger,record)
    return dict(key='initial_full_policy_boundary',kind='full_policy_boundary',record=record)


def codex_runtime(reg):
    executable=reg['codex_executable']
    if not Path(executable).is_absolute() or not os.access(executable,os.X_OK):raise ValueError('Verified absolute Codex executable required')
    env=dict(os.environ,PATH=str(Path(executable).parent)+os.pathsep+os.environ.get('PATH','/usr/bin:/bin'))
    return executable,env


def agent_turn(root,event,reg,index):
    directory=root/f'agent_{index:03d}';directory.mkdir(exist_ok=False)
    write(directory/'event.json',event)
    prompt=(root/'agent_prompt.txt').read_text().replace('{EVENT}',str(directory/'event.json')).replace('{REPLY}',str(directory/'decision.json'))
    executable,env=codex_runtime(reg)
    args=[executable,'exec','--json','--sandbox','danger-full-access','-c','approval_policy="never"',
          '--output-last-message',str(directory/'reply.txt'),'-']
    # Same local provider/model/auth configuration as the successful preflight;
    # no credential copying, disabled safety guard or external notification.
    for attempt in (1,2):
        notify(root,'agent_research_active',event=event['kind'],agent_directory=str(directory),transport_attempt=attempt)
        with (directory/f'events_{attempt}.jsonl').open('w') as log,(directory/f'stderr_{attempt}.log').open('w') as err:
            child=subprocess.Popen(args,cwd=reg['code_root'],env=env,stdin=subprocess.PIPE,stdout=log,stderr=err,text=True,start_new_session=True)
            write(directory/f'launch_{attempt}.json',dict(pid=child.pid,process=process(child.pid),utc=time.time()))
            child.stdin.write(prompt);child.stdin.close()
            # Ordinary scripts may continue work; this bounds one reasoning turn
            # and the original absolute session, without extending experiments.
            until=min(reg['absolute_deadline']-60,time.time()+2700)
            while child.poll() is None and time.time()<until:time.sleep(2)
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            code=child.returncode
        decision=read(directory/'decision.json')
        write(directory/f'exit_{attempt}.json',dict(code=code,utc=time.time(),decision_present=decision is not None))
        if code==0 and decision is not None:return decision
        events=(directory/f'events_{attempt}.jsonl').read_text(errors='replace')
        transient=any(word in events.lower() for word in ['stream disconnected','rate limit','429','503','connection reset','transport error'])
        if not transient or attempt==2:break
        # Do not replay work invisibly. The new agent receives the previous log.
        prompt+='\nPrevious invocation may have partially acted. Inspect '+str(directory)+' before doing anything; never repeat completed parameter updates or trials.\n'
        time.sleep(15)
    write(root/'agent_unavailable.json',dict(directory=str(directory),utc=time.time(),reason='Agent process failed, timed out or omitted a machine-readable continuation decision'))
    return None


def run(root):
    reg=read(root/'registration.json');outer=Path(reg['outer_root']);handled=read(root/'handled.json',[])
    parked=None;last_signature=None;last_progress=time.time();plan_sha=None;index=len(list(root.glob('agent_[0-9][0-9][0-9]')))
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        executable,env=codex_runtime(reg)
        probe=subprocess.run([executable,'--version'],env=env,capture_output=True,text=True,timeout=20,check=True)
        write(root/'runtime_preflight.json',dict(utc=time.time(),version=probe.stdout.strip(),verified_executable=executable,
            scope='Actual service environment and executable only; model/tool access verified separately'))
        write(root/'launch.json',dict(pid=os.getpid(),process=process(os.getpid()),utc=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        try:
            while time.time()<reg['absolute_deadline']-90:
                plan=read(root/'watch_plan.json');watched=validate_watch(plan,outer)
                signature=progress_signature(watched);new_plan_sha=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
                if signature!=last_signature or new_plan_sha!=plan_sha:last_progress=time.time();last_signature=signature;plan_sha=new_plan_sha
                event=None
                if plan.get('initial_boundary'):
                    event=park_initial_owner(root,watched,reg['compute_deadline'])
                    parked=read(root/'initial_boundary.json')
                if event is None and not (parked and not parked.get('complete')):
                    event=choose_event(plan,watched,handled,time.time(),last_progress)
                if event and event['key'] not in handled:
                    write(root/'pending_event.json',event);index+=1
                    decision=agent_turn(root,event,reg,index)
                    if decision is None:
                        notify(root,'AGENT_INVOCATION_REQUIRED',event=event,reason='See agent_unavailable.json; no claim of autonomous reasoning');return
                    handled.append(event['key']);write(root/'handled.json',handled);write(root/'latest_decision.json',decision)
                    if decision.get('next_watch'):
                        validate_watch(decision['next_watch'],outer);write(root/'watch_plan.json',decision['next_watch']);parked=None
                    elif decision.get('status') in ('complete','supported_bottleneck','external_block'):
                        notify(root,'research_agent_resolved',decision=decision);return
                    else:
                        notify(root,'AGENT_INVOCATION_REQUIRED',reason='Agent returned neither a next job nor a concrete terminal research conclusion',decision=decision);return
                notify(root,'watching_actual_experiment',watch=plan,last_progress_utc=last_progress,
                    current_stage=read(watched/plan.get('status_file','follow_status.json'),{}))
                time.sleep(5)
            notify(root,'original_absolute_deadline_reached',compute_deadline=reg['compute_deadline'],absolute_deadline=reg['absolute_deadline'])
        finally:
            # Never leave the coordinator stopped if this external supervisor exits.
            parked=read(root/'initial_boundary.json')
            if parked and not parked.get('complete') and same_process(parked['owner']):
                os.kill(parked['owner']['pid'],signal.SIGCONT)
            write(root/'exit.json',dict(utc=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Supervisor stop')))
    try:run(args.root.resolve())
    except BaseException:
        write(args.root/'supervisor_error.json',dict(utc=time.time(),traceback=traceback.format_exc()));raise
