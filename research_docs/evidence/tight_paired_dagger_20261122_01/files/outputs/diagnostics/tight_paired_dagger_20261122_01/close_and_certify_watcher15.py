"""Root-only closure: read lifecycle evidence; write only owned marker/certificate.

No signals or process termination. Historical missing birth snapshots are handled
by durable PID/start/exit windows plus a fresh exact-owned-process/cgroup census.
No policy results, model payloads or scientific summaries are opened.
"""
from pathlib import Path
import argparse,base64,hashlib,json,math,os,stat,subprocess,time
import psutil
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS')
A=BASE/'outputs/diagnostics/tight_paired_dagger_20261122_01'
R=BASE/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
OWNER=R/'owners/test_continuation15'
DRIVER=A/'driver_test_continuation15'
WATCH_UNIT='tight-paired-watch-continuation15.service'
DRIVER_UNIT='tight-paired-test-continuation15.service'
MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_continuation15'
WATCH_MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.shared_capacity15'
MARKER_TEXT=b'Root verified continuation15 driver/native/helpers closed; stop only owned watcher15.\n'


def require(value,message):
    if not value:raise ValueError(message)


def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>0


def canonical(path,root,*,present=False,directory=False):
    path,root=Path(path),Path(root)
    require(path.is_absolute() and root.is_absolute() and path==path.resolve(strict=False)
        and root==root.resolve(strict=True) and path.is_relative_to(root),'Noncanonical or escaped path: '+str(path))
    for part in [path,*path.parents]:
        if part==root.parent:break
        require(not part.is_symlink(),'Symlink path refused: '+str(part))
    if present:require(path.is_dir() if directory else path.is_file(),'Missing regular path: '+str(path))
    return path


def read(path,root=None):
    path=canonical(path,root or A,present=True);require(path.stat().st_size<=8*2**20,'Large lifecycle metadata refused')
    return json.loads(path.read_bytes())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def exclusive_bytes(path,raw,root):
    path=canonical(path,root);canonical(path.parent,root,present=True,directory=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    sync_dir(path.parent)


def write_new(path,row):exclusive_bytes(path,json.dumps(row,indent=2,allow_nan=False).encode(),A)


def absent(row,*,lookup=None):
    require(type(row.get('pid')) is int and row['pid']>0 and finite(row.get('created')),'Exact positive PID/birth required')
    lookup=psutil.Process if lookup is None else lookup
    try:p=lookup(row['pid'])
    except psutil.NoSuchProcess:return True
    return abs(p.create_time()-row['created'])>.001


def closed_window(pid,start,end,*,lookup=None):
    require(type(pid) is int and pid>0 and finite(start) and finite(end) and start<=end,'Positive PID and finite lifecycle window required')
    lookup=psutil.Process if lookup is None else lookup
    try:p=lookup(pid)
    except psutil.NoSuchProcess:return dict(pid=pid,birth_source='not_captured',pid_absent_now=True,start_utc=start,end_utc=end)
    birth=p.create_time()
    require(finite(birth) and birth>end,'Historical PID remains live without proven later reuse: '+str(pid))
    return dict(pid=pid,birth_source='not_captured',pid_absent_now=False,later_reused_birth=birth,start_utc=start,end_utc=end)


def service(name):
    fields=['MainPID','Result','ActiveState','ControlGroup','InvocationID']
    raw=subprocess.check_output(['systemctl','--user','show',name,*[arg for key in fields for arg in ('-p',key)]],text=True,timeout=10)
    row=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    require(set(fields)<=set(row),'Incomplete systemd service census')
    return raw,row


def closed_service(row):return row['MainPID']=='0' and row['Result']=='success' and row['ActiveState']=='inactive'


def cgroup_empty(row):
    value=row.get('ControlGroup','')
    if not value:return dict(control_group=value,pids=[],scope='Inactive service has no remaining cgroup')
    require(value.startswith('/') and '..' not in Path(value).parts,'Unsafe service cgroup')
    root=Path('/sys/fs/cgroup');path=root/value.lstrip('/')
    if not path.exists():return dict(control_group=value,pids=[],scope='Inactive cgroup no longer exists')
    canonical(path,root,present=True,directory=True);pids=[]
    for file in [path/'cgroup.procs',*path.rglob('*/cgroup.procs')]:
        canonical(file,root,present=True);pids.extend(int(v) for v in file.read_text().split())
    require(not pids,'Owned systemd cgroup still contains processes')
    return dict(control_group=value,pids=pids)


def own_command(command):
    """Literal modules/paths only; external tests and generic project names excluded."""
    if not isinstance(command,list) or not command:return False
    args=command[1:]
    if len(args)>=2 and args[:2] in (['-m',MODULE],['-m','visuotactile.state_resources']):
        return args[:2]==['-m',MODULE] or args[2:]==['--root',str(OWNER)]
    if len(args)>=2 and args[:2]==['-m','tools.tight_paired_evaluate']:
        return '--output' in args and args[args.index('--output')+1:args.index('--output')+2]==[str(R/'evaluation/test')]
    if str(A/'audit_collection_recovery05.py') in args:
        return '--collection' in args and args[args.index('--collection')+1:args.index('--collection')+2]==[str(R/'collection')]
    if str(A/'root_monitor_continuation15.py') in args:return True
    return False


def own_census(*,processes=None):
    rows=[];errors=[]
    processes=psutil.process_iter(['pid','cmdline','uids']) if processes is None else processes
    for process in processes:
        try:
            if process.pid==os.getpid():continue
            info=process.info
            if info.get('uids') is not None and info['uids'].real!=os.getuid():continue
            command=info.get('cmdline') or []
            if own_command(command):rows.append(dict(pid=process.pid,created=process.create_time(),command=command))
        except psutil.NoSuchProcess:continue
        except (psutil.AccessDenied,psutil.ZombieProcess,OSError,ValueError) as error:errors.append(dict(pid=process.pid,error=repr(error)))
    require(not errors,'Owned process census incomplete: '+repr(errors))
    require(not rows,'Owned driver/native/helper still live: '+repr(rows))
    return dict(utc=time.time(),matching_processes=rows,errors=errors,signals=0,
        scope='Exact continuation15 modules/native monitor/original two tail helpers/root lifecycle monitor; current watcher excluded')


def lifecycle():
    """No scientific results. Read durable owner/helper/process lifecycle only."""
    pins={}
    def get(path):
        row=read(path,R if Path(path).is_relative_to(R) else A);pins[str(path)]=sha(path);return row
    activation=get(A/'TEST_CONTINUATION15_ACTIVATED.json');reg=get(A/'TEST_CONTINUATION15_REGISTRATION.json')
    require(activation['registry_sha256']==sha(A/'TEST_CONTINUATION15_REGISTRATION.json')
        and activation['root_review_sha256']==sha(A/'ROOT_TEST_CONTINUATION15_REVIEW.json'),'Activated source/review binding changed')
    driver_actor=activation['actors'][DRIVER_UNIT];watch_actor=activation['actors'][WATCH_UNIT]
    for actor in (driver_actor,watch_actor):
        require(type(actor.get('pid')) is int and actor['pid']>0 and finite(actor.get('created')),'Activated exact PID/birth required')
        require(actor.get('cwd')==str(BASE),'Activated canonical cwd required')
    require(driver_actor.get('command',[])[-1:] == ['driver'] and MODULE in driver_actor['command']
        and WATCH_MODULE in watch_actor.get('command',[]),'Activated driver/watcher command binding differs')
    require(absent(driver_actor),'Activated driver remains live')
    d=get(DRIVER/'driver_exit.json');closure=get(DRIVER/'closure.json');activated=get(DRIVER/'registration.json')
    require(d.get('pid')==driver_actor['pid'] and d.get('returned_after_owned_helpers') is True
        and closure.get('status')=='original_missing47_queue_and_audits_complete'
        and activated.get('activated') is True and activated['activated_utc']<=closure['utc']<=d['utc'],
        'Actual driver exit/closure binding required')
    begin=get(OWNER/'project_job_started.json');end=get(OWNER/'project_exit.json');done=get(OWNER/'project_complete.json')
    monitor=get(OWNER/'monitor.json');monitor_exit=get(OWNER/'monitor_exit.json');status=get(OWNER/'project_status.json')
    require(end.get('children_reaped') is True and end.get('worker_reaped') is True and end.get('monitor_reaped') is True
        and done.get('status')=='finite47_queue_consumed_all' and done.get('stages')==47 and done.get('all_children_reaped') is True
        and begin['utc']<=done['utc']<=end['utc']<=closure['utc']<=d['utc'],'Finite47 native owner reaping required')
    require(monitor_exit.get('pid')==monitor['pid'],'Native monitor exit identity differs')
    windows=[closed_window(begin['pid'],begin['utc'],end['utc']),
             closed_window(monitor['pid'],monitor['started'],monitor_exit['ended'])]
    require(status.get('stage')==46,'Final native stage status required')
    indices=list(range(17,64));stage_dirs={name:OWNER/name for name in ('launches','stage_exits','dispatches')}
    for name in ('launches','stage_exits'):
        canonical(stage_dirs[name],R,present=True,directory=True)
        require({p.name for p in stage_dirs[name].iterdir()}=={f'wave{i:02d}.json' for i in indices},'Exact47 '+name+' required')
    canonical(stage_dirs['dispatches'],R,present=True,directory=True)
    require({p.name for p in stage_dirs['dispatches'].iterdir()}<={f'wave{i:02d}.json' for i in indices},
            'Unknown worker dispatch identity')
    previous=begin['utc']
    for stage,index in enumerate(indices):
        launch=get(stage_dirs['launches']/f'wave{index:02d}.json');ex=get(stage_dirs['stage_exits']/f'wave{index:02d}.json')
        require(launch.get('index')==ex.get('index')==index and launch.get('stage')==ex.get('stage')==stage
            and ex.get('child_reaped') is True and ex.get('process_started') is True
            and ex.get('queue_aborted') is False and type(ex.get('returncode')) is int
            and previous<=launch['utc']<=ex['utc']<=done['utc'],'Every consumed worker must have durable actual reaping')
        fence=stage_dirs['dispatches']/f'wave{index:02d}.json'
        if fence.exists():
            row=get(fence);require(row.get('index')==index and row.get('dispatch_attempt')==1 and row.get('rerun') is False,
                                    'Worker dispatch fence differs')
            windows.append(closed_window(row['pid'],row['utc'],ex['utc']))
            if stage==46:require(row['pid']==status['pid'],'Last native worker PID differs')
        previous=ex['utc']
    windows.append(closed_window(status['pid'],status['utc'],previous))
    require(previous<=end['utc'],'Last worker exit after owner exit')
    # Missing historical helper birth snapshots are explicit; rc0 receipts plus
    # closed driver service/cgroup and live exact helper census certify closure.
    helpers=[]
    require([tail.get('name') for tail in reg.get('tail_stages',[])]==
        ['summarize_test','audit_collection_cuda_noise_recovery05'],'Exact original two tail helpers required')
    for tail in reg['tail_stages']:
        name=tail['name'];directory=R/'cycle_logs_recovery12' if tail['skip_closed'] else R/'cycle_logs_continuation15'
        ex=get(directory/(name+'_exit.json'));require(ex.get('returncode')==0 and ex['utc']<=closure['utc'],'Actual tail helper failed/unclosed')
        helpers.append(dict(name=name,exit_receipt=str(directory/(name+'_exit.json')),exit_utc=ex['utc'],
            birth_source='not_captured',proof='durable reaped-helper rc0; inactive driver cgroup; fresh exact helper census'))
    sampled=get(A/'ROOT_LIVE15_IDENTITIES.json').get('identities',{})
    require(isinstance(sampled,dict),'Historical identity snapshot schema invalid')
    for row in sampled.values():require(absent(row),'Sampled actual own PID/birth remains live')
    raw,svc=service(DRIVER_UNIT);require(closed_service(svc),'Actual driver service must be inactive/MainPID0/success')
    census=own_census();cgroup=cgroup_empty(svc)
    return dict(activation=activation,watch_actor=watch_actor,pins=pins,closed_windows=windows,
        sampled_exact_births=list(sampled.values()),helpers=helpers,driver_service=svc,driver_service_text=raw,
        driver_cgroup=cgroup,owned_process_census=census,historical_birth_limit='Only retained exact birth snapshots are birth-matched; other historical windows prove PID absent or later reuse, plus fresh exact-owned helper/cgroup census.')


def restored_backups(actor):
    W=canonical(A/'watch_shared_capacity15',A,present=True,directory=True)
    exitpath=W/'resource_watch_v4_exit.json';ex=read(exitpath)
    require(ex.get('pid')==actor['pid'] and ex.get('conflicts')==[] and ex.get('modified_schedules')==0
        and ex.get('exact_original_byte_restoration') is True and ex.get('no_process_signals') is True,'Actual watcher exit/restoration required')
    dirs=list(W.glob('resource_watch_v4_backups_*'))
    require(len(dirs)==1 and dirs[0].name.endswith('_'+str(actor['pid'])),'Exact single dynamic backup directory required')
    directory=canonical(dirs[0],A,present=True,directory=True);restored=[]
    for p in sorted(directory.iterdir()):
        canonical(p,A,present=True);row=read(p);target=canonical(Path(row['path']),R)
        require(target.name=='concurrency.json' and p.name==hashlib.sha256(str(target).encode()).hexdigest()+'.json',
                'Exact owned schedule backup identity required')
        require(type(row.get('existed')) is bool,'Boolean original existence required')
        if row['existed']:
            raw=base64.b64decode(row['base64'],validate=True)
            require(hashlib.sha256(raw).hexdigest()==row['sha256'] and target.is_file() and target.read_bytes()==raw,'Original schedule bytes not restored')
        else:require(row.get('base64') is None and row.get('sha256') is None and not target.exists(),'Original schedule absence not restored')
        restored.append(dict(path=str(target),existed=row['existed'],sha256=row['sha256'],backup_sha256=sha(p)))
    require(restored and len({x['path'] for x in restored})==len(restored),'Empty/duplicate dynamic restoration proof')
    return exitpath,restored


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stop','certify']);args=parser.parse_args()
    proof=lifecycle();actor=proof['watch_actor'];marker=R/'STOP_RESOURCE_WATCH15';intent=A/'ROOT_WATCHER15_STOP_INTENT.json'
    if args.mode=='stop':
        require(not intent.exists() and not intent.is_symlink(),'Preserve attempted stop intent')
        require(not absent(actor),'Exact own watcher must remain live before marker stop')
        process=psutil.Process(actor['pid'])
        require(abs(process.create_time()-actor['created'])<=.001
            and process.cmdline()==actor['command'] and process.cwd()==str(BASE),'Exact live watcher birth/command/cwd required')
        exclusive_bytes(marker,MARKER_TEXT,R);metadata=marker.stat()
        write_new(intent,dict(utc=time.time(),marker=str(marker),sha256=sha(marker),device=metadata.st_dev,inode=metadata.st_ino,
            watcher_pid=actor['pid'],watcher_created=actor['created'],proof=proof,script_sha256=sha(Path(__file__))))
        print('Own watcher15 stop marker written durably; no process signals sent.');return
    stop=read(intent);canonical(marker,R,present=True);metadata=marker.stat()
    require(marker.read_bytes()==MARKER_TEXT and sha(marker)==stop['sha256'] and metadata.st_dev==stop['device']
        and metadata.st_ino==stop['inode'] and stop['watcher_pid']==actor['pid'] and stop['watcher_created']==actor['created'],
        'Exact root-created stop marker identity required')
    require(absent(actor),'Own watcher PID/birth remains live')
    text,svc=service(WATCH_UNIT);require(closed_service(svc),'Own watcher service must be inactive/MainPID0/success')
    cgroup=cgroup_empty(svc);exitpath,restored=restored_backups(actor)
    receipt=dict(utc=time.time(),pid=actor['pid'],created=actor['created'],source_pid_reaped=True,
        service_name=WATCH_UNIT,service=svc,service_text=text,watcher_cgroup=cgroup,exit_sha256=sha(exitpath),
        restored_verified=restored,actual_restoration_identities=len(restored),conflicts=[],stop_marker_preserved=str(marker),
        stop_intent_sha256=sha(intent),driver_and_owned_native_actors_reaped=True,lifecycle_reaping_proof=proof,
        no_external_signals=True,script_sha256=sha(Path(__file__)))
    write_new(A/'WATCHER15_ACTUAL_CLOSED.json',receipt)
    print(json.dumps(dict(status='actual_watcher15_closed',restored=len(restored))))

if __name__=='__main__':main()
