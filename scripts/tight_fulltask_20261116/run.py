"""Finite deadline-bound full task continuation. No automatic DAgger loop."""
import copy,json,os,subprocess,sys,time,traceback,threading
from pathlib import Path
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read
from scripts.tight_stability_20261116.run import ROOT as DIAG
from scripts.tight_stability_20261116.analyze_existing import TREE,OLD
from visuotactile.compact import sha256
ROOT=TREE/'outputs/visuotactile/run/tight_fulltask_20261116_01'
DEADLINE=1794814200.0 # 2026-11-16 15:30 UTC+8
STATUS_LOCK=threading.Lock()

def prepare():
    from tools.tight_fresh_batch_run import prepare as batch
    now=time.time();assert now<DEADLINE;ROOT.mkdir()
    session=read(OLD/'session.json');session.update(started_utc=now,deadline=DEADLINE,training_deadline=DEADLINE,
        offline_deadline=DEADLINE,cache_deadline=DEADLINE,resource_root=str(ROOT.resolve()),
        owned_worker_roots=[str(ROOT.resolve())],authorization='Full-task paired data, training, videos and evaluation; hard stop 2026-11-16 15:30 Singapore')
    atomic_json(ROOT/'session.json',session)
    for name in ['metadata.json','gate.json']:(ROOT/name).symlink_to(OLD/name)
    for role in 'AB':
        (ROOT/role).mkdir()
        for name in ['state_data','cache_index.json']:(ROOT/role/name).symlink_to(OLD/role/name)
    data=read(OLD/'A/state_data/data.json');ids=data['normalization_train_ids'][:2]
    template=read(DIAG/'candidates/plan.json')
    plan=batch(OLD,ROOT/'collection',lanes=4,identities=ids,delays=[0,0],round_index=20,
        seed=28600100,session_root=ROOT,execution_backend=template['execution_backend'],queued_gpu=True,
        verification_workers=2,verification_max_pending=2,capture_audit_memo=True,vision_microbatch=2,
        vision_tail_profile=template['batch'].get('vision_tail_profile'),inactive_render_profile=template['batch'].get('inactive_render_profile'))
    plan['pairs']=template['pairs'];plan['source_sha256'].update({str(p):sha256(p) for p in Path('scripts/tight_fulltask_20261116').glob('*.py')})
    atomic_json(ROOT/'collection/plan.json',plan)
    atomic_json(ROOT/'registration.json',dict(deadline=DEADLINE,started=now,pairs=plan['pairs'],families=ids,
        collection='2 original train families x A/B, real contact+height trigger, no fixed fallback',
        common_data='All accepted A/B corrections pooled for both roles; existing 348 training records retained',
        training_epochs=1,rear_epochs=1,latest_repeat=12,reason='Bounded full-pool refresh under hard wall-clock deadline, not convergence claim',
        optimizer='Fresh AdamW/EMA from selected frontend EMA; rear initialized from selected matching rear; no strict optimizer-resume claim',
        history_asymmetry='A selected DP14/rear22; B DP23/rear6; same incremental budget, different historical exposure',
        evaluation_seeds=[28500100,28500101,28500105,28500106],full_native_termination=True))

def execute(name,args):
    assert time.time()<DEADLINE
    with STATUS_LOCK:
        atomic_json(ROOT/'status.json',dict(stage=name,utc=time.time(),deadline=DEADLINE))
    with (ROOT/(name+'.log')).open('x') as f:
        p=subprocess.Popen([sys.executable,'-m']+list(map(str,args)),cwd=TREE,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),start_new_session=True)
        row=dict(stage=name,pid=p.pid,started=time.time(),command=p.args)
        while p.poll() is None:
            row['heartbeat']=time.time();atomic_json(ROOT/(name+'_worker.json'),row)
            if time.time()>=DEADLINE-20 or (ROOT/'STOP').exists():
                os.killpg(p.pid,15)
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:os.killpg(p.pid,9);p.wait()
                raise TimeoutError('Hard deadline/STOP; current incomplete output retained')
            time.sleep(3)
        row.update(ended=time.time(),exit_code=p.returncode,seconds=time.time()-row['started']);atomic_json(ROOT/(name+'_worker.json'),row)
        if p.returncode:raise RuntimeError(name+' failed')

def monitor_start():
    return subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],cwd=TREE,
        stdout=(ROOT/'monitor.log').open('x'),stderr=subprocess.STDOUT)

def collect():
    monitor=monitor_start()
    try:
        until=time.time()+15
        while not (ROOT/'shared_resources.json').exists():
            assert monitor.poll() is None and time.time()<until;time.sleep(.2)
        execute('collection',['tools.tight_fresh_batch_run','run','--output',ROOT/'collection'])
        atomic_json(ROOT/'collection_owner_complete.json',dict(utc=time.time(),complete=True))
    finally:
        (ROOT/'STOP_MONITOR').touch();monitor.wait(timeout=20)

if __name__=='__main__':
    try: {'prepare':prepare,'collect':collect}[sys.argv[1]]()
    except BaseException:
        if ROOT.exists():atomic_json(ROOT/'failure.json',dict(utc=time.time(),error=traceback.format_exc()))
        raise
