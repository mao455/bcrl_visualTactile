"""One pre-action evaluation recovery; preserve training and original deadline."""
import os,subprocess,sys,time,traceback
from . import control as c
from scripts.tight_layout_20261116.pilot import stop
ORIGINAL=c.ROOT
ROOT=ORIGINAL/'evaluation_recovery01'

def prepare():
    assert c.read(ORIGINAL/'owner_exit.json')['children_reaped']
    assert not list((ORIGINAL/'comparison00').glob('*/numeric.h5'))
    assert c.read(ORIGINAL/'A/training/finished.json')['updates']==494
    assert c.read(ORIGINAL/'A/backend/A/training/finished.json')['updates']==369
    ROOT.mkdir(exist_ok=False);(ROOT/'logs').mkdir()
    for name in ['A','B','registration.json','prepared.json']:(ROOT/name).symlink_to(ORIGINAL/name,target_is_directory=name in ['A','B'])
    session=c.read(ORIGINAL/'session.json');session.update(resource_root=str(ROOT),owned_worker_roots=[str(ROOT)])
    c.write(ROOT/'session.json',session)
    c.write(ROOT/'recovery.json',dict(original=str(ORIGINAL),reason='Launcher omitted vendored rl_games PYTHONPATH; site package lacks _load_checkpoint_weights. Import failed before policy creation/reset/actions; Kit teardown hung. Preserve failed root, no task result counted.',
        change='Set PYTHONPATH to existing vendored rl_games before startup; no source/network/control/weights change.',
        original_deadline=session['deadline'],training_repeated=False,utc=time.time()))

def worker(part):
    c.ROOT=ROOT;c.evaluation_worker(part)

def run():
    assert not (ROOT/'owner_started.json').exists()
    c.write(ROOT/'owner_started.json',dict(pid=os.getpid(),utc=time.time()));monitor=child=None
    try:
        with (ROOT/'monitor.log').open('x') as f:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.time()+30
        while not (ROOT/'shared_resources.json').exists():
            assert monitor.poll() is None and time.time()<until;time.sleep(.25)
        c.ROOT=ROOT
        for part in range(4):
            c.evaluation_prepare(part);start=time.time();log=ROOT/'logs'/f'eval{part}.log'
            cmd=[sys.executable,'-m',__spec__.name,'worker',str(part),'--root',str(ROOT)]
            with log.open('x') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            c.write(ROOT/'logs'/f'eval{part}_started.json',dict(command=cmd,pid=child.pid,utc=start))
            while child.poll() is None:
                assert monitor.poll() is None
                if time.time()>c.read(ROOT/'session.json')['deadline']-30 or (ROOT/'STOP').exists():raise TimeoutError('Original deadline/explicit stop')
                # A Python traceback with no result is a failed attempt, not a
                # reason to spend the remaining deadline in Kit destruction.
                if time.time()-start>30 and 'Traceback (most recent call last)' in log.read_text(errors='replace'):
                    raise RuntimeError('Evaluation traceback; stop owned Kit, preserve evidence')
                c.write(ROOT/'status.json',dict(stage=f'eval{part}',pid=child.pid,elapsed_s=time.time()-start,utc=time.time()));time.sleep(3)
            assert child.returncode==0 and (ROOT/f'comparison{part:02d}/result.json').exists()
            c.write(ROOT/'logs'/f'eval{part}_complete.json',dict(seconds=time.time()-start,utc=time.time()));child=None
        c.write(ROOT/'complete.json',dict(complete=True,utc=time.time(),no_training=True))
    except BaseException:
        c.write(ROOT/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
    finally:
        stop(child);(ROOT/'STOP_MONITOR').touch()
        if monitor:
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:stop(monitor)
        c.write(ROOT/'owner_exit.json',dict(children_reaped=True,utc=time.time()))

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2])
    else:globals()[sys.argv[1]]()
